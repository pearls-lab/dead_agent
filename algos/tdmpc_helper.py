# Offical implementation ffrom https://github.com/nicklashansen/tdmpc/blob/main/src/algorithm/helper.py
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


__REDUCE__ = lambda b: 'mean' if b else 'none'


def l1(pred, target, reduce=False):
	"""Computes the L1-loss between predictions and targets."""
	return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))


def mse(pred, target, reduce=False):
	"""Computes the MSE loss between predictions and targets."""
	return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))


def _get_out_shape(in_shape, layers):
	"""Utility function. Returns the output shape of a network for a given input shape."""
	x = torch.randn(*in_shape).unsqueeze(0)
	return (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x).squeeze(0).shape


def orthogonal_init(m):
	"""Orthogonal layer initialization."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if m.bias is not None:
			nn.init.zeros_(m.bias)
	elif isinstance(m, nn.Conv2d):
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data, gain)
		if m.bias is not None:
			nn.init.zeros_(m.bias)


def ema(m, m_target, tau):
	"""Update slow-moving average of online network (target network) at rate tau."""
	with torch.no_grad():
		for p, p_target in zip(m.parameters(), m_target.parameters()):
			p_target.data.lerp_(p.data, tau)


def set_requires_grad(net, value):
	"""Enable/disable gradients for a given (sub)network."""
	for param in net.parameters():
		param.requires_grad_(value)


class TruncatedNormal(pyd.Normal):
	"""Utility class implementing the truncated normal distribution."""
	def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
		super().__init__(loc, scale, validate_args=False)
		self.low = low
		self.high = high
		self.eps = eps

	def _clamp(self, x):
		clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
		x = x - x.detach() + clamped_x.detach()
		return x

	def sample(self, clip=None, sample_shape=torch.Size()):
		shape = self._extended_shape(sample_shape)
		eps = _standard_normal(shape,
							   dtype=self.loc.dtype,
							   device=self.loc.device)
		eps *= self.scale
		if clip is not None:
			eps = torch.clamp(eps, -clip, clip)
		x = self.loc + eps
		return self._clamp(x)


class NormalizeImg(nn.Module):
	"""Normalizes pixel observations to [0,1) range."""
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.)


class Flatten(nn.Module):
	"""Flattens its input to a (batched) vector."""
	def __init__(self):
		super().__init__()
		
	def forward(self, x):
		return x.view(x.size(0), -1)


def enc(args):
	"""Returns a TOLD encoder."""
	if args['modality'] == 'pixels':
		C = int(3*args['frame_stack'])
		layers = [NormalizeImg(),
				  nn.Conv2d(C, args['num_channels'], 7, stride=2), nn.ReLU(),
				  nn.Conv2d(args['num_channels'], args['num_channels'], 5, stride=2), nn.ReLU(),
				  nn.Conv2d(args['num_channels'], args['num_channels'], 3, stride=2), nn.ReLU(),
				  nn.Conv2d(args['num_channels'], args['num_channels'], 3, stride=2), nn.ReLU()]
		out_shape = _get_out_shape((C, args['img_size'], args['img_size']), layers)
		layers.extend([Flatten(), nn.Linear(np.prod(out_shape), args['latent_dim'])])
	else:
		layers = [nn.Linear(args['obs_shape'][0], args['enc_dim']), nn.ELU(),
				  nn.Linear(args['enc_dim'], args['latent_dim'])]
	return nn.Sequential(*layers)


def mlp(in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
	"""Returns an MLP."""
	if isinstance(mlp_dim, int):
		mlp_dim = [mlp_dim, mlp_dim]
	return nn.Sequential(
		nn.Linear(in_dim, mlp_dim[0]), act_fn,
		nn.Linear(mlp_dim[0], mlp_dim[1]), act_fn,
		nn.Linear(mlp_dim[1], out_dim))

def q(args, act_fn=nn.ELU()):
	"""Returns a Q-function that uses Layer Normalization."""
	return nn.Sequential(nn.Linear(args['latent_dim']+args['action_dim'], args['mlp_dim']), nn.LayerNorm(args['mlp_dim']), nn.Tanh(),
						 nn.Linear(args['mlp_dim'], args['mlp_dim']), nn.ELU(),
						 nn.Linear(args['mlp_dim'], 1))


class RandomShiftsAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, args):
		super().__init__()
		self.pad = int(args['img_size']/21) if args['modality'] == 'pixels' else None

	def forward(self, x):
		if not self.pad:
			return x
		n, c, h, w = x.size()
		assert h == w
		padding = tuple([self.pad] * 4)
		x = F.pad(x, padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class Episode(object):
	"""Storage object for a single episode."""
	def __init__(self, args, init_obs):
		self.args = args
		self.device = torch.device(args['device'])
		dtype = torch.float32 if args['modality'] == 'state' else torch.uint8
		self.obs = torch.empty((args['episode_length']+1, *init_obs.shape), dtype=dtype, device=self.device)
		self.obs[0] = torch.tensor(init_obs, dtype=dtype, device=self.device)
		self.action = torch.empty((args['episode_length'], args['action_dim']), dtype=torch.float32, device=self.device)
		self.reward = torch.empty((args['episode_length'],), dtype=torch.float32, device=self.device)
		self.cumulative_reward = 0
		self.done = False
		self._idx = 0
	
	def __len__(self):
		return self._idx

	@property
	def first(self):
		return len(self) == 0
	
	def __add__(self, transition):
		self.add(*transition)
		return self

	def add(self, obs, action, reward, done):
		self.obs[self._idx+1] = torch.tensor(obs, dtype=self.obs.dtype, device=self.obs.device)
		self.action[self._idx] = action
		self.reward[self._idx] = reward
		self.cumulative_reward += reward
		self.done = done
		self._idx += 1


class ReplayBuffer():
	"""
	Storage and sampling functionality for training TD-MPC / TOLD.
	The replay buffer is stored in GPU memory when training from state.
	Uses prioritized experience replay by default."""
	def __init__(self, args):
		self.args = args
		self.device = torch.device(args['device'])
		self.capacity = min(args['train_steps'], args['max_buffer_size'])
		dtype = torch.float32 if args['modality'] == 'state' else torch.uint8
		obs_shape = args['obs_shape'] if args['modality'] == 'state' else (3, *args['obs_shape'][-2:])
		self._obs = torch.empty((self.capacity+1, *obs_shape), dtype=dtype, device=self.device)
		self._last_obs = torch.empty((self.capacity//args['episode_length'], *args['obs_shape']), dtype=dtype, device=self.device)
		self._action = torch.empty((self.capacity, args['action_dim']), dtype=torch.float32, device=self.device)
		self._reward = torch.empty((self.capacity,), dtype=torch.float32, device=self.device)
		self._priorities = torch.ones((self.capacity,), dtype=torch.float32, device=self.device)
		self._eps = 1e-6
		self._full = False
		self.idx = 0

	def __add__(self, episode: Episode):
		self.add(episode)
		return self

	def add(self, episode: Episode):
		self._obs[self.idx:self.idx+self.args['episode_length']] = episode.obs[:-1] if self.args['modality'] == 'state' else episode.obs[:-1, -3:]
		self._last_obs[self.idx//self.args['episode_length']] = episode.obs[-1]
		self._action[self.idx:self.idx+self.args['episode_length']] = episode.action
		self._reward[self.idx:self.idx+self.args['episode_length']] = episode.reward
		if self._full:
			max_priority = self._priorities.max().to(self.device).item()
		else:
			max_priority = 1. if self.idx == 0 else self._priorities[:self.idx].max().to(self.device).item()
		mask = torch.arange(self.args['episode_length']) >= self.args['episode_length']-self.args['horizon']
		new_priorities = torch.full((self.args['episode_length'],), max_priority, device=self.device)
		new_priorities[mask] = 0
		self._priorities[self.idx:self.idx+self.args['episode_length']] = new_priorities
		self.idx = (self.idx + self.args['episode_length']) % self.capacity
		self._full = self._full or self.idx == 0

	def update_priorities(self, idxs, priorities):
		self._priorities[idxs] = priorities.squeeze(1).to(self.device) + self._eps

	def _get_obs(self, arr, idxs):
		if self.args['modality'] == 'state':
			return arr[idxs]
		obs = torch.empty((self.args['batch_size'], 3*self.args['frame_stack'], *arr.shape[-2:]), dtype=arr.dtype, device=torch.device('cuda'))
		obs[:, -3:] = arr[idxs].cuda()
		_idxs = idxs.clone()
		mask = torch.ones_like(_idxs, dtype=torch.bool)
		for i in range(1, self.args['frame_stack']):
			mask[_idxs % self.args['episode_length'] == 0] = False
			_idxs[mask] -= 1
			obs[:, -(i+1)*3:-i*3] = arr[_idxs].cuda()
		return obs.float()

	def sample(self):
		probs = (self._priorities if self._full else self._priorities[:self.idx]) ** self.args['per_alpha']
		probs /= probs.sum()
		total = len(probs)
		idxs = torch.from_numpy(np.random.choice(total, self.args['batch_size'], p=probs.cpu().numpy(), replace=not self._full)).to(self.device)
		weights = (total * probs[idxs]) ** (-self.args['per_beta'])
		weights /= weights.max()

		obs = self._get_obs(self._obs, idxs)
		next_obs_shape = self._last_obs.shape[1:] if self.args['modality'] == 'state' else (3*self.args['frame_stack'], *self._last_obs.shape[-2:])
		next_obs = torch.empty((self.args['horizon']+1, self.args['batch_size'], *next_obs_shape), dtype=obs.dtype, device=obs.device)
		action = torch.empty((self.args['horizon']+1, self.args['batch_size'], *self._action.shape[1:]), dtype=torch.float32, device=self.device)
		reward = torch.empty((self.args['horizon']+1, self.args['batch_size']), dtype=torch.float32, device=self.device)
		for t in range(self.args['horizon']+1):
			_idxs = idxs + t
			next_obs[t] = self._get_obs(self._obs, _idxs+1)
			action[t] = self._action[_idxs]
			reward[t] = self._reward[_idxs]

		mask = (_idxs+1) % self.args['episode_length'] == 0
		next_obs[-1, mask] = self._last_obs[_idxs[mask]//self.args['episode_length']].cuda().float()
		if not action.is_cuda:
			action, reward, idxs, weights = \
				action.cuda(), reward.cuda(), idxs.cuda(), weights.cuda()

		return obs, next_obs, action, reward.unsqueeze(2), idxs, weights


def linear_schedule(schdl, step):
	"""
	Outputs values following a linear decay schedule.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	try:
		return float(schdl)
	except ValueError:
		match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
		if match:
			init, final, duration = [float(g) for g in match.groups()]
			mix = np.clip(step / duration, 0.0, 1.0)
			return (1.0 - mix) * init + mix * final
	raise NotImplementedError(schdl)