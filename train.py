import gymnasium as gym
import argparse
import os
import json
import torch
import numpy as np
import time
import utils
from gridworld_env import GridWorldEnv
from algos.val_it import ValueIteration
from sb3.stable_baselines3 import A2C, DQN, PPO
from sb3.stable_baselines3.common.monitor import Monitor
from algos.tdmpc import TDMPC
from algos.tdmpc_helper import Episode, ReplayBuffer

def format_save_file(params):
    file_string    = ""
    ignored_params = ['grid_size', 'theta', 'g_r', 'p_r', 'da_r']
    for key, value in params.items():
        if key not in ignored_params:
            file_string += str(key) + "lll" + str(value) + "_"

    os.makedirs(file_string[:-1], mode=0o777, exist_ok=True)

    return file_string[:-1]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ddsp', default= False, type=bool)
    parser.add_argument('--snr', default= False, type=bool)
    parser.add_argument('--sp', default=-1, type=int,
                        help="step penalty")
    parser.add_argument('--iter', default=5, type=int,
                        help="Iterations for val_it and tdmpc") # tdmpc: iterations
    parser.add_argument('--gamma', default='.9', type=float,
                        help="gamma/discount factor") # Called discount in tdmpc
    parser.add_argument('--theta', default='1e-4', type=float,
                        help="theta for val it")
    parser.add_argument('--seed', default=420, type=int)
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--max_env_steps', default = 100, type=int) # Called episode length in tdmpc
    parser.add_argument('--reward_file', default= 'rewards.json', type=str)
    parser.add_argument('--algo', default='val_it',type=str,
                        help="val_it, ppo, dqn or tdmpc")
    parser.add_argument('--tp', default=False,type=bool,
                        help="Terminal Poison: Whether or not agent dies after stepping in poison. Default: False")
    parser.add_argument('--dt', default=-1,type=int, 
                        help="Death Timer: Number of steps after stepping in poison that the agent dies. -1 means random. Default:-1")
    
    # TDMPC
    # Environment
    parser.add_argument('--task', default='quadruped-run', type=str,
                        help="Task/environment name")
    parser.add_argument('--modality', default='state', type=str,
                        help="Input modality (e.g., state or pixels)")
    parser.add_argument('--action_repeat', default=1, type=int,
                        help="Action repeat factor")  # Default inferred from missing value
    parser.add_argument('--discount', default=0.99, type=float,
                        help="Discount factor for reward")
    parser.add_argument('--episode_length', default=100, type=int,
                        help="Episode length")
    parser.add_argument('--train_steps', default=500000, type=int,
                        help="Training steps")

    # Planning
    parser.add_argument('--iterations', default=6, type=int,
                        help="Number of planning iterations")
    parser.add_argument('--num_samples', default=512, type=int,
                        help="Number of samples for planning")
    parser.add_argument('--num_elites', default=64, type=int,
                        help="Number of elite samples for planning")
    parser.add_argument('--mixture_coef', default=0.05, type=float,
                        help="Mixture coefficient for sampling")
    parser.add_argument('--min_std', default=0.05, type=float,
                        help="Minimum standard deviation")
    parser.add_argument('--temperature', default=0.5, type=float,
                        help="Temperature for exploration")
    parser.add_argument('--momentum', default=0.1, type=float,
                        help="Momentum for planning")

    # Learning
    parser.add_argument('--batch_size', default=8, type=int,
                        help="Batch size for training")
    parser.add_argument('--max_buffer_size', default=100, type=int,
                        help="Maximum replay buffer size")
    parser.add_argument('--horizon', default=5, type=int,
                        help="Horizon length for planning")
    parser.add_argument('--reward_coef', default=0.5, type=float,
                        help="Reward coefficient")
    parser.add_argument('--value_coef', default=0.1, type=float,
                        help="Value coefficient")
    parser.add_argument('--consistency_coef', default=2, type=float,
                        help="Consistency coefficient")
    parser.add_argument('--rho', default=0.5, type=float,
                        help="Rho parameter")
    parser.add_argument('--kappa', default=0.1, type=float,
                        help="Kappa parameter")
    parser.add_argument('--lr', default=1e-3, type=float,
                        help="Learning rate")
    parser.add_argument('--std_schedule', default='linear(0.5, 0.05, 25000)', type=str,
                        help="Schedule for standard deviation")
    parser.add_argument('--horizon_schedule', default='linear(1, 5, 25000)', type=str,
                        help="Schedule for planning horizon")
    parser.add_argument('--per_alpha', default=0.6, type=float,
                        help="PER alpha parameter")
    parser.add_argument('--per_beta', default=0.4, type=float,
                        help="PER beta parameter")
    parser.add_argument('--grad_clip_norm', default=10, type=float,
                        help="Gradient clipping norm")
    parser.add_argument('--seed_steps', default=10, type=int,
                        help="Initial steps for seeding the buffer")
    parser.add_argument('--update_freq', default=2, type=int,
                        help="Update frequency for training")
    parser.add_argument('--tau', default=0.01, type=float,
                        help="Soft update coefficient")

    # Architecture
    parser.add_argument('--enc_dim', default=256, type=int,
                        help="Encoder dimension")
    parser.add_argument('--mlp_dim', default=512, type=int,
                        help="MLP dimension")
    parser.add_argument('--latent_dim', default=50, type=int,
                        help="Latent dimension size")

    # WandB
    parser.add_argument('--use_wandb', default=False, type=bool,
                        help="Use Weights & Biases for logging")
    parser.add_argument('--wandb_project', default='none', type=str,
                        help="Weights & Biases project name")
    parser.add_argument('--wandb_entity', default='none', type=str,
                        help="Weights & Biases entity")

    # Miscellaneous
    parser.add_argument('--exp_name', default='default', type=str,
                        help="Experiment name")
    parser.add_argument('--eval_freq', default=20000, type=int,
                        help="Frequency of evaluation")
    parser.add_argument('--eval_episodes', default=10, type=int,
                        help="Number of evaluation episodes")
    parser.add_argument('--save_video', default=False, type=bool,
                        help="Save evaluation videos")
    parser.add_argument('--save_model', default=False, type=bool,
                        help="Save trained model checkpoints")

    parser.set_defaults(gat=True)
    args   = parser.parse_args()
    params = vars(args)
    return params

if __name__ == '__main__':
    args             = parse_args()

    rewardDictionary = {
        "goal": 
        {
            "location": [9,9],
            "reward": 5,
            "available": True,
            "terminal": True
        },
        "start":
        {
            "location": [0,0],
            "reward": 0,
            "available": True,
            "terminal": False
        }
    }

    # save_location    = format_save_file(params)
    # print(f'Saving files at {save_location}')
    # with open('./' + save_location + '/args.json', 'w') as fp: json.dump(params, fp)

    env_size   = 10
    env        = gym.make("gymnasium_env/GridWorld-v0", size=env_size, max_steps=args['episode_length'] - 4).env
    env.load_rewards(rewardDictionary)
    # Testing val it
    # model      = ValueIteration(args, env_size, rewardDictionary)
    # obs, info  = env.reset()
    # terminated = False
    # total_rew  = 0
    # while not terminated:
    #     act                                   = model.forward(obs)
    #     print("Act", act)
    #     obs, rew, terminated, truncated, info = env.step(act)
    #     total_rew                            += rew

    # print("Final reward:", total_rew)
    start_time    = time.time()
    algos         = {'ppo' : PPO, 'dqn': DQN, 'val_it': ValueIteration}
    models_tested = {}
    if args['algo']   == 'ppo': models_tested[args['algo']] = PPO("MlpPolicy", Monitor(env), verbose=1, device = 'cuda')
    elif args['algo'] == 'dqn': models_tested[args['algo']] = DQN("MlpPolicy", Monitor(env), verbose=1, device = 'cuda')
    elif args['algo'] == 'all':
        for algo, model in algos.items():
            monitored_env = Monitor(env)
            if algo == 'val_it': models_tested[algo] = model(args, env_size, rewardDictionary)
            else: models_tested[algo]                = (model("MlpPolicy", monitored_env, verbose=1, device = 'cuda'), monitored_env)

    for algo, model_env_tuple in models_tested.items():
        model, monitored_env = model_env_tuple
        print("Testing algo:", algo)
        model.learn(total_timesteps=10000, progress_bar = True)
        model.save("gw_test")
        utils.plot_array_and_save(
            utils.exponential_moving_average(
                monitored_env.get_episode_rewards()), 
                "./graphs/" + algo + "_episode_rewards", title = algo + "_episodeRews", 
                x_label = "episodes", y_label = "rewards", y_max = 6)
        utils.plot_array_and_save(
            utils.exponential_moving_average(
                monitored_env.get_episode_lengths()), 
                "./graphs/" + algo + "_episode_lengths", title = "title: placeholder", 
                x_label = "episodes", y_label = "total steps", y_max = max(monitored_env.get_episode_lengths()) + 5)

        obs, info = env.reset()
        done = False
        steps = 0
        for i in range(20):
            action, _state = model.predict(obs, deterministic=True)
            # print("Action", action)
            # action, _state = model.predict(FlattenObservation(obs), deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            # env.render()
            if done:
                steps = i
                break
        print(f"Completed in {steps} steps with score of {reward}")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    
    # Testing TRPO
    # model = PPO("MlpPolicy", env, verbose=1, device = 'cuda')
    # model.learn(total_timesteps=100000)
    # model.save("gw_test")
    # obs, info = env.reset()
    # env.render()

        # for i in range(20):
    #     action, _state = model.predict(obs, deterministic=True)
    #     print("Action", action)
    #     # action, _state = model.predict(FlattenObservation(obs), deterministic=True)
    #     obs, reward, done, truncated, info = env.step(action)
    #     env.render()
    #     if done:
    #         break

    # Testing TDMPC
    # obs, info = env.reset()
    # args['obs_shape'] = torch.tensor(obs).size()
    # args['action_dim'] = 1
    # args['device'] = 'cuda'
    # model  = TDMPC(args)
    # buffer = ReplayBuffer(args)
    # eps_idx = 0
    # print("Env max steps:", env.max_steps)
    # print(env.size)
    # obs, info = env.reset()

    # assert env.max_steps < args['max_buffer_size']

    # # Learning for TDMPC:
    # for i in range(args['train_steps']):
    #     print("Step: ", i)
    #     all_actions = []
    #     # Collecting trajectory
    #     obs, info = env.reset()
    #     episode = Episode(args, obs)
    #     ste = 0
    #     while not episode.done:
    #         action = model.plan(obs = obs, step=i, t0 = episode.first)
    #         all_actions.append(int(action[0].cpu().numpy()))
    #         obs, reward, done, _, _ = env.step(int(action[0].cpu().numpy()))
    #         ste += 1
    #         episode += (obs, action, reward, done)
    #     # assert len(episode) == args['episode_length']
    #     buffer += episode

    #     # Update model
    #     train_metrics = {}
    #     if i >= args['seed_steps']:
    #         num_updates = args['seed_steps'] if i == args['seed_steps'] else args['episode_length']
    #         for j in range(num_updates):
    #             train_metrics.update(model.update(buffer, i + j))

    #     # Log training episode
    #     eps_idx += 1
    #     env_step = int(i * args['action_repeat'])
    #     common_metrics = {
	# 		'episode': eps_idx,
	# 		'step': i,
	# 		'env_step': env_step,
	# 		'episode_reward': episode.cumulative_reward}
    #     train_metrics.update(common_metrics)
    #     print(all_actions)
    #     print(np.unique(all_actions))
    #     if i % 10 == 0:
    #         print(train_metrics)

    # print("last reward:", reward)
    # # print(env.observation_space)
    # # print(env.get_grid())
    # # print(env.size)
    # # print("obs", obs)
    # # print("Flattened grid:", env.get_flattened_grid())
    # # print(info)
