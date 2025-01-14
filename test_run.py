import minigrid
import gymnasium as gym
from gridworld_env import GridWorldEnv
from minigrid_utils.feature_extractor import MinigridFeaturesExtractor
from minigrid_custom import SimpleEnv
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback
from home.gridworld.modified_algos.dqn2 import DQN2
import matplotlib as plt
from utils import display_environment

policy_kwargs = dict(
    net_arch = [512, 512, 512],
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128)
)
save_path = "./minigrid.png"

env = gym.make("gymnasium_env/minigrid_toy", size = 6)
env.reset()
img = env.get_frame()

# Save the rendered image
plt.pyplot.imshow(img)
plt.pyplot.savefig(save_path, bbox_inches='tight', pad_inches=0)
print(f"Image saved to {save_path}")
env = ImgObsWrapper(env)


model = DQN2("CnnPolicy", env, buffer_size = 1000000, gradient_steps = 1, learning_rate= 0.0001, gamma = 0.9, 
                     target_update_interval = 100, exploration_final_eps = 0.2, device = 'cuda', policy_kwargs=policy_kwargs, verbose=1)

# model = PPO("CnnPolicy", env, device = 'cuda', policy_kwargs=policy_kwargs, verbose=1)
model.learn(1e6, progress_bar = True)