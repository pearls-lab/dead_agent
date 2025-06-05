import os
import torch
import gymnasium as gym
from stable_baselines3 import PPO           # change if you trained with another algo
from gridworld_env import GridWorldEnv      # your custom env
from parse_args import parse_args           # same helper you used for training

# ------------------------------------------------------------------
# 1) Re-create the env in the same way you did for training
# ------------------------------------------------------------------
args = parse_args()                         # loads CLI / default params
env_size = 10                               # keep consistent with training
eval_env = gym.make(
    "gymnasium_env/GridWorld-v0",
    args=args,
    size=env_size
).env

# ------------------------------------------------------------------
# 2) Build the path to the best model & load it
# ------------------------------------------------------------------
# The directory layout in training was:
#   {root_checkpoint_save_dir}/{env}/models/{folder_subpath}{algo}/best_model.zip
# We need to rebuild folder_subpath the same way:
folder_subpath = f"./logs/dead_ears/{args['composer']}/"
for _ in range(args['layers'] + 1):
    folder_subpath += f"{args['parameters']}_"
folder_subpath = folder_subpath[:-1] + "/"     # remove trailing "_" and add "/"

algo_name = "ppo"                              # or dqn / a2c / etc.
best_model_path = os.path.join(
    args['root_checkpoint_save_dir'],
    args['env'],
    "models",
    f"{folder_subpath}{algo_name}",
    "best_model",                              # stable-baselines adds ".zip" automatically
)

model = PPO.load(
    best_model_path,
    env=eval_env,                              # let SB3 wrap the env for you
    device="cuda" if torch.cuda.is_available() else "cpu",
)

print("Loaded model from:", best_model_path)

# ------------------------------------------------------------------
# 3) Run ONE deterministic evaluation episode
# ------------------------------------------------------------------
obs, _ = eval_env.reset(seed=args['seed'])     # for reproducibility
terminated = truncated = False
episode_return = 0.0

while not (terminated or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    episode_return += reward

print(f"Episode return: {episode_return}")
