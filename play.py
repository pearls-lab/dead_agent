import gymnasium as gym
import argparse
import os
import json
import torch
import numpy as np
import time
import utils
from gridworld_env import GridWorldEnv
from minigrid_custom import SimpleEnv
from algos.val_it import ValueIteration
from sb3.stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from algos.tdmpc import TDMPC
from algos.tdmpc_helper import Episode, ReplayBuffer
from dqn2 import DQN2

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
    parser.add_argument('--env', default='minigrid',type=str,
                        help="minigrid or gridworld")
    parser.add_argument('--tp', default=False,type=bool,
                        help="Terminal Poison: Whether or not agent dies after stepping in poison. Default: False")
    parser.add_argument('--dt', default=-1,type=int, 
                        help="Death Timer: Number of steps after stepping in poison that the agent dies. -1 means random. Default:-1")
    parser.add_argument('--episode_length', default=100, type=int,
                        help="Episode length")
    parser.add_argument('--layers', default=2, type=int,
                        help="Number of layers for network")
    parser.add_argument('--parameters', default=64, type=int,
                        help="Number of parameters per layer")
    parser.add_argument('--reward_dict', default="rewards.json", type=str,
                        help="file path to reward dict")
    parser.add_argument('--train_steps', default=1000000, type=int,
                        help="steps to train model")
    

    parser.set_defaults(gat=True)
    args   = parser.parse_args()
    params = vars(args)
    return params

if __name__ == '__main__':
    args             = parse_args()

    # Load reward dictionary for environment
    rewardDictionary = {}
    with open(args['reward_dict'], 'r') as file: rewardDictionary = json.load(file)
    assert len(rewardDictionary) > 1, "Environment must contain at least 1 reward" 

    # Gridworld
    if args['env'] == 'gridworld':
        env_size   = 10
        env        = gym.make("gymnasium_env/GridWorld-v0", size=env_size, max_steps=args['episode_length'] - 4).env
        env.load_rewards(rewardDictionary)
    # Minigrid
    elif args['env'] == 'minigrid':
        env = gym.make("gymnasium_env/minigrid_toy")
        env.reset()
        print(env.observation_space)
        print("Env", env)

    valid_envs = ['minigrid', 'gridworld']
    assert args['env'] in valid_envs


    start_time    = time.time()

   
    obs, info = env.reset()
    env.print_target_agent()
    done = False
    steps = 0
    path = []
    for i in range(100):
        action = int(input())
        obs, reward, done, truncated, info = env.step(action)
        print("Action:", action)
        print("location:", env.get_agent_loc())
        print("Done?", done)
        path.append(env.get_agent_loc(True))
        # env.render()
        if done:
            steps = i
            break
    print(f"Completed in {steps} steps with score of {reward}")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
