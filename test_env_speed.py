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
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
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
    parser.add_argument('--plot_graphs', default=False,type=bool,
                        help="Default: False")
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
    parser.add_argument('--trials', default=5, type=int,
                        help="number of runs to do")
    parser.add_argument('--gradient_steps', default=1, type=int,
                        help="number of gradient steps")
    parser.add_argument('--death_timer', default=1, type=int,
                        help="number of steps agent can take before dying after stepping in a dead area.")
    

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

    network_arch = []
    folder_subpath = args['reward_dict'].replace(".json", "/")
    for layer in range(args['layers'] + 1):
        network_arch.append(args['parameters'])
        folder_subpath += str(args['parameters']) + "_"       
    folder_subpath = folder_subpath[:-1] + "/"

    print("Folder subpath: ", folder_subpath)

    network_arch_str = ""
    for layer in network_arch:
        network_arch_str += str(layer) + "_"
    network_arch_str = network_arch_str[:-1]

    # Gridworld
    if args['env'] == 'gridworld':
        env_size   = 10
        env        = gym.make("gymnasium_env/GridWorld-v0", args = args, size=env_size).env
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
    base_path = "/root/home/gridworld/graphs/" + folder_subpath + args['algo'] + "/"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print("Making directory: ", base_path)

    print(base_path)
    # policy_kwargs = dict(net_arch=dict(pi=[32, 32], vf=[32, 32]))

    seeds = [1, 2, 3, 4, 5]
    print("Seeds used:", seeds)

    obs, info = env.reset()
    env.print_target_agent()
    done = False
    steps = 0
    path = []
    for i in range(1000000):
        action = np.random.randint(0, 4)
        obs, reward, done, truncated, info = env.step(action)
        if done:
            obs, info = env.reset()
    print(f"Completed in {steps} steps with score of {reward}")
    # with open("graphs/" + algo + "_path.txt", "w") as text_file: text_file.write(env.print_path(path, return_string = True))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
