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
    base_path = "/root/home/gridworld/graphs/" + folder_subpath + args['algo'] + "/"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print("Making directory: ", base_path)

    print(base_path)
    # policy_kwargs = dict(net_arch=dict(pi=[32, 32], vf=[32, 32]))

    seeds = [1, 2, 3, 4, 5]
    print("Seeds used:", seeds)
    for trial_no in range(args['trials']):
        seed          = seeds[trial_no]
        utils.set_seed(seed)
        algos         = {'ppo' : PPO, 'dqn': DQN2, 'val_it': ValueIteration}
        models_tested = {}

        # Personal preference: Only print verbose if it is the first trial
        if trial_no == 0: verbose = 1
        else: verbose = 0

        if args['algo']   == 'val_it':
            models_tested[args['algo']] = ValueIteration(args, env_size, rewardDictionary)
        elif args['algo']   == 'ppo': 
            monitored_env = Monitor(env)
            models_tested[args['algo']] = (
                PPO("MlpPolicy", monitored_env, verbose=verbose, ent_coef = .9, device = 'cuda', seed = seed),
                monitored_env)
        elif args['algo'] == 'dqn': 
            monitored_env = Monitor(env)
            policy_kwargs = {"net_arch": network_arch}
            models_tested[args['algo']] = (
                DQN2("MlpPolicy", monitored_env, verbose=verbose, buffer_size = 100000, target_update_interval = 100, exploration_final_eps = 0.2, device = 'cuda', seed = seed, policy_kwargs=policy_kwargs),  
                monitored_env)
        elif args['algo'] == 'all':
            for algo, model in algos.items():
                monitored_env = Monitor(env)
                if algo == 'val_it': models_tested[algo] = model(args, env_size, rewardDictionary)
                else: models_tested[algo]                = (model("MlpPolicy", monitored_env, verbose=verbose, device = 'cuda'), monitored_env)

        for algo, model_env_tuple in models_tested.items():
            if algo == 'val_it':
                model = model_env_tuple
            else:
                model, monitored_env = model_env_tuple
                print("Testing algo:", algo)
                print("Model size:", model.policy)
                ##########################################################################################################
                ##########################################################################################################
                # Saving best model
                if args['env'] == 'gridworld':
                    env_eval        = gym.make("gymnasium_env/GridWorld-v0", size=env_size, max_steps=args['episode_length'] - 4).env
                    env_eval.load_rewards(rewardDictionary, eval_env = True)
                elif args['env'] == 'minigrid':
                    env_eval = gym.make("gymnasium_env/minigrid_toy").env

                monitored_eval_env = Monitor(env_eval)
                eval_callback = EvalCallback(monitored_eval_env, best_model_save_path='/root/home/gridworld/models/' + folder_subpath + algo + '/', eval_freq=500,
                                deterministic=False, render=False, verbose=1)
                ##########################################################################################################
                ##########################################################################################################

                ##########################################################################################################
                ##########################################################################################################
                # Model training
                model.learn(total_timesteps=args['train_steps'], callback=eval_callback, progress_bar = True)
                # model.learn(total_timesteps=50000, progress_bar = True)
                # model.save("gw_test")
                ##########################################################################################################
                ##########################################################################################################
                print(len(monitored_env.get_episode_rewards()))
                print(len(monitored_env.get_episode_lengths()))
                print(model.all_losses)
                print(model.buffer_logs)
                # print(model.replay_buffer.observations)
                model = model.load("/root/home/gridworld/models/" + folder_subpath + algo + "/" "best_model")

                # Path plotting. Recommended not to do this unless unit testing as it takes a long time to plot the graphs: 
                only_text = True
                if args['plot_graphs']: only_text = False

                monitored_env.plot_game_loc_diversity("./graphs/" + folder_subpath + algo + "_run" + str(trial_no), algo, only_text)

                # Losses from training
                utils.plot_array_and_save(
                    utils.exponential_moving_average(
                        model.all_losses), 
                        "./graphs/" + folder_subpath + algo + "_training_loss" + "_run" + str(trial_no), title = algo + " Loss", 
                        x_label = "Training step", y_label = "loss", y_max = 10, only_text = only_text)
                
                # Unique areas sampled from buffer at each training step
                utils.plot_array_and_save(
                    utils.exponential_moving_average(
                        model.buffer_logs), 
                        "./graphs/" + folder_subpath + algo + "_training_buffer_diversity" + "_run" + str(trial_no), title = algo + " TBD", 
                        x_label = "Training step", y_label = "unique areas", y_max = 6, only_text = only_text)

                # Rewards from each episode 
                utils.plot_array_and_save(
                    utils.exponential_moving_average(
                        monitored_env.get_episode_rewards()), 
                        "./graphs/" + folder_subpath + algo + "_episode_rewards" + "_run" + str(trial_no), title = algo + " Episode Rewards", 
                        x_label = "episodes", y_label = "rewards", y_max = 6, only_text = only_text)
                print("Saved episode rewards")

                # Length of each episode
                utils.plot_array_and_save(
                    utils.exponential_moving_average(
                        monitored_env.get_episode_lengths()), 
                        "./graphs/" + folder_subpath + algo + "_episode_lengths" + "_run" + str(trial_no), title = algo + " Episode Steps", 
                        x_label = "episodes", y_label = "total steps", y_max = max(monitored_env.get_episode_lengths()) + 5, only_text = only_text)
                print("Saved episode lengths")

        print(f'Trial completed for seed: {seed}')



        obs, info = env.reset()
        env.print_target_agent()
        done = False
        steps = 0
        path = []
        for i in range(100):
            action, _state = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action)
            print("Action:", action)
            print("location:", env.get_agent_loc())
            path.append(env.get_agent_loc(True))
            # env.render()
            if done:
                steps = i
                break
        print(f"Completed in {steps} steps with score of {reward}")
        # with open("graphs/" + algo + "_path.txt", "w") as text_file: text_file.write(env.print_path(path, return_string = True))
        env.print_path(path)
        env.print_path_image(path,'./graphs/' + folder_subpath + algo + "path.png", algo + " path")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
