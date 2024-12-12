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
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
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
    parser.add_argument('--episode_length', default=100, type=int,
                        help="Episode length")
    

    parser.set_defaults(gat=True)
    args   = parser.parse_args()
    params = vars(args)
    return params

if __name__ == '__main__':
    args             = parse_args()

    rewardDictionary = {
        "goal": 
        {
            "location": [[9,9]],
            "reward": 5,
            "available": True,
            "terminal": True,
            "identifier": 19
        },
        "start":
        {
            "location": [[0,0]],
            "reward": 0,
            "available": True,
            "terminal": False,
            "identifier": 1
        }
        ,"dead_areas": 
        {
            "location": [[3,0], [3,1], [3,2], [3,3], [3,4],
                         [7,9], [7,8], [7,7], [7,6], [7,5]],
            "reward": -10,
            "available": True,
            "terminal": False,
            "identifier": -99
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
    seed = 9
    if args['algo']   == 'val_it':
        models_tested[args['algo']] = ValueIteration(args, env_size, rewardDictionary)
    elif args['algo']   == 'ppo': 
        monitored_env = Monitor(env)
        models_tested[args['algo']] = (
            PPO("MlpPolicy", monitored_env, verbose=0, device = 'cuda', seed = seed), 
            monitored_env)
    elif args['algo'] == 'dqn': 
        monitored_env = Monitor(env)
        models_tested[args['algo']] = (
            DQN("MlpPolicy", monitored_env, verbose=1, target_update_interval = 100, exploration_final_eps = 0.2, device = 'cuda', seed = seed), 
            monitored_env)
    elif args['algo'] == 'all':
        for algo, model in algos.items():
            monitored_env = Monitor(env)
            if algo == 'val_it': models_tested[algo] = model(args, env_size, rewardDictionary)
            else: models_tested[algo]                = (model("MlpPolicy", monitored_env, verbose=0, device = 'cuda'), monitored_env)

    for algo, model_env_tuple in models_tested.items():
        if algo == 'val_it':
            model = model_env_tuple
        else:
            model, monitored_env = model_env_tuple
            print("Testing algo:", algo)
            ##########################################################################################################
            ##########################################################################################################
            # Saving best model
            env_eval        = gym.make("gymnasium_env/GridWorld-v0", size=env_size, max_steps=args['episode_length'] - 4).env
            env_eval.load_rewards(rewardDictionary, eval_env = True)
            monitored_eval_env = Monitor(env_eval)
            eval_callback = EvalCallback(monitored_eval_env, best_model_save_path='/root/home/gridworld/models/' + algo + '/', eval_freq=500,
                             deterministic=False, render=False, verbose=1)
            ##########################################################################################################
            ##########################################################################################################

            ##########################################################################################################
            ##########################################################################################################
            # Model training
            model.learn(total_timesteps=50000, callback=eval_callback, progress_bar = True)
            # model.learn(total_timesteps=50000, progress_bar = True)
            # model.save("gw_test")
            ##########################################################################################################
            ##########################################################################################################
            print(len(monitored_env.get_episode_rewards()))
            print(len(monitored_env.get_episode_lengths()))
            print(model.replay_buffer.observations)
            model = model.load("/root/home/gridworld/models/" + algo + "/best_model")
            utils.plot_array_and_save(
                utils.exponential_moving_average(
                    monitored_env.get_episode_rewards()), 
                    "./graphs/" + algo + "_episode_rewards", title = algo + " Episode Rewards", 
                    x_label = "episodes", y_label = "rewards", y_max = 6)
            utils.plot_array_and_save(
                utils.exponential_moving_average(
                    monitored_env.get_episode_lengths()), 
                    "./graphs/" + algo + "_episode_lengths", title = algo + " Episode Steps", 
                    x_label = "episodes", y_label = "total steps", y_max = max(monitored_env.get_episode_lengths()) + 5)



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
        env.print_path_image(path,'./graphs/' + algo + "path.png", algo + " path")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
