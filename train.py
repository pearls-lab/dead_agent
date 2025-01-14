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
from minigrid_utils.feature_extractor import MinigridFeaturesExtractor
from algos.val_it import ValueIteration
from sb3.stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from algos.tdmpc import TDMPC
from algos.tdmpc_helper import Episode, ReplayBuffer
from modified_algos.dqn2 import DQN2
import wandb
from wandb.integration.sb3 import WandbCallback
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
import matplotlib as plt

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
    parser.add_argument('--trials', default=4, type=int,
                        help="number of runs to do")
    parser.add_argument('--gradient_steps', default=1, type=int,
                        help="number of gradient steps")
    parser.add_argument('--lr', default=0.001, type=float,
                        help="default learning rate")
    parser.add_argument('--buffer_size', default=100000, type=int,
                        help="buffer size")
    parser.add_argument('--multi_buffer', default=False, type=bool,
                        help="Experimental: Add a second buffer for specfic trajectories")
    parser.add_argument('--teacher_force', default=False, type=bool,
                        help="gridworld only rn: will cause the agent to use the most optimal trajectory found")
    parser.add_argument('--script_id', default="default_name", type=str,
                        help="Script identifier for wandb")
    parser.add_argument('--death_timer', default=1, type=int,
                        help="number of steps agent can take before dying after stepping in a dead area.")
    

    parser.set_defaults(gat=True)
    args   = parser.parse_args()
    params = vars(args)
    return params

if __name__ == '__main__':
    args             = parse_args()

    valid_envs       = ['minigrid', 'gridworld']
    assert args['env'] in valid_envs
    # Load reward dictionary for environment
    try:
        rewardDictionary = {}
        with open(args['reward_dict'], 'r') as file: rewardDictionary = json.load(file)
        assert len(rewardDictionary) > 1, "Environment must contain at least 1 reward" 
    except:
        print("Reward dictionary not loaded. Will be a basic grid-box environment")

    network_arch = []
    folder_subpath = "./logs/"
    if args['env'] == 'gridworld':
        folder_subpath += 'gridworld/' + args['reward_dict'].replace(".json", "/")
    if args['env'] == 'minigrid':
        folder_subpath += "minigrid/"
    for layer in range(args['layers'] + 1):
        network_arch.append(args['parameters'])
        folder_subpath += str(args['parameters']) + "_"       
    folder_subpath = folder_subpath[:-1] + "/"

    print("Folder subpath: ", folder_subpath)

    base_path = folder_subpath
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print("Making directory: ", base_path)

    network_arch_str = ""
    for layer in network_arch:
        network_arch_str += str(layer) + "_"
    network_arch_str = network_arch_str[:-1]

    policy_kwargs = {"net_arch": network_arch}
    # Gridworld
    if args['env'] == 'gridworld':
        input_pol  = "MlpPolicy"
        env_size   = 10
        env        = gym.make("gymnasium_env/GridWorld-v0", args = args, size=env_size).env
        env.load_rewards(rewardDictionary)
        env_eval        = gym.make("gymnasium_env/GridWorld-v0", args = args, size=env_size).env
        env_eval.load_rewards(rewardDictionary, eval_env = True)
    # Minigrid
    elif args['env'] == 'minigrid':
        policy_kwargs["features_extractor_class"]  = MinigridFeaturesExtractor
        policy_kwargs["features_extractor_kwargs"] = dict(features_dim=128)
        
        input_pol       = "CnnPolicy"
        env_size        = 10
        env             = gym.make("gymnasium_env/minigrid_toy", size = env_size)
        env_eval        = gym.make("gymnasium_env/minigrid_toy", size = env_size)
        print(env.observation_space)
        # Credit: https://github.com/DLR-RM/stable-baselines3/issues/689
        env             = ImgObsWrapper(env)  # Get rid of the 'mission' field
        reset_vars      = env.reset()

        img = env.get_frame()
        plt.pyplot.imshow(img)
        plt.pyplot.savefig(folder_subpath + "visualization", bbox_inches='tight', pad_inches=0)
        print(f"Minigrid Layout image saved to {folder_subpath}")

        env_eval        = ImgObsWrapper(env_eval)
        reset_vars      = env_eval.reset()
        print(env.observation_space)
        print("Env", env)

    monitored_env      = Monitor(env)
    monitored_eval_env = Monitor(env_eval)

    start_time    = time.time()
    # policy_kwargs = dict(net_arch=dict(pi=[32, 32], vf=[32, 32]))

    seeds = [1, 2, 3, 4, 5]
    print("Seeds used:", seeds)
    for trial_no in range(args['trials']):
        seed          = seeds[trial_no]
        utils.set_seed(seed)
        algos         = {'ppo' : PPO, 'dqn': DQN2, 'val_it': ValueIteration}
        models_tested = {}
        run = wandb.init(
            # Set the project where this run will be logged
            project="dead-agent",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=folder_subpath + args['script_id'],
            # Track hyperparameters and run metadata
            config={
                "policy_type": "MlpPolicy",
                "architecture": network_arch_str,
                "env": args['env'],
                "epochs": args['train_steps'],
                "grad_steps": args['gradient_steps'],
                "death_timer": args['death_timer'],
                "algo": args['algo'],
                "reward_dict": args['reward_dict'],
                "script_id": args['script_id'],
                "lr": args['lr'],
                "buffer_size": args['buffer_size'],
                'gamma': args['gamma'],
                "multi_buffer": args['multi_buffer']
            },
            sync_tensorboard=True)

        # Personal preference: Only print verbose if it is the first trial
        if trial_no == 0: verbose = 1
        else: verbose = 0

        if args['algo']   == 'val_it':
            models_tested[args['algo']] = ValueIteration(args, env_size, rewardDictionary)
        elif args['algo']   == 'ppo': 
            monitored_env = Monitor(env)
            # models_tested[args['algo']] = (
            #     PPO("MlpPolicy", monitored_env, verbose=verbose, ent_coef = .9, device = 'cuda', seed = seed),
            #     monitored_env)
            models_tested[args['algo']] = (
                PPO("MlpPolicy", monitored_env, verbose=verbose, n_steps = 128, batch_size= 64, gae_lambda= 0.95, gamma = .99, n_epochs = 10, ent_coef = .0, learning_rate = 2.5e-4, clip_range = 0.2 , device = 'cuda', seed = seed),
                monitored_env)
        elif args['algo'] == 'dqn': 
            models_tested[args['algo']] = (
                DQN2(input_pol, monitored_env, verbose=verbose, 
                     buffer_size            = args['buffer_size'], 
                     gradient_steps         = args['gradient_steps'], 
                     learning_rate          = args['lr'],
                     gamma                  = args['gamma'],
                     multi_buffer           = args['multi_buffer'],
                     target_update_interval = 100, 
                     exploration_final_eps  = 0.2, 
                     device = 'cuda', seed = seed, policy_kwargs=policy_kwargs, tensorboard_log=f"runs/{run.id}"),  
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
                eval_callback = EvalCallback(monitored_eval_env, best_model_save_path='/root/home/' + args['env'] + '/models/' + folder_subpath + algo + '/', eval_freq=int(args['train_steps']/1000),
                                deterministic=False, render=False, verbose=0)
                ##########################################################################################################
                ##########################################################################################################

                ##########################################################################################################
                ##########################################################################################################
                # Model training
                wandb_callback = WandbCallback(
                                gradient_save_freq=100000,
                                model_save_path=f"models/{run.id}",
                                verbose=0,
                            )
                callback = CallbackList([wandb_callback, eval_callback])
                model.learn(total_timesteps=args['train_steps'], callback=callback, progress_bar = False)
                # model.learn(total_timesteps=50000, progress_bar = True)
                # model.save("gw_test")
                ##########################################################################################################
                ##########################################################################################################
                print(len(monitored_env.get_episode_rewards()))
                print(len(monitored_env.get_episode_lengths()))

                model = model.load("/root/home/" + args['env'] + "/models/" + folder_subpath + algo + "/" "best_model")

        print(f'Trial completed for seed: {seed}')
        run.finish()
        # monitored_env.save_trajectories()


        # obs, info = env.reset()
        # env.print_target_agent()
        # done = False
        # steps = 0
        # path = []
        # for i in range(100):
        #     action, _state = model.predict(obs, deterministic=False)
        #     obs, reward, done, truncated, info = env.step(action)
        #     print("Action:", action)
        #     print("location:", env.get_agent_loc())
        #     path.append(env.get_agent_loc(True))
        #     # env.render()
        #     if done:
        #         steps = i
        #         break
        # print(f"Completed in {steps} steps with score of {reward}")
        # with open("graphs/" + algo + "_path.txt", "w") as text_file: text_file.write(env.print_path(path, return_string = True))
        # env.print_path(path)
        # env.print_path_image(path,'./graphs/' + folder_subpath + algo + "path.png", algo + " path")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
