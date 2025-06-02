import gymnasium as gym
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
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from algos.tdmpc import TDMPC
from algos.tdmpc_helper import Episode, ReplayBuffer
# from modified_algos.dqn2 import DQN2
# from modified_algos.safe_dqn.safe_dqn import SAFE_DQN
import wandb
from wandb.integration.sb3 import WandbCallback
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
import matplotlib as plt
from parse_args import parse_args

def format_save_file(params):
    file_string    = ""
    ignored_params = ['grid_size', 'theta', 'g_r', 'p_r', 'da_r']
    for key, value in params.items():
        if key not in ignored_params:
            file_string += str(key) + "lll" + str(value) + "_"

    os.makedirs(file_string[:-1], mode=0o777, exist_ok=True)

    return file_string[:-1]

if __name__ == '__main__':
    args             = parse_args()

    network_arch = []
    folder_subpath = "./logs/dead_ears/"

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
    input_pol  = "MlpPolicy"
    env_size   = 10
    env        = gym.make("gymnasium_env/GridWorld-v0", args = args, size=env_size).env
    env.reset()
    env_eval        = gym.make("gymnasium_env/GridWorld-v0", args = args, size=env_size).env
    env_eval.reset()

    monitored_env      = Monitor(env)
    monitored_eval_env = Monitor(env_eval)

    start_time    = time.time()

    seeds = [1, 2, 3, 4, 5]
    print("Seeds used:", seeds)
    for trial_no in range(args['trials']):
        seed          = seeds[trial_no]
        utils.set_seed(seed)
        algos         = {'ppo' : PPO}
        models_tested = {}
        config = {"policy_type": "MlpPolicy", "architecture": network_arch_str}
        for key, value in args.items():
            config[key] = value
        config['seed'] = seed
        run = wandb.init(
            # Set the project where this run will be logged
            project="dead-agent",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=folder_subpath + args['script_id'],
            # Track hyperparameters and run metadata
            config=config,
            sync_tensorboard=True)

        # Personal preference: Only print verbose if it is the first trial
        if trial_no == 0: verbose = 1
        else: verbose = 0

        models_tested['ppo'] = (
            PPO(input_pol, monitored_env, verbose=verbose,
                learning_rate = args['lr'],         # Default 0.0003
                n_steps       = args['n_steps'],    # Default 2048
                batch_size    = args['batch_size'], # Default 64
                gae_lambda    = args['gae_lambda'], # Default 0.95
                gamma         = args['gamma'],      # Default .99
                n_epochs      = args['n_epochs'],   # Default 10
                ent_coef      = args['ent_coef'],   # Default .0
                clip_range    = args['clip_range'], # Default 0.2
                vf_coef       = args['vf_coef'],    # Default 0
                max_grad_norm = args['mgm'],        # Default 0.5
                device = 'cuda', seed = seed, policy_kwargs=policy_kwargs, tensorboard_log=f"runs/{run.id}"), monitored_env)

        
        for algo, model_env_tuple in models_tested.items():
            model, monitored_env = model_env_tuple
            print("Testing algo:", algo)
            print("Model size:", model.policy)
            ##########################################################################################################
            ##########################################################################################################
            eval_callback = EvalCallback(monitored_eval_env, best_model_save_path=args['root_checkpoint_save_dir'] + args['env'] + '/models/' + folder_subpath + algo + '/', eval_freq=int(args['train_steps']/1000),
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

            model = model.load(args['root_checkpoint_save_dir'] + args['env'] + "/models/" + folder_subpath + algo + "/" "best_model")

        print(f'Trial completed for seed: {seed}')
        run.finish()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
