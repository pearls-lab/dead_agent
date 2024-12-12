import os
import pickle as pkl
import random
import sys
import time
import gymnasium as gym
from pprint import pprint

import optuna
from absl import flags
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor

from gridworld_env import GridWorldEnv
from optuna_utils.sample_params.ppo import sample_ppo_params
from optuna_utils.trial_eval_callback import TrialEvalCallback

FLAGS = flags.FLAGS
FLAGS(sys.argv)


study_path = "/root/home/gridworld/optuna/"


def objective(trial: optuna.Trial) -> float:

    time.sleep(random.random() * 16)

    sampled_hyperparams = sample_ppo_params(trial)

    path = f"{study_path}/trial_{str(trial.number)}"
    os.makedirs(path, exist_ok=True)

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
    }

    # save_location    = format_save_file(params)
    # print(f'Saving files at {save_location}')
    # with open('./' + save_location + '/args.json', 'w') as fp: json.dump(params, fp)

    env_size   = 10
    env        = gym.make("gymnasium_env/GridWorld-v0", size=env_size, max_steps=99).env
    env.load_rewards(rewardDictionary)

    env = Monitor(env)
    model = PPO("MlpPolicy", env=env, seed=None, verbose=0, tensorboard_log=path, device = 'cuda', **sampled_hyperparams)

    stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=30, min_evals=50, verbose=1)
    eval_callback = TrialEvalCallback(
        env, trial, best_model_save_path=path, log_path=path,
        n_eval_episodes=5, eval_freq=10000, deterministic=False, callback_after_eval=stop_callback
    )

    params = sampled_hyperparams
    with open(f"{path}/params.txt", "w") as f:
        f.write(str(params))

    try:
        model.learn(50000, callback=eval_callback)
        env.close()
    except (AssertionError, ValueError) as e:
        env.close()
        print(e)
        print("============")
        print("Sampled params:")
        pprint(params)
        raise optuna.exceptions.TrialPruned()

    is_pruned = eval_callback.is_pruned
    reward = eval_callback.best_mean_reward

    del model.env
    del model

    if is_pruned:
        raise optuna.exceptions.TrialPruned()

    return reward


if __name__ == "__main__":

    sampler = TPESampler(n_startup_trials=10, multivariate=True)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=10)

    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
        direction="maximize",
    )

    try:
        study.optimize(objective, n_jobs=1, n_trials=128, show_progress_bar = True)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    trial = study.best_trial
    print(f"Best trial: {trial.number}")
    print("Value: ", trial.value)

    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    study.trials_dataframe().to_csv(f"{study_path}/report.csv")

    with open(f"{study_path}/study.pkl", "wb+") as f:
        pkl.dump(study, f)

    try:
        fig1 = plot_optimization_history(study)
        fig2 = plot_param_importances(study)
        fig3 = plot_parallel_coordinate(study)

        fig1.show()
        fig2.show()
        fig3.show()

    except (ValueError, ImportError, RuntimeError) as e:
        print("Error during plotting")
        print(e)