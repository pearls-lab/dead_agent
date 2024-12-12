from typing import Dict, Any, Union, Callable

import optuna
from torch import nn

def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func

def sample_dqn_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for DQN hyperparameters.

    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512, 1024])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    gamma = trial.suggest_categorical("gamma", [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    epsilon_start = trial.suggest_float("epsilon_start", 0.8, 1.0)
    epsilon_end = trial.suggest_float("epsilon_end", 0.01, 0.1)
    epsilon_decay = trial.suggest_int("epsilon_decay", 1000, 50000)
    target_update_frequency = trial.suggest_categorical("target_update_frequency", [100, 500, 1000, 5000])
    replay_buffer_size = trial.suggest_int("replay_buffer_size", 10000, 1000000)
    n_steps = trial.suggest_categorical("n_steps", [1, 2, 3, 4, 5])

    # Neural network architecture
    net_arch_width = trial.suggest_categorical("net_arch_width", [8, 16, 32, 64, 128, 256, 512])
    net_arch_depth = trial.suggest_int("net_arch_depth", 1, 3)
    activation_fn = trial.suggest_categorical("activation_fn", ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    net_arch = [dict(pi=[net_arch_width] * net_arch_depth, vf=[net_arch_width] * net_arch_depth)]

    return {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "epsilon_decay": epsilon_decay,
        "target_update_frequency": target_update_frequency,
        "replay_buffer_size": replay_buffer_size,
        "n_steps": n_steps,
        "policy_kwargs": dict(
            net_arch=net_arch,
            activation_fn=activation_fn,
        ),
    }
