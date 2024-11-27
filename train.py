import gymnasium as gym
import argparse
import os
import json
from gridworld_env import GridWorldEnv
from algos.val_it import ValueIteration
from sb3.stable_baselines3 import A2C, DQN, PPO

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

    parser.set_defaults(gat=True)
    args   = parser.parse_args()
    params = vars(args)
    return params


if __name__ == '__main__':
    args             = parse_args()

    rewardDictionary = {
        "goal": 
        {
            "location": [9,9],
            "reward": 5,
            "available": True,
            "terminal": True
        },
        "start":
        {
            "location": [0,0],
            "reward": 0,
            "available": True,
            "terminal": False
        }
    }

    # save_location    = format_save_file(params)
    # print(f'Saving files at {save_location}')
    # with open('./' + save_location + '/args.json', 'w') as fp: json.dump(params, fp)

    env_size   = 10
    env        = gym.make("gymnasium_env/GridWorld-v0", size=env_size)
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

    # # Testing PPO
    # model = PPO("MlpPolicy", env, verbose=1, device = 'cuda')
    # model.learn(total_timesteps=100000)
    # model.save("gw_test")
    
    # Testing TRPO
    model = PPO("MlpPolicy", env, verbose=1, device = 'cuda')
    model.learn(total_timesteps=100000)
    model.save("gw_test")
    obs, info = env.reset()
    env.render()

    for i in range(20):
        action, _state = model.predict(obs, deterministic=True)
        print("Action", action)
        # action, _state = model.predict(FlattenObservation(obs), deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done:
            break

    print("last reward:", reward)
    # # print(env.observation_space)
    # # print(env.get_grid())
    # # print(env.size)
    # # print("obs", obs)
    # # print("Flattened grid:", env.get_flattened_grid())
    # # print(info)
