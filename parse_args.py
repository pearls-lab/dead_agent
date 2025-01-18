import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ddsp', default= False, type=bool)
    parser.add_argument('--snr', default= False, type=bool)
    parser.add_argument('--sp', default=-1, type=int,
                        help="step penalty")
    parser.add_argument('--iter', default=5, type=int,
                        help="Iterations for val_it and tdmpc") # tdmpc: iterations
    parser.add_argument('--gamma', default='.99', type=float,
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
    parser.add_argument('--trials', default=1, type=int,
                        help="number of runs to do")
    parser.add_argument('--lr', default=0.001, type=float,
                        help="default learning rate")
    parser.add_argument('--teacher_force', default=False, type=bool,
                        help="gridworld only rn: will cause the agent to use the most optimal trajectory found")
    parser.add_argument('--script_id', default="default_name", type=str,
                        help="Script identifier for wandb")
    parser.add_argument('--death_timer', default=1, type=int,
                        help="number of steps agent can take before dying after stepping in a dead area.")
    parser.add_argument('--final_exploration_rate', default=0.05, type=float,
                        help="Final exploration rate for agents (decays from 1)")
    parser.add_argument('--safe_rl', default=False, type=bool,
                        help="Experimental: Uses CMDP paradigm instead")
    
    # DQN
    parser.add_argument('--target_net_update', default=100, type=float,
                        help="How often to update the target network")
    parser.add_argument('--buffer_size', default=100000, type=int,
                        help="buffer size")
    parser.add_argument('--multi_buffer', default=False, type=bool,
                        help="Experimental: Add a second buffer for specfic trajectories")
    parser.add_argument('--double_dqn', default=False, type=bool,
                        help="Uses double dqn")
    parser.add_argument('--gradient_steps', default=1, type=int,
                        help="number of gradient steps")

    # PPO
    parser.add_argument('--n_steps', default=2048, type=int,
                        help="The number of steps to run for each environment per update (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)")
    parser.add_argument('--batch_size', default=64, type=int,
                        help="Minibatch size")
    parser.add_argument('--gae_lambda', default=0.95, type=float,
                        help="Factor for trade-off of bias vs variance for Generalized Advantage Estimator")
    parser.add_argument('--n_epochs', default=10, type=int,
                        help="Number of epoch when optimizing the surrogate loss")
    parser.add_argument('--ent_coef', default=0.0, type=float,
                        help="Entrophy coefficient")
    parser.add_argument('--clip_range', default=0.2, type=float,
                        help="Clipping parameter, it can be a function of the current progress remaining (from 1 to 0)")
    parser.add_argument('--vf_coef', default=0.0, type=float,
                        help="Value function coefficient for the loss calculation")
    parser.add_argument('--mgm', default=0.5, type=float,
                        help="Max grad norm")

    

    parser.set_defaults(gat=True)
    args   = parser.parse_args()
    params = vars(args)
    return params