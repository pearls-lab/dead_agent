# python train.py --algo ppo --env gridworld --layers 2 --parameters 512 --gradient_steps 8 --reward_dict ./two_wall.json --lr 0.0003 --vf_coef 0.05 --composer "debussy" --ent_coef 0.1 --batch_size 128 --script_id dead_ears_music_agent > ./logs/two_wall/algo_dqn_lr_0.0001.txt

python train.py --algo ppo --env gridworld --layers 2 --parameters 512 --gradient_steps 8 --reward_dict ./two_wall.json --lr 0.0003 --vf_coef 0.05 --composer "chpn" --ent_coef 0.1 --batch_size 128 --script_id dead_ears_music_agent_final > last_minute.txt
