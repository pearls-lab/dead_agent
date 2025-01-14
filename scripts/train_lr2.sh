python train.py --algo dqn --env gridworld --layers 2 --parameters 512 --gradient_steps 6 --reward_dict ./two_wall.json --lr 1e-06 --script_id learning_rate > ./logs/two_wall/algo_dqn_lr_1e-06.txt
 
echo "Learning rate 1e-06 finished"
 
python train.py --algo dqn --env gridworld --layers 2 --parameters 512 --gradient_steps 6 --reward_dict ./two_wall.json --lr 1e-07 --script_id learning_rate > ./logs/two_wall/algo_dqn_lr_1e-07.txt
 
echo "Learning rate 1e-07 finished"