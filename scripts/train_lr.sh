python train.py --algo dqn --env gridworld --layers 2 --parameters 512 --gradient_steps 6 --reward_dict ./two_wall.json --lr 0.0001 --script_id learning_rate > ./logs/two_wall/algo_dqn_lr_0.0001.txt
 
echo "Learning rate 0.0001 finished"
 
python train.py --algo dqn --env gridworld --layers 2 --parameters 512 --gradient_steps 6 --reward_dict ./two_wall.json --lr 1e-05 --script_id learning_rate > ./logs/two_wall/algo_dqn_lr_1e-05.txt
 
echo "Learning rate 1e-05 finished"