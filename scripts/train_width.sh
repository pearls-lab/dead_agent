python train.py --algo dqn --env gridworld --layers 1 --parameters 512 --gradient_steps 4 --reward_dict ./two_wall.json > ./logs/two_wall/algo_dqn_layers_1_parameters_512_grad_steps_4.txt
echo "grad steps 4 for parameters 512 for layers 1 for algos dqn finished"
python train.py --algo dqn --env gridworld --layers 1 --parameters 1024 --gradient_steps 4 --reward_dict ./two_wall.json > ./logs/two_wall/algo_dqn_layers_1_parameters_1024_grad_steps_4.txt
echo "grad steps 4 for parameters 1024 for layers 1 for algos dqn finished"
python train.py --algo dqn --env gridworld --layers 1 --parameters 2048 --gradient_steps 4 --reward_dict ./two_wall.json > ./logs/two_wall/algo_dqn_layers_1_parameters_2048_grad_steps_4.txt
echo "grad steps 4 for parameters 2048 for layers 1 for algos dqn finished"
python train.py --algo dqn --env gridworld --layers 2 --parameters 512 --gradient_steps 4 --reward_dict ./two_wall.json > ./logs/two_wall/algo_dqn_layers_2_parameters_512_grad_steps_4.txt
echo "grad steps 4 for parameters 512 for layers 2 for algos dqn finished"
python train.py --algo dqn --env gridworld --layers 2 --parameters 1024 --gradient_steps 4 --reward_dict ./two_wall.json > ./logs/two_wall/algo_dqn_layers_2_parameters_1024_grad_steps_4.txt
echo "grad steps 4 for parameters 1024 for layers 2 for algos dqn finished"
python train.py --algo dqn --env gridworld --layers 2 --parameters 2048 --gradient_steps 4 --reward_dict ./two_wall.json > ./logs/two_wall/algo_dqn_layers_2_parameters_2048_grad_steps_4.txt
echo "grad steps 4 for parameters 2048 for layers 2 for algos dqn finished"