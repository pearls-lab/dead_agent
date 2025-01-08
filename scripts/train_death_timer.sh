python train.py --algo dqn --env gridworld --layers 1 --parameters 2048 --gradient_steps 4 --death_timer 1 --script_id "death timer" --trials 3 --reward_dict ./two_wall.json > ./logs/two_wall/dt1.txt

python train.py --algo dqn --env gridworld --layers 1 --parameters 2048 --gradient_steps 4 --death_timer 2 --script_id "death timer" --trials 3 --reward_dict ./two_wall.json > ./logs/two_wall/dt2.txt

python train.py --algo dqn --env gridworld --layers 1 --parameters 2048 --gradient_steps 4 --death_timer 3 --script_id "death timer" --trials 3 --reward_dict ./two_wall.json > ./logs/two_wall/dt3.txt

python train.py --algo dqn --env gridworld --layers 1 --parameters 2048 --gradient_steps 4 --death_timer 4 --script_id "death timer" --trials 3 --reward_dict ./two_wall.json > ./logs/two_wall/dt4.txt

python train.py --algo dqn --env gridworld --layers 1 --parameters 2048 --gradient_steps 4 --death_timer -1 --script_id "death timer" --trials 3 --reward_dict ./two_wall.json > ./logs/two_wall/rdt1.txt

python train.py --algo dqn --env gridworld --layers 1 --parameters 2048 --gradient_steps 4 --death_timer -2 --script_id "death timer" --trials 3 --reward_dict ./two_wall.json > ./logs/two_wall/rdt2.txt

python train.py --algo dqn --env gridworld --layers 1 --parameters 2048 --gradient_steps 4 --death_timer -3 --script_id "death timer" --trials 3 --reward_dict ./two_wall.json > ./logs/two_wall/rdt3.txt

python train.py --algo dqn --env gridworld --layers 1 --parameters 2048 --gradient_steps 4 --death_timer -4 --script_id "death timer" --trials 3 --reward_dict ./two_wall.json > ./logs/two_wall/rdt4.txt