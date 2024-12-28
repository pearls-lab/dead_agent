#!/bin/bash

# Parameters: 64
echo "Training Started"
python train.py --algo dqn --env gridworld --layers 1 --parameters 64 --reward_dict ./two_wall.json > ./logs/two_wall/64.txt
python train.py --algo dqn --env gridworld --layers 2 --parameters 64 --reward_dict ./two_wall.json > ./logs/two_wall/64_64.txt
python train.py --algo dqn --env gridworld --layers 3 --parameters 64 --reward_dict ./two_wall.json > ./logs/two_wall/64_64_64.txt
python train.py --algo dqn --env gridworld --layers 4 --parameters 64 --reward_dict ./two_wall.json > ./logs/two_wall/64_64_64_64.txt
python train.py --algo dqn --env gridworld --layers 5 --parameters 64 --reward_dict ./two_wall.json > ./logs/two_wall/64_64_64_64_64.txt
python train.py --algo dqn --env gridworld --layers 6 --parameters 64 --reward_dict ./two_wall.json > ./logs/two_wall/64_64_64_64_64_64.txt
python train.py --algo dqn --env gridworld --layers 7 --parameters 64 --reward_dict ./two_wall.json > ./logs/two_wall/64_64_64_64_64_64_64.txt
python train.py --algo dqn --env gridworld --layers 8 --parameters 64 --reward_dict ./two_wall.json > ./logs/two_wall/64_64_64_64_64_64_64_64.txt
python train.py --algo dqn --env gridworld --layers 9 --parameters 64 --reward_dict ./two_wall.json > ./logs/two_wall/64_64_64_64_64_64_64_64_64.txt
python train.py --algo dqn --env gridworld --layers 10 --parameters 64 --reward_dict ./two_wall.json > ./logs/two_wall/64_64_64_64_64_64_64_64_64_64.txt

echo "64 layers done" 

# Parameters: 128
python train.py --algo dqn --env gridworld --layers 1 --parameters 128 --reward_dict ./two_wall.json > ./logs/two_wall/128.txt
python train.py --algo dqn --env gridworld --layers 2 --parameters 128 --reward_dict ./two_wall.json > ./logs/two_wall/128_128.txt
python train.py --algo dqn --env gridworld --layers 3 --parameters 128 --reward_dict ./two_wall.json > ./logs/two_wall/128_128_128.txt
python train.py --algo dqn --env gridworld --layers 4 --parameters 128 --reward_dict ./two_wall.json > ./logs/two_wall/128_128_128_128.txt
python train.py --algo dqn --env gridworld --layers 5 --parameters 128 --reward_dict ./two_wall.json > ./logs/two_wall/128_128_128_128_128.txt
python train.py --algo dqn --env gridworld --layers 6 --parameters 128 --reward_dict ./two_wall.json > ./logs/two_wall/128_128_128_128_128_128.txt
python train.py --algo dqn --env gridworld --layers 7 --parameters 128 --reward_dict ./two_wall.json > ./logs/two_wall/128_128_128_128_128_128_128.txt
python train.py --algo dqn --env gridworld --layers 8 --parameters 128 --reward_dict ./two_wall.json > ./logs/two_wall/128_128_128_128_128_128_128_128.txt
python train.py --algo dqn --env gridworld --layers 9 --parameters 128 --reward_dict ./two_wall.json > ./logs/two_wall/128_128_128_128_128_128_128_128_128.txt
python train.py --algo dqn --env gridworld --layers 10 --parameters 128 --reward_dict ./two_wall.json > ./logs/two_wall/128_128_128_128_128_128_128_128_128_128.txt

echo "128 layers done" 

# Parameters: 256
python train.py --algo dqn --env gridworld --layers 1 --parameters 256 --reward_dict ./two_wall.json > ./logs/two_wall/256.txt
python train.py --algo dqn --env gridworld --layers 2 --parameters 256 --reward_dict ./two_wall.json > ./logs/two_wall/256_256.txt
python train.py --algo dqn --env gridworld --layers 3 --parameters 256 --reward_dict ./two_wall.json > ./logs/two_wall/256_256_256.txt
python train.py --algo dqn --env gridworld --layers 4 --parameters 256 --reward_dict ./two_wall.json > ./logs/two_wall/256_256_256_256.txt
python train.py --algo dqn --env gridworld --layers 5 --parameters 256 --reward_dict ./two_wall.json > ./logs/two_wall/256_256_256_256_256.txt
python train.py --algo dqn --env gridworld --layers 6 --parameters 256 --reward_dict ./two_wall.json > ./logs/two_wall/256_256_256_256_256_256.txt
python train.py --algo dqn --env gridworld --layers 7 --parameters 256 --reward_dict ./two_wall.json > ./logs/two_wall/256_256_256_256_256_256_256.txt
# python train.py --algo dqn --env gridworld --layers 8 --parameters 256 --reward_dict ./two_wall.json > ./logs/two_wall/256_256_256_256_256_256_256_256.txt
# python train.py --algo dqn --env gridworld --layers 9 --parameters 256 --reward_dict ./two_wall.json > ./logs/two_wall/256_256_256_256_256_256_256_256_256.txt
# python train.py --algo dqn --env gridworld --layers 10 --parameters 256 --reward_dict ./two_wall.json > ./logs/two_wall/256_256_256_256_256_256_256_256_256_256.txt

echo "256 layers done" 

# Parameters 512
python train.py --algo dqn --env gridworld --layers 1 --parameters 512 --reward_dict ./two_wall.json > ./logs/two_wall/512.txt
python train.py --algo dqn --env gridworld --layers 2 --parameters 512 --reward_dict ./two_wall.json > ./logs/two_wall/512_512.txt
python train.py --algo dqn --env gridworld --layers 3 --parameters 512 --reward_dict ./two_wall.json > ./logs/two_wall/512_512_512.txt
python train.py --algo dqn --env gridworld --layers 4 --parameters 512 --reward_dict ./two_wall.json > ./logs/two_wall/512_512_512_512.txt
python train.py --algo dqn --env gridworld --layers 5 --parameters 512 --reward_dict ./two_wall.json > ./logs/two_wall/512_512_512_512_512.txt
# python train.py --algo dqn --env gridworld --layers 6 --parameters 512 --reward_dict ./two_wall.json > ./logs/two_wall/512_512_512_512_512_512.txt
# python train.py --algo dqn --env gridworld --layers 7 --parameters 512 --reward_dict ./two_wall.json > ./logs/two_wall/512_512_512_512_512_512_512.txt
# python train.py --algo dqn --env gridworld --layers 8 --parameters 512 --reward_dict ./two_wall.json > ./logs/two_wall/512_512_512_512_512_512_512_512.txt
# python train.py --algo dqn --env gridworld --layers 9 --parameters 512 --reward_dict ./two_wall.json > ./logs/two_wall/512_512_512_512_512_512_512_512_512.txt
# python train.py --algo dqn --env gridworld --layers 10 --parameters 512 --reward_dict ./two_wall.json > ./logs/two_wall/512_512_512_512_512_512_512_512_512_512.txt

echo "512 layers done" 

# Parameters 1024
python train.py --algo dqn --env gridworld --layers 1 --parameters 1024 --reward_dict ./two_wall.json > ./logs/two_wall/1024.txt
python train.py --algo dqn --env gridworld --layers 2 --parameters 1024 --reward_dict ./two_wall.json > ./logs/two_wall/1024_1024.txt
python train.py --algo dqn --env gridworld --layers 3 --parameters 1024 --reward_dict ./two_wall.json > ./logs/two_wall/1024_1024_1024.txt
python train.py --algo dqn --env gridworld --layers 4 --parameters 1024 --reward_dict ./two_wall.json > ./logs/two_wall/1024_1024_1024_1024.txt
python train.py --algo dqn --env gridworld --layers 5 --parameters 1024 --reward_dict ./two_wall.json > ./logs/two_wall/1024_1024_1024_1024_1024.txt
# python train.py --algo dqn --env gridworld --layers 6 --parameters 1024 --reward_dict ./two_wall.json > ./logs/two_wall/1024_1024_1024_1024_1024_1024.txt
# python train.py --algo dqn --env gridworld --layers 7 --parameters 1024 --reward_dict ./two_wall.json > ./logs/two_wall/1024_1024_1024_1024_1024_1024_1024.txt
# python train.py --algo dqn --env gridworld --layers 8 --parameters 1024 --reward_dict ./two_wall.json > ./logs/two_wall/1024_1024_1024_1024_1024_1024_1024_1024.txt
# python train.py --algo dqn --env gridworld --layers 9 --parameters 1024 --reward_dict ./two_wall.json > ./logs/two_wall/1024_1024_1024_1024_1024_1024_1024_1024_1024.txt
# python train.py --algo dqn --env gridworld --layers 10 --parameters 1024 --reward_dict ./two_wall.json > ./logs/two_wall/1024_1024_1024_1024_1024_1024_1024_1024_1024_1024.txt
