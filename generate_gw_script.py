algos = ["dqn"]
layers = [2]
parameters = [512]
grad_steps = [6]
lrs = [0.0001, 0.00001, 0.000001, 0.0000001]

# for algo in algos:
#     for layer in layers:
#         for parameter in parameters:
#             for grad_step in grad_steps:
#                 print(f'python train.py --algo {algo} --env gridworld --layers {layer} --parameters {parameter} --gradient_steps {grad_step} --reward_dict ./two_wall.json > ./logs/two_wall/algo_{algo}_layers_{str(layer)}_parameters_{str(parameter)}_grad_steps_{str(grad_step)}.txt')
#                 print(f'echo "grad steps {grad_step} for parameters {parameter} for layers {layer} for algos {algo} finished"')

script_id = "learning_rate"

for algo in algos:
    for layer in layers:
        for parameter in parameters:
            for grad_step in grad_steps:
                for lr in lrs:
                    # print(f'python train.py --algo {algo} --env gridworld --layers {layer} --parameters {parameter} --gradient_steps {grad_step} --reward_dict ./two_wall.json > ./logs/two_wall/algo_{algo}_layers_{str(layer)}_parameters_{str(parameter)}_grad_steps_{str(grad_step)}.txt')
                    print(f'python train.py --algo {algo} --env gridworld --layers {layer} --parameters {parameter} --gradient_steps {grad_step} --reward_dict ./two_wall.json --lr {lr} --script_id {script_id} > ./logs/two_wall/algo_{algo}_lr_{lr}.txt')
                    print(" ")
                    # print(f'echo "grad steps {grad_step} for parameters {parameter} for layers {layer} for algos {algo} finished"')
                    print(f'echo "Learning rate {lr} finished"')
                    print(" ")