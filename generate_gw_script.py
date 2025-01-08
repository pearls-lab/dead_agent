algos = ["dqn"]
layers = [2, 3]
parameters = [16, 32, 64]
grad_steps = [6, 7, 8, 9, 10, 11, 12]

# for algo in algos:
#     for layer in layers:
#         for parameter in parameters:
#             for grad_step in grad_steps:
#                 print(f'python train.py --algo {algo} --env gridworld --layers {layer} --parameters {parameter} --gradient_steps {grad_step} --reward_dict ./two_wall.json > ./logs/two_wall/algo_{algo}_layers_{str(layer)}_parameters_{str(parameter)}_grad_steps_{str(grad_step)}.txt')
#                 print(f'echo "grad steps {grad_step} for parameters {parameter} for layers {layer} for algos {algo} finished"')


for algo in algos:
    for layer in layers:
        for parameter in parameters:
            for grad_step in grad_steps:
                print(f'python train.py --algo {algo} --env gridworld --layers {layer} --parameters {parameter} --gradient_steps {grad_step} --reward_dict ./two_wall.json > ./logs/two_wall/algo_{algo}_layers_{str(layer)}_parameters_{str(parameter)}_grad_steps_{str(grad_step)}.txt')
                print(f'echo "grad steps {grad_step} for parameters {parameter} for layers {layer} for algos {algo} finished"')