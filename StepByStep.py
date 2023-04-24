import tensorflow as tf
from netTraining import train
import numpy as np
from ValidModel import create_population
from math_ops import fitness_function, trap, one_max

variables = 9
overlap = 0
function = 1
groups = []
funcs = []
params = []
i = 0
while i < variables-1:
    groups += [list(range(i, i+3))]
    if overlap == 1:
        i += 2
    else:
        i += 3
    if function == 1:
        funcs += [trap]
        params += [{"size": 3, "multiplier": 1}]
    else:
        funcs += [one_max]
        params += [None]
i = np.max(groups)+1

h_neurons = int(len(groups)*5)

pop = create_population(i, 2**i, True)
# np.random.shuffle(pop)

scores = np.array([fitness_function(groups, funcs, ind, params) for ind in pop])

print(pop, scores)
sess = tf.Session()
train(net_input=pop, net_output=scores, display_step=1, save_step=9999999, start=True, sigma=15.0, hidden_neurons=30, learning_rate=0.001, batch_size=20, groups=groups, structure=False)
