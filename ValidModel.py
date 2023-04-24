import numpy as np
import os
import random
from math_ops import boltzmann, deceptive, one_max, fitness_function, trap
from netTraining import back_drive_net, train
from structure import initialize_variables, initialize_back_variables
import tensorflow as tf
import argparse
save_step = 5000
display_step = 200

"""
This function creates a binary solution of a desired size. If provided, the solution will correspond to
the binary form of a given integer
"""


def create_individual(ind_size, index=-1):
    """
    :param ind_size: Size of the solution
    :param index: Integer to which binary form the solution will correspond, if provided
    :return:
    """
    if index == -1:  # If not a desired binary representation of certain integer, create a random int
        index = random.randint(0, 2**ind_size-1)
    ind = [int(xi) for xi in list(str(bin(index)))[2:]]  # Binarize integer

    aux = [0] * (ind_size-len(ind))  # Add 0's at the start if the length is not enough to match target
    ind = aux + ind
    return ind

"""
Given a size of the solutions and the amount of solutions (n) desired this function returns a 
population. Additionally, if desired, the solutions in the population will correspond to the binary 
forms of the first n integers.
"""


def create_population(ind_size, pop_size, index=False):
    """
    :param ind_size: Amount of variables in the function
    :param pop_size: Amount of solutions in the population
    :param index: Booleans, whether the solutions will correspond to the first integers (True) or not
    :return: the population according to the mentioned criteria
    """
    population = []
    for fixed_ind in range(pop_size):
        if index:
            ind = create_individual(ind_size, fixed_ind)
        else:
            ind = create_individual(ind_size)
        population.append(ind)
    population = np.array(population)

    return population

"""
Auxiliary function. This function provides the location of the last model saved in the previous
training phase, independently of whether the training was successfully carried out, or abruptly ended.
"""


def restore_file(path, name1, name2):
    """
    :param path: Path where the models are being saved
    :param name1: Name of the file created in case the training was successfully ended
    :param name2: Name of the file created in case the training was abruptly ended
    :return: The file name from which
    """
    files = os.listdir(path)
    f = None
    epoch = 0
    for file in files:

        if name1 in file:
            pt = file.rfind(".")
            gn = file.find("-")
            f = file[:pt]
            epoch = int(file[gn+1:pt])
            break
        elif name2 in file:
            pt = file.rfind(".")
            gn = file.find("-")
            ep = int(file[gn+1:pt])
            if ep > epoch:
                f = file[:pt]
                epoch = ep

    return f, epoch


"""
Auxiliary function. This function deletes the models saved from past runs
"""


def delete_checkpoint(path, file):

    p = path + file + ".index"
    if os.path.isfile(p):
        os.remove(p)

    p = path + file + ".meta"
    if os.path.isfile(p):
        os.remove(p)

    p = path + file + ".data-00000-of-00001"
    if os.path.isfile(p):
        os.remove(p)

"""
Main function. It receives the network configuration and whether the net is going to be reinitialized
or not and whether training is going o occur. It performs according to all those parameters. It constructs (or loads)
a network, trains it (if it is required to) and returns all the interesting components of the net (such as the 
input/output, weights, biases, ...)
"""


def main(start, training, population, fitness, convergence, batch_size, learning_rate, ind_len, sigma, hidden_neurons, groups, seed):
    """
    :param start: Boolean, whether the net is going to be loaded (False) or not (True)
    :param training: Boolean, whether training is going to occur (True) or not (False)
    :param population: The population of solutions
    :param fitness: Fitness scores ove which the net is trained
    :param convergence: Accuracy desired from the model
    :param batch_size: Amount of entries in each training epoch
    :param learning_rate: Multiplier for the gradients
    :param ind_len: Amount of variables in the problem
    :param sigma: Parameter for the logistic function used to binarize solutions
    :param hidden_neurons0: Amount of neurons in the first hidden layer
    :param hidden_neurons1: Amount of neurons in the second hidden layer
    :param groups: Variable groups in the ADF
    :return: The TF components of the trained network
    """
    training = start or training

    if not start:
        path, epoch = restore_file("Board/", "result.ckpt", "model.ckpt")
        sess, x, y, weights, hidden_weights, biases, hidden_biases, net, loss, optimizer, sum_ford, saver, writer, log = initialize_variables(input_size=ind_len, h_neurons=hidden_neurons, batch_size=batch_size, learning_rate=learning_rate, sigma=sigma, variable_groups=groups, delete=start, structure=structure)
        saver.restore(sess, "Board/" + path)
    if training:
        if start:
            sess, x, y, weights, hidden_weights, biases, hidden_biases, net, loss, optimizer, sum_ford, saver, writer, log, preds = train(net_input=population, net_output=fitness, start=start, sigma=sigma, hidden_neurons=hidden_neurons, batch_size=batch_size, learning_rate=learning_rate, display_step=display_step, save_step=save_step, groups=groups, structure=structure)
        else:
            preds = train(net_input=population, net_output=fitness, start=start, sess=sess, x=x, y=y, net=net, loss=loss, optimizer=optimizer, sum_ford=sum_ford, saver=saver, writer=writer, log=log, epoch=epoch, batch_size=batch_size, display_step=10, save_step=10000, groups=groups, structure=structure)
            delete_checkpoint("Board/", path)
        preds.to_csv("results/Predictions" + str(seed) + "-" + str(structure) + "-" + str(function) + "-" + str(variables) + "-" + str(overlap) + ".csv")
    return sess, net, x, weights, hidden_weights, biases, hidden_biases


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("a", nargs='+')
    args = parser.parse_args().__dict__["a"]
    seed = int(args[0])  # Seed: Used to set different outcomes of the stochastic program
    structure = int(args[1])  # Whether to use the structure that exploits the ADF characteristic
    function = int(args[2])  # 0: OneMax, 1: Trap
    variables = int(args[3])  # Number of variables
    overlap = int(args[4])  # 0: No overlap, 1: Overlap

    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Common parameters
    batch_size = 10
    sigma = 15.0
    t = 1  # Boltzmann t

    # Each sublist contains a group of variables that affect each subfunction in the ADF
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
    #np.random.shuffle(pop)

    scores = [fitness_function(groups, funcs, ind, params) for ind in pop]

    boltz = boltzmann(scores, t)
    scores = np.log(boltz)

    sess, net, x, weights, h_weights, biases, h_biases = main(start=True, training=True, population=pop, fitness=scores, convergence=1e-3, batch_size=batch_size, learning_rate=1e-3, ind_len=i, sigma=sigma, hidden_neurons=h_neurons, groups=groups, seed=seed)

    w, h_w, b, h_b = sess.run([weights, h_weights, biases, h_biases])

    learning_rate = 0.01
    convergence = 0.01

    samples = 10
    sam = np.random.beta(0.5, 0.5, samples)

    sam = sam*(np.max(scores)-np.min(scores))+np.min(scores)

    sample_scores = []

    for j in range(0, samples):

        print(j, "Samples obtained")
        individual, clipped_ind, target, back_weights, back_hidden_weights, back_biases, back_hidden_biases, back_net, perf_loss, apply_grads, grads, sum_back, saver, writer, log = initialize_back_variables(sess=sess, input_size=i, h_neurons=h_neurons, sigma=sigma, variable_groups=groups, structure=structure)
        back_drive_net(sess, convergence, np.max(scores), target, apply_grads, perf_loss, back_weights, back_hidden_weights, back_biases, back_hidden_biases, w, h_w, b, h_b, 5000, clipped_ind)

        index = np.where(np.all(pop == np.round(np.reshape(sess.run([clipped_ind]), (-1))), axis=1))
        print(sess.run(individual))
        sample_scores += [scores[index][0]]

    np.savetxt("results/Back" + str(seed) + "-" + str(structure) + "-" + str(function) + "-" + str(variables) + "-" + str(overlap) + ".txt", np.array([sam, sample_scores]))

