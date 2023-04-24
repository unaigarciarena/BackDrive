import random
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import argparse
import numpy as np
from deap import creator, base, tools
from EDAs import create_population, UMDA, AdvEDA, BackDriveEDA


def create_fibonacci_init_conf(n):
    si, sj = [1], [0]
    for i in range(1, n):
        aux = sj
        sj = si+sj
        si = aux
    return np.array(sj)


def print_protein(vector):
    global HPInitConf
    pos = off_find_pos(2.0*np.pi*vector-np.pi)  # OffFindPos translates the vector of angles to the  positions into the lattice.
    size_chain = HPInitConf.shape[0]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    for i in range(size_chain):
        if HPInitConf[i] == 0:
            plt.plot(pos[i, 0], pos[i, 1], 'g*')
        else:
            plt.plot(pos[i, 0], pos[i, 1], 'ro')
    plt.plot(pos[:, 0], pos[:, 1], 'b-')
    val = eval_off_lattice(vector)
    plt.text(1, 1, "Energy: "+str(val))
    fig.savefig('Prot.png')
    plt.show()


def off_find_pos(vector):
    """
    OffFindPos translates the vector of angles to the  positions into the lattice.
    INPUTS
    vector: Sequence of residues ( (H)ydrophobic or (P)olar, respectively represented by zero and one)
    OUTPUTS
    Pos: Matrix of positions Pos(sizechain,2) calculated from the angles
    """

    size_chain = HPInitConf.shape[0]

    pos = np.zeros((size_chain, 2))
    pos[0, :] = [0, 0]   # Position for the initial molecule
    pos[1, :] = [0, 1]   # Position for the second molecule

    for j in range(2, size_chain):

        pos[j, 0] = pos[j-1, 0]+np.sin(np.sum(vector[1:j]))
        pos[j, 1] = pos[j-1, 1]+np.cos(np.sum(vector[1:j]))

    return pos


def eval_off_lattice(vector):

    """
    Computes the energy of a protein configuration. First the protein is located in a lattice.
    Then, the energy is calculated
    Previous to call the function, the global variable HPInitConf, with the sequence of residues
    should be initialized with the function CreateFibbInitConf

    INPUTS
    vector: Angles of the residues
    OUTPUTS
    Energy value to be minimized.
    """
    global HPInitConf
    size_chain = HPInitConf.shape[0]

    # The position of the sequence in the lattice are calculated
    vector = vector + shift_vector
    vector = np.array([i-1 if i > 1 else i for i in vector])
    vector = 2.0*np.pi*vector-np.pi
    #print(vector)

    pos = off_find_pos(vector)

    tot_ab = 0
    tot_cos = 0

    for i in range(0, size_chain-1):

        #  Cosine part of the evaluation
        if i == 0:
            tot_cos = 0
        else:
            tot_cos = tot_cos + (1-np.cos(vector[i]))  # The 0.25 is multiplied at the  end
        # Distance part of the evaluation
        for j in range(i+2, size_chain):
            dist = np.sqrt((pos[i, 0]-pos[j, 0])**2 + (pos[i, 1]-pos[j, 1])**2)
            if HPInitConf[i] == 1 and HPInitConf[j] == 1:
                ab_effect = 1
            elif HPInitConf[i] == 0 and HPInitConf[j] == 0:
                ab_effect = 0.5
            else:
                ab_effect = -0.5

            tot_ab = tot_ab + (1/(dist**12) - ab_effect/(dist**6))

    ev = tot_cos*0.25 + tot_ab*4

    return ev


def ea_generate_update(toolbox, ngen, halloffame=None, stats=None, verbose=__debug__):
    """This is algorithm implements the ask-tell model proposed in
    [Colette2010]_, where ask is called `generate` and tell is called `update`.

    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.

    :returns: The final population.

    The toolbox should contain a reference to the generate and the update method
    of the chosen strategy.

    .. [Colette2010] Collette, Y., N. Hansen, G. Pujol, D. Salazar Aponte and
       R. Le Riche (2010). On Object-Oriented Programming of Optimizers -
       Examples in Scilab. In P. Breitkopf and R. F. Coelho, eds.:
       Multidisciplinary Design Optimization in Computational Mechanics,
       Wiley, pp. 527-565;

    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    population = None

    for gen in range(ngen):

        # Generate a new population
        population = toolbox.generate()

        if halloffame is not None:
            halloffame.update(population)

        # Update the strategy with the evaluated individuals
        toolbox.update(population)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def random_search(evaluations, eval_fun):
    pop = create_population(number_variables, evaluations)
    fitness = list(map(eval_fun, pop))
    index = np.argmin(fitness)
    return fitness[index], pop[index, :]


def main(seed):

    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    solvers = [random_search, UMDA, AdvEDA, BackDriveEDA]
    reinit = 1
    h_layers = 3
    solver = solvers[svr]
    measure = 0
    up = 1
    low = 0
    sel_method = 1
    offspring = 0.5
    tour_size = 10
    gens = 300

    print("Params:")
    print("\t Layers: " + str(h_layers))
    print("\t Solver: " + solver.__name__)
    print("\t Reinit: " + str(reinit))
    print("\t Measure: " + str(measure))
    print("\t Selection: " + str(sel_method))
    print("\t Seed: " + str(seed))
    if "random" in solver.__name__:
        fit, ind = random_search(int((gens / 2 + 0.5) * pop_size), eval_off_lattice)
        return fit, ind
    elif "UMDA" in solver.__name__:
        strategy = UMDA(eval_off_lattice, pop_size, number_variables, method=sel_method, offspring=offspring, tour_size=tour_size)
    elif "Adv" in solver.__name__:
        strategy = AdvEDA(eval_off_lattice, pop_size, number_variables, method=sel_method, offspring=offspring, sel_pop_size=pop_size//3)
    elif "Back" in solver.__name__:
        strategy = BackDriveEDA(eval_off_lattice, pop_size, number_variables, method=sel_method, offspring=offspring)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", eval_off_lattice)

    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    hof = tools.HallOfFame(50, similar=np.array_equal)
    stats = tools.Statistics(lambda individual: individual.fitness.values)
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)

    ea_generate_update(toolbox, ngen=gens, stats=stats, halloffame=hof)

    return hof[0].fitness.values[0], hof[0]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('integers', metavar='int', type=int, choices=range(50001), nargs='+', help='an integer in the range 0..3000')
    parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, default=max, help='sum the integers (default: find the max)')
    args = parser.parse_args()
    my_seed = args.integers[0]               # Seed: Used to set different outcomes of the stochastic program
    dimension = args.integers[1]
    svr = args.integers[2]
    pop_size = args.integers[3]

    print(my_seed, dimension, svr, pop_size)
    # shift_vector = np.random.rand(number_variables)
    shift_vectors = np.load("shift_vectors_" + str(dimension) + ".npy")
    shift_vector = shift_vectors[my_seed - 1]

    HPInitConf = create_fibonacci_init_conf(dimension)   # [6-13,7-21,8-34,9-55]
    objectives = 1
    number_variables = HPInitConf.shape[0]

    print(time.time())
    a, b = main(my_seed)
    print(time.time())
    np.savetxt("results/BestInd_VAE_EE_Seed" + str(my_seed) + "_Dim" + str(dimension) + "_Svr" + str(svr) + "_Pop" + str(pop_size) + ".csv", np.concatenate((np.reshape(a,(1,1)), np.reshape(b, (1,-1))), axis=1))
