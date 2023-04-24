import random
from optproblems.cec2005.unimodal import F1, F2, F3, F4, F5
from optproblems.cec2005.basic_multimodal import F6, F8, F9, F10, F11, F12
from optproblems.cec2005.expanded_multimodal import F13, F14
from optproblems.cec2005.f15 import F15
from optproblems.cec2005.f16 import F16
from optproblems.cec2005.f17 import F17
from optproblems.cec2005.f18 import F18
from optproblems.cec2005.f19 import F19
from optproblems.cec2005.f20 import F20
from optproblems.cec2005.f21 import F21
from optproblems.cec2005.f22 import F22
from optproblems.cec2005.f23 import F23
from optproblems.cec2005.f24 import F24
import tensorflow as tf
import time
import argparse

from deap import creator, base, tools
from EDAs import create_population, UMDA, AdvEDA, BackDriveEDA

import numpy as np


# https://pypi.org/project/optproblems/
# https://ls11-www.cs.tu-dortmund.de/people/swessing/optproblems/doc/cec2005.html#test-problems
# http://www.cmap.polytechnique.fr/~nikolaus.hansen/Tech-Report-May-30-05.pdf

evals = []

def fitness(ind):
    global evals
    ind = np.array(ind)*(np.array(problem.max_bounds) - np.array(problem.min_bounds)) + np.array(problem.min_bounds)
    obj = problem.objective_function(ind)
    evals += [obj]
    return obj


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
        toolbox.update(population, 0, 0)

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
    sel_method = 1
    offspring = 0.5
    tour_size = 10

    print("Params:")
    print("\t Layers: " + str(h_layers))
    print("\t Solver: " + solver.__name__)
    print("\t Reinit: " + str(reinit))
    print("\t Measure: " + str(measure))
    print("\t Selection: " + str(sel_method))
    print("\t Seed: " + str(seed))
    if "random" in solver.__name__:
        fit, ind = random_search(int((gens / 2 + 0.5) * pop_size), fitness)
        return fit, ind
    elif "UMDA" in solver.__name__:
        strategy = UMDA(fitness, pop_size, number_variables, method=sel_method, offspring=offspring, tour_size=tour_size)
    elif "Adv" in solver.__name__:
        strategy = AdvEDA(fitness, pop_size, number_variables, method=sel_method, offspring=offspring, sel_pop_size=pop_size//3)
    elif "Back" in solver.__name__:
        strategy = BackDriveEDA(fitness, pop_size, number_variables, method=sel_method, offspring=offspring, outputs=outs, retain_inds=retain_inds, retain_ws=retain_ws, sol_disc=sol_disc, train_m=train_m)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", fitness)

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
    number_variables = args.integers[1]
    svr = args.integers[2]
    pop_size = args.integers[3]
    gens = args.integers[4]
    problems = [F1, F2, F3, F4, F5, F6, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20, F21, F22, F23, F24]
    p = args.integers[5]
    problem = problems[p](number_variables)
    outs = args.integers[6]
    retain_ws = args.integers[7]
    retain_inds = args.integers[8]
    sol_disc = args.integers[9]
    train_m = args.integers[10]
    print(my_seed, number_variables, svr, pop_size)
    # shift_vector = np.random.rand(number_variables)

    print(time.time())
    a, b = main(my_seed)
    print(time.time())
    np.save("Evals_" + str(my_seed) + "_" + str(number_variables) + "_" + str(svr) + "_" + str(pop_size) + "_" + str(gens) + "_" + str(p) + "_" + str(outs) + "_" + str(retain_ws) + "_" + str(retain_inds) + "_" + str(sol_disc) + "_" + str(train_m) + ".npy", np.array(evals))
