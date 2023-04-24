from netTraining import train, initialize_variables, back_drive_net
from structure import initialize_back_variables
from EDAs import BackDriveEDA, UMDA
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
import argparse
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import stats
from deap import creator, base, tools


def load_data(force=True):
    total_path = "total_prelim.npy"
    total_df_path = "total_prelim.csv"
    seeds = 4
    outputs = 5
    percents = [0, 10, 20, 50]
    total = np.zeros((seeds, len(problems), outputs, len(percents), 2, 1000))
    if os.path.isfile(total_path) and not force:
        total = np.load(total_path)
    else:
        for seed in range(seeds):
            print(seed)
            for f in range(len(problems)):
                print("\t" + str(f))
                for ops in range(1, outputs+1):
                    print("\t\t" + str(ops))
                    for ip, percent in enumerate(percents):
                        print("\t\t\t" + str(percent))
                        for ns in [0, 1]:
                            path = "primary/fitnesses_" + str(seed) + "_10_1000_" + str(f) + "_" + str(ops) + "_" + str(percent) + "_" + str(ns) + ".npy"
                            if os.path.isfile(path):
                                print("tuputamadre")
                                aux = np.load(path)

                                total[seed, f, ops-1, ip, ns, :] = aux[1000:]
                            else:
                                print(path)
        for ip in range(len(problems)):
            print(np.min(total[:, ip, :, :, :, :]))
            total[:, ip, :, :, :, :] = total[:, ip, :, :, :, :] - np.min(total[:, ip, :, :, :, :])
            print(np.max(total[:, ip, :, :, :, :]))
            total[:, ip, :, :, :, :] = total[:, ip, :, :, :, :] / np.max(total[:, ip, :, :, :, :])
            print()
        np.save(total_path, total)
    if os.path.isfile(total_df_path) and not force:
        total_df = pd.read_csv(total_df_path)
    else:
        total_df = pd.DataFrame(columns=["Seed", "Problem", "Outputs", "Percentile", "Noise", "Fitness"])
        for seed in range(seeds):
            print(seed)
            for f in range(len(problems)):
                print("\t" + str(f))
                for ops in range(1, outputs+1):
                    print("\t\t" + str(ops))
                    for ip, percent in enumerate(percents):
                        print("\t\t\t" + str(percent))
                        for ns in [0, 1]:
                            aux = np.array([[seed] * 1000, [problem_names[f]] * 1000, [ops] * 1000, [percent] * 1000, ["Noisy" if ns == 1 else "Exact"] * 1000, total[seed, f, ops-1, ip, ns, :]]).T
                            total_df = pd.concat((total_df, pd.DataFrame(columns=["Seed", "Problem", "Outputs", "Percentile", "Noise", "Fitness"], data=aux)))

        total_df.to_csv(total_df_path)
    return total, total_df


def visualize():
    data = pd.read_csv("total_prelim.csv")
    data = data[data["Percentile"] < 50]
    sns.boxplot("Problem", "Fitness", "Percentile", data)
    plt.show()


def summations():
    data = np.load("primary/total_prelim.npy")

    seeds = 4
    outputs = 5
    percents = 4
    huehue = 30
    # (seeds, len(problems), outputs, len(percents), 2, 10000)
    scores = np.zeros((len(problems), outputs, percents, 2))
    yes = 0
    no = 0
    for f in range(len(problems)):
        print(f)
        for seed in range(seeds):
            print("\t" + str(seed))
            for ops in range(outputs):
                print("\t\t" + str(ops))
                for ip in range(percents):
                    for ns in range(2):
                        for opsj in range(ops, outputs):
                            for ipj in range(ip, percents):
                                for nsj in range(ns, 2):
                                    for j in range(10):

                                        samplej = data[seed, f, opsj, ipj, nsj, j*huehue:(j+1)*huehue]
                                        sample = data[seed, f, ops, ip, ns, j * huehue:(j + 1) * huehue]
                                        a = stats.kruskal(sample, samplej)
                                        if a[1] < 0.01:
                                            yes += 1
                                            if np.median(sample) < np.median(samplej):
                                                scores[f, ops, ip, ns] += 1
                                                scores[f, opsj, ipj, nsj] -= 1
                                            else:
                                                scores[f, ops, ip, ns] -= 1
                                                scores[f, opsj, ipj, nsj] += 1
                                        else:
                                            no += 1
                print(yes, no)
    np.save("scores_med.npy", scores)


problems = [F1, F2, F3, F4, F5, F6, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18, F19, F20, F21, F22, F23, F24]
problem_names = ["F1", "F2", "F3", "F4", "F5", "F6", "F8", "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19", "F20", "F21", "F22", "F23", "F24"]


def fitness(ind):
    ind = np.array(ind)*(np.array(problem.max_bounds) - np.array(problem.min_bounds)) + np.array(problem.min_bounds)
    return problem.objective_function(ind)


def test():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
    global problem
    for my_seed in range(30):
        for function in range(len(problems)):

            problem = problems[function](number_variables)

            for outputs in range(1, 6):

                for percentile in [0, 10, 20, 50]:
                    tgt = [percentile/100, np.log(percentile/100+0.01), np.sqrt(percentile/100), np.power(percentile/100, 2), np.sin(percentile/100)][:outputs]
                    for noise in [0, 1]:

                        engine = BackDriveEDA(fitness, pop_size, number_variables, outputs, 0, 0, retain_ws=0, retain_inds=0, sol_disc=0, train_m=0)
                        first_pop = engine.generate(creator.Individual)
                        fitnesses = np.array([fitness(x) for x in first_pop])
                        engine.update(first_pop, percentile, noise)
                        second_pop = engine.generate(creator.Individual)
                        new_fit = np.array([fitness(x) for x in second_pop])

                        all_fit = np.concatenate((fitnesses, new_fit))

                        np.save("fitnesses_" + str(my_seed) + "_" + str(number_variables) + "_" + str(pop_size) + "_" + str(function) + "_" + str(outputs) + "_" + str(percentile) + "_" + str(noise) + ".npy", all_fit)
                        engine.close()


def testUMDA():
    if not os.path.isfile("primary/withUMDA.csv"):
        global problem
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
        data = pd.read_csv("primary/total_prelim.csv", index_col=0)
        scales = [[304.7230601861877,96254.60041177456],[1744.479429543569,389350.5102609679],[2036397.4051678646,8709027473.740986],[-100574.18533294507,1096082.1383622119],[2068.554762144875,45902.20548292878],[20067244.63658706,143917458690.36066],[-119.60096533012387,1.9000288481478833],[-290.4072452388134,305.2222568580428],[-294.5736113121565,528.8240161679114],[99.35145254293326,23.039342304051885],[5269.851819749438,1577799.5301678746],[-125.28007599412223,16959.374644540898],[-296.10238073984453,2.3212604381378696],[619.5904173855874,1838.7674671512991],[364.3915409804609,2688.8407826462803],[160.4258871461939,4061.12170836879],[991.1873198522254,5556.305033231682],[964.8484013936767,5326.03221209098],[1022.4145125764264,5179.557845622093],[1418.6764230203607,1781.094067000282],[1355.059791690815,3428172.592008038],[1558.0958986613705,1980.8147457051296],[1036.2676005624494,2410.5364738810094]]
        for seed in range(4):
            print(seed)
            for function in range(len(problems)):
                print("\t" + str(function))
                problem = problems[function](number_variables)

                engine = UMDA(fitness, pop_size, number_variables, method=0, offspring=0.5)
                first_pop = engine.generate(creator.Individual)
                engine.update(first_pop)
                second_pop = engine.generate(creator.Individual)
                new_fit = np.array([fitness(x) for x in second_pop])
                new_fit = new_fit - scales[function][0]
                new_fit = new_fit / scales[function][1]
                aux = np.array([[seed] * 1000, [problem_names[function]] * 1000, [0] * 1000, ["UMDA"] * 1000, [0] * 1000, new_fit]).T
                data = pd.concat((data, pd.DataFrame(columns=["Seed", "Problem", "Outputs", "Percentile", "Noise", "Fitness"], data=aux)))
                #print(data)
        data.to_csv("primary/withUMDA.csv")
    else:
        data = pd.read_csv("primary/withUMDA.csv", index_col=0)
    print(data)
    sns.boxplot("Problem", "Fitness", "Percentile", data)
    plt.show()


def best_combination():
    data = np.load("scores_med.npy")
    bplot = pd.DataFrame(columns=["Outputs", "Percentile", "Noise", "Frequency"])
    outs = [r"$log(f)$", r"$f$", r"$\sqrt{f}$", r"$f^2$", r"$sin(f)$"]
    percentiles = [r"$0\%$", r"$10\%$", r"$20\%$", r"$50\%$"]
    noises = ["Yes", "No"]
    hmap = np.zeros((10, 6))
    ns = 3

    for o in outs:
        for p in percentiles:
            for n in noises:
                bplot.loc[bplot.shape[0]] = [o, p, n, 0]
    for problem in range(len(problem_names)-1):
        data1 = data[0]
        for n in range(ns):
            print(problem*ns+n)
            ind = np.unravel_index(np.argmax(data), data.shape)
            #bplot.set_value((bplot["Outputs"] == outs[ind[1]]) & (bplot["Percentile"] == percentiles[ind[2]]) & (bplot["Noise"] == noises[ind[3]]), "Frequency", 1 + bplot.loc[(bplot["Outputs"] == outs[ind[0]]) & (bplot["Percentile"] == percentiles[ind[1]]) & (bplot["Noise"] == noises[ind[2]])]["Frequency"])
            hmap[2*ind[1]+ind[3], ind[2]] += 1
            data[ind] = -100000
    fig, ax = plt.subplots()
    ax_c = ax.twinx()
    ax.imshow(hmap[:, :4])
    for i in range(10):
        for j in range(6):
            ax.text(j, i, hmap[i, j], ha="center", va="center", color="w" if hmap[i, j] < 9 else "b")
    ax.set_yticks(np.arange(0.5, 11.5, 2), minor=False)
    ax.set_yticklabels(outs, size="large")
    ax.set_yticks(np.arange(-0.5, 11.5, 2), minor=True)
    ax.yaxis.grid(True, which='minor', linewidth=2, color="w")

    ax.set_xticks(np.arange(0, 4, 1), minor=False)
    ax.set_xticklabels(percentiles, size="large")
    ax.xaxis.grid(True, which='minor', linewidth=2, color="w")
    ax.set_xticks(np.arange(0.5, 4.5, 1), minor=True)

    ax_c.set_yticks(np.arange(0.05, 1, 0.1), minor=False)
    ax_c.set_yticklabels(["Noisy", "Exact"]*6)
    plt.show()


def variances_percent():
    data = np.load("primary/total_prelim.npy")
    heat = np.zeros((23, 4))
    for percent in range(4):
        data1 = data[:, :, :, percent, :, :]
        for problem in range(23):
            for seed in range(4):
                for outputs in range(5):
                    for noise in range(2):
                        heat[problem, percent] += np.var(data1[seed, problem, outputs, noise])
    fig, ax = plt.subplots()
    heat = heat/
    ax.imshow(heat, aspect='auto')
    plt.yticks(np.arange(0, len(problem_names), 1), problem_names)
    plt.ylim((-0.5, len(problem_names)-0.5))
    plt.xticks(np.arange(0, 4, 1), [r"$0\%$", r"$10\%$", r"$20\%$", r"$50\%$"])

    for i in range(23):
        for j in range(4):
            ax.text(j, i, np.around(heat[i, j], decimals=3), ha="center", va="center", color="w" if heat[i, j] < 0.4 else "b")

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('integers', metavar='int', type=int, choices=range(2000001), nargs='+', help='an integer in the range 0..3000')
    parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, default=max, help='sum the integers (default: find the max)')
    args = parser.parse_args()
    number_variables = args.integers[0]      # Number of variables on each solution
    pop_size = args.integers[1]              # Number of individuals in each population
    #test()
    #load_data()
    #visualize()
    #summations()
    #best_combination()
    #testUMDA()
    variances_percent()