import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
from matplotlib import cm
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

seeds = 30
problems = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
retain_ws = [0, 1]
retain_inds = [0, 1]
sol_disc = [0, 1]
train_m = [0, 1]
root_path = "EvRes/"
total_path = "total_evo.npy"
total_df_path = "total_df_evo.csv"
evaluations = 100000


def load_data(force=True):

    if (not os.path.isfile(root_path + total_path)) or force:
        total = np.zeros((seeds, len(problems), 2, 2, 2, 2, 200000))
        for seed in range(seeds):
            print(seed)
            for ip, problem in enumerate(problems):
                print("\t" + str(problem))
                for rw in retain_ws:
                    for ri in retain_inds:
                        for sd in sol_disc:
                            for tm in train_m:
                                path = root_path + "Evals_" + str(seed) + "_10_3_1000_200_" + str(problem) + "_1_" + str(rw) + "_" + str(ri) + "_" + str(sd) + "_" + str(tm) + ".npy"
                                if os.path.isfile(path):
                                    data = np.load(path)
                                    aux = [np.max(data)]
                                    for i in range(evaluations):
                                        aux += [np.min([aux[-1], data[i]])]
                                    total[seed, ip, rw, ri, sd, tm, :evaluations] = aux[:evaluations]
                                else:
                                    print(path)

        for ip, problem in enumerate(problems):
            total[:, ip, :, :, :, :] -= np.min(total[:, ip, :, :, :, :])
            total[:, ip, :, :, :, :] += 0.001*np.max(total[:, ip, :, :, :, :])
            total[:, ip, :, :, :, :] /= np.max(total[:, ip, :, :, :, :])

        np.save(root_path + total_path, total)


def vis_curves():
    data = np.load(root_path + total_path)
    problem_names = ["F1", "F2", "F3", "F4", "F5", "F6", "F8", "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19", "F20", "F21", "F22", "F23", "F24"]
    markers = ["-.", "-", "--", ":"]
    viridis = cm.get_cmap('viridis', 100)
    for ip, problem in enumerate(problems):
        for rw in retain_ws:
            for ri in retain_inds:
                for sd in sol_disc:
                    for tm in train_m:
                        aux = []
                        for seed in range(seeds):
                            if np.sum(data[seed, ip, rw, ri, sd, tm]) != 0:
                                ax = data[seed, ip, rw, ri, sd, tm]
                                aux += [[np.array(ax)]]
        aux = np.concatenate(aux)
        print(aux.shape)
        aux = np.mean(aux, axis=0)
        plt.plot(np.log(aux), color=viridis(ip/len(problems)))
    plt.xlabel("Evaluations")
    plt.ylabel(r"$log(scale(f))$")
    plt.show()


def stackplot():
    if not os.path.isfile("linear_data.npy"):
        num = 3

        top = np.zeros((evaluations, 2, 2, 2, 2))
        data = np.load(root_path + total_path)

        data = np.mean(data, axis=0)
        for ip, problem in enumerate(problems):
            prob_data = data[ip]
            for ev in range(evaluations):
                aux = prob_data[:, :, :, :, ev]
                for i in range(num):
                    a = np.argmin(aux)
                    ind = np.unravel_index(a, aux.shape)
                    top[ev, ind[0], ind[1], ind[2], ind[3]] += 1
                    aux[ind] = np.max(aux)
        print(top[-1])
        linear_data = []

        labels = []
        for rw in retain_ws:
            for ri in retain_inds:
                for tm in train_m:
                    for sd in sol_disc:
                        linear_data += [top[:, rw, ri, sd, tm]]
        linear_data = np.array(linear_data)
        print(linear_data[:, -1])
        np.save("linear_data.npy", linear_data)
    else:
        linear_data = np.load("linear_data.npy")
    viridis = cm.get_cmap('viridis', 100)
    colors = [viridis(20), viridis(0.33), viridis(0.66), viridis(300)]
    stacks = plt.stackplot(range(evaluations), linear_data, colors=colors)
    print(stacks)
    patterns = ('-', '\\', '|','/', 'O', '.', '\\')
    for s in range(0, len(stacks), 4):
        stacks[s].set_hatch(patterns[s//4])
        stacks[s+1].set_hatch(patterns[s//4])
        stacks[s + 2].set_hatch(patterns[s // 4])
        stacks[s + 3].set_hatch(patterns[s // 4])

    plt.xlabel("Evaluations")
    plt.ylabel("Top3 Frequency")
    legend_elements = [Line2D([0], [0], marker='s', color=colors[0], label='Scatter',markerfacecolor=colors[0], markersize=15), Line2D([0], [0], marker='s', color=colors[1], label='Scatter',markerfacecolor=colors[1], markersize=15),Line2D([0], [0], marker='s', color=colors[2], label='Scatter',markerfacecolor=colors[2], markersize=15), Line2D([0], [0], marker='s', color=colors[3], label='Scatter',markerfacecolor=colors[3], markersize=15),
                       mpatches.Patch(edgecolor='black',alpha=1,hatch=patterns[0]), mpatches.Patch(edgecolor='black',alpha=1,hatch=patterns[1]), mpatches.Patch(edgecolor='black',alpha=1,hatch=patterns[2]), mpatches.Patch(edgecolor='black',alpha=1,hatch=patterns[3])]
    plt.legend(legend_elements, ['Sel: Whole pop., Train: Last gen.', 'Sel: Sol. picking, Train: Last gen.', 'Sel: Whole pop., Train: Historic', 'Sel: Sol. picking, Train: Historic', r"$\theta$: Reinit., Inds: Reinit.", r"$\theta$: Reinit., Inds: Maintain", r"$\theta$: Maintain, Inds: Reinit.", r"$\theta$: Maintain, Inds: Maintain"], loc=(0.55, 0.35))
    plt.show()


def par_coord(force=False):
    last_gen_df_path = "EvRes/total_evo.csv"
    if not os.path.isfile(last_gen_df_path) or force:
        df = pd.DataFrame(columns=["Seed", "Function", "Training method", "Model Init.", "Individual Init.", "Solution disc.", "Fitness"])
        data = np.load(root_path + "last_gen.npy")
        train_ms = ["Last Gen.", "Historic"]
        rws = [r"Retain \theta", r"Random \theta"]
        ris = ["Retain inds.", "Random inds."]
        sds = ["Sol. picking", "Whole pop."]
        for seed in range(data.shape[0]):
            for f in range(data.shape[1]):
                for rw in retain_ws:
                    for ri in retain_inds:
                        for sd in sol_disc:
                            for tm in train_m:
                                if data[seed, f, rw, ri, sd, tm] > 0:
                                    df.loc[df.shape[0]] = [seed, f, tm, rw, ri, sd, data[seed, f, rw, ri, sd, tm]]
        df.to_csv(last_gen_df_path)
    else:
        df = pd.read_csv(last_gen_df_path)

    parallel_coordinates(df[["Training method", "Model Init.", "Individual Init.", "Solution disc.", "Fitness"]], class_column="Fitness")
    plt.show()


if __name__ == "__main__":
    #load_data()
    vis_curves()
    #stackplot()
    #par_coord()