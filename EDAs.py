from deap import tools
import foolbox as fb
import numpy as np
import tensorflow.keras as keras
from keras.utils import to_categorical
from netTraining import train, initialize_variables, back_drive_net
from structure import initialize_back_variables, close
import seaborn as sns
import matplotlib.pyplot as plt


class EDA(object):
    def __init__(self, pop_sz, ind_size, evl, method, offspring, **kwargs):
        self.pop_size = pop_sz
        self.ind_size = ind_size
        self.eval = evl
        self.high_scale = None
        self.low_scale = None
        self.best = None
        self.low_scale = None
        self.high_scale = None
        self.sel_method = method
        self.offspring = offspring
        self.args = kwargs

    def initial_population(self, ind_init):
        pop = create_population(self.ind_size, self.pop_size)

        ind_list = list(map(ind_init, pop))
        fitnesses = list(map(self.eval, pop))

        for ind, fit in zip(ind_list, fitnesses):
            ind.fitness.values = (fit,)

        return ind_list

    def best_inds(self, population, method):
        if method == 0:
            best = tools.selBest(population, k=int(self.pop_size*(1-self.offspring)))  # Get the best half of individuals
        else:
            best = tools.selTournament(population, k=int(self.pop_size*(1-self.offspring)), tournsize=self.args["tour_size"])

        return best

    def new_population(self, new_pop, ind_init):

        ind_list = list(map(ind_init, new_pop))  # Transform the population to DEAP format (with fitness, ...)

        # Compute and add the fitness values to the individuals

        fitnesses = list(map(self.eval, new_pop))
        for ind, fit in zip(ind_list, fitnesses):
            ind.fitness.values = (fit,)
        return ind_list


def create_individual(ind_size):
    """
    :param ind_size: Size of the solution
    :return:
    """

    ind = []
    for i in range(ind_size):
        ind.append(np.random.rand())

    return ind


def create_population(ind_size, pop_sz, index=False):
    """
    :param ind_size: Amount of variables in the function
    :param pop_sz: Amount of solutions in the population
    :param index: Booleans, whether the solutions will correspond to the first integers (True) or not
    :return: the population according to the mentioned criteria
    """
    population = []
    for fixed_ind in range(pop_sz):
        if index:
            ind = create_individual(ind_size)
        else:
            ind = create_individual(ind_size)
        population.append(ind)
    population = np.array(population)

    return population


class UMDA(EDA):

    def __init__(self, eval_function, pop_sz, num_variables, method, offspring, **kwargs):
        super().__init__(pop_sz, num_variables, eval_function, method, offspring, **kwargs)
        self.mu = None
        self.sigma = None
        self.best = None

    # This function generates individuals
    def generate(self, ind_init):

        if self.best is None:
            ind_list = self.initial_population(ind_init)
        else:
            pop = np.random.normal(self.mu, self.sigma, (int(self.pop_size*self.offspring), self.ind_size))

            pop = np.clip(pop, 0, 1)
            ind_list = self.new_population(pop, ind_init) + self.best  # The best individuals in the previous generation and the new ones

        return ind_list

    def update(self, population):

        self.best = self.best_inds(population, self.sel_method, )
        self.mu = np.mean(self.best, axis=0)
        self.sigma = np.sqrt(np.var(self.best, axis=0))


class AdvEDA(EDA):

    def __init__(self, eval_function, pop_sz, num_variables, method, offspring, **kwargs):
        super().__init__(pop_sz, num_variables, eval_function, method, offspring, **kwargs)

        self.model = self.initialize_nn(num_variables, num_variables*3, num_variables*2)
        self.best = None
        self.worst = None
        self.foolbox_model = None
        self.retain = []
        self.retain_f = []
    # This function generates individuals
    def generate(self, ind_init):

        if self.best is None:
            ind_list = self.initial_population(ind_init)
        else:
            new_pop = []
            for bad in self.worst:
                pred = np.argmax(self.model.predict(np.array([bad])), axis=1)[0]
                new_pop += [self.generate_adv_example(self.foolbox_model, bad, pred)]

            ind_list = self.new_population(np.array(new_pop), ind_init)

        return ind_list

    @staticmethod
    def generate_adv_example(model, sample, sample_label, target_label=None):

        if target_label is None:
            attack = fb.v1.attacks.FGSM(model)
        else:
            criterion = fb.criteria.TargetClass(target_label)
            attack = fb.v1.attacks.FGSM(model, criterion)

        return attack(sample, sample_label)

    def update(self, population):
        X_train, y_train = self.select_best_worst(population, self.sel_method)
        self.model.fit(X_train, y_train, batch_size=100, epochs=250, verbose=0, callbacks=[], validation_data=None, shuffle=True, class_weight=None, sample_weight=None)
        self.foolbox_model = fb.models.TensorFlowModel.from_keras(self.model, bounds=(0, 1))

    def select_best_worst(self, population, method):

        fitness = np.array([-x.fitness.wvalues[0] for x in population])
        individuals = np.array([np.array(population[i]) for i in range(len(population))])

        best_sols = np.argsort(fitness)

        # We select the Trunc percentage of best X
        self.best = individuals[best_sols[:self.args["sel_pop_size"]], :]
        self.retain = individuals[:best_sols[-self.args["sel_pop_size"]], :]
        self.retain_f = fitness[:best_sols[-self.args["sel_pop_size"]]]
        # We select the Trunc percentage of worst X
        self.worst = individuals[best_sols[-self.args["sel_pop_size"]:], :]

        X_train = np.vstack(([self.best, self.worst]))
        y_train = np.hstack((np.ones(self.args["sel_pop_size"]), np.zeros(self.args["sel_pop_size"])))
        y_train = to_categorical(y_train)

        return X_train, y_train

    @staticmethod
    def initialize_nn(x_dim, z_dim_1, z_dim_2):
        # initialize model
        model = keras.models.Sequential()
        # add input layer
        model.add(keras.layers.Dense(units=z_dim_1, input_shape=(x_dim,), kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='sigmoid'))
        # add hidden layer
        model.add(keras.layers.Dense(units=z_dim_2, input_dim=z_dim_1, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='sigmoid'))
        # add output layer
        model.add(keras.layers.Dense(units=2, input_dim=z_dim_2, kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='softmax'))
        # define SGD optimizer
        sgd_optimizer = keras.optimizers.Adam(lr=0.001)
        # compile model
        model.compile(optimizer=sgd_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        return model


class BackDriveEDA(EDA):

    def __init__(self, eval_function, pop_sz, num_variables, outputs, method, offspring, retain_ws, retain_inds, sol_disc, train_m, **kwargs):
        super().__init__(pop_sz, num_variables, eval_function, method, offspring, **kwargs)
        self.sess, self.x, self.y, self.weights, self.hidden_weights, self.biases, self.hidden_biases, self.net, self.loss, self.optimizer, _, _, _, _ = initialize_variables(input_size=num_variables, output_size=outputs, h_neurons=self.args["h_neurons"] if "h_neurons" in self.args else 50, batch_size=self.args["batch_size"] if "batch_size" in self.args else 50, learning_rate=self.args["lr"] if "lr" in self.args else 0.001, sigma=self.args["sigma"] if "sigma" in self.args else 15, variable_groups=None, structure=0, delete=True)
        self.individual, self.ind_initializer, self.clipped_ind, self.target, self.back_weights, self.back_hidden_weights, self.back_biases, self.back_hidden_biases, self.back_net, self.perf_loss, self.apply_grads, self.grads, _, self.sum_back, _, _, _ = initialize_back_variables(sess=self.sess, input_size=num_variables, output_size=outputs, h_neurons=self.args["h_neurons"] if "h_neurons" in self.args else 50, sigma=self.args["sigma"] if "sigma" in self.args else 15, variable_groups=None, structure=0, number=pop_sz*(1 if sol_disc == 0 else 20))
        self.best = None
        self.retain_ws = retain_ws
        self.retain_inds = retain_inds
        self.sol_disc = sol_disc
        self.rand_pop = None
        self.rand_fit = None
        self.elite_pop = None
        self.elite_fit = None
        self.historic_pop = None
        self.historic_fit = None
        self.train_m = train_m

    # This function generates individuals
    def generate(self, ind_init):

        if self.best is None:
            ind_list = self.initial_population(ind_init)
            self.rand_fit = np.array([-x.fitness.wvalues[0] for x in ind_list])
            self.rand_pop = np.array(ind_list)
            self.elite_pop = self.rand_pop
            self.elite_fit = self.rand_fit
            self.historic_pop = self.rand_pop
            self.historic_fit = self.rand_fit
        else:
            w, h_w, b, h_b = self.sess.run([self.weights, self.hidden_weights, self.biases, self.hidden_biases])
            if self.retain_inds == 0:
                self.sess.run(self.ind_initializer)
            back_drive_net(self.sess, self.args["convergence"] if "convergence" in self.args else 0.01, self.best, self.target, self.apply_grads, self.perf_loss, self.back_weights, self.back_hidden_weights, self.back_biases, self.back_hidden_biases, w, h_w, b, h_b, 500, self.clipped_ind, self.back_net)
            ind = np.clip(self.sess.run(self.individual), 0, 1)
            if ind.shape[0] > self.pop_size:
                preds = self.sess.run(self.net, feed_dict={self.x: ind})[:, 0]
                ind = ind[preds <= np.nanquantile(preds, self.pop_size/ind.shape[0])]
                if ind.shape[0] == 0 or ind.shape[0] > 5*self.pop_size:
                    ind = self.historic_pop[self.pop_size*2:self.pop_size*3]
            ind_list = self.new_population(ind, ind_init)
        return ind_list

    def update(self, population, percentile, noise):
        fitness = np.array([-x.fitness.wvalues[0] for x in population])
        individuals = np.array([np.array(population[i]) for i in range(len(population))])

        self.historic_pop = np.concatenate((individuals, self.historic_pop[:(10000-individuals.shape[0])]))
        self.historic_fit = np.concatenate((fitness, self.historic_fit[:(10000 - fitness.shape[0])]))

        self.elite_fit = np.concatenate((self.elite_fit, fitness))
        self.elite_pop = np.concatenate((self.elite_pop, individuals))

        indices = self.elite_fit <= np.nanquantile(self.elite_fit, self.pop_size/self.elite_pop.shape[0])

        self.elite_pop = self.elite_pop[indices]
        self.elite_fit = self.elite_fit[indices]
        if self.train_m == 1:
            individuals = np.concatenate((individuals, self.rand_pop, self.elite_pop, self.historic_pop))
            fitness = np.concatenate((fitness, self.rand_fit, self.elite_fit, self.historic_fit))

        fitness = fitness - np.nanmin(fitness)
        fitness = fitness + 0.01*np.nanmax(fitness)
        fitness = fitness / np.nanmax(fitness)

        outs = np.array([np.log(fitness), fitness, np.sqrt(fitness), np.power(fitness, 2), np.sin(fitness)])[:self.net.shape[1]].T

        outs = outs - np.nanmin(outs, axis=0)
        outs = outs + 0.01 * np.nanmax(outs, axis=0)
        outs = outs / np.nanmax(outs, axis=0)
        self.best = np.nanpercentile(outs, percentile, axis=0).reshape(-1, self.net.shape[1])

        if noise == 1:
            self.best = np.array([np.random.normal(loc=var, scale=0.01, size=self.pop_size) for var in self.best[0]]).T

        train(net_input=individuals, net_output=outs, start=False, sess=self.sess, x=self.x, y=self.y, net=self.net, loss=self.loss, optimizer=self.optimizer, sum_ford=None, saver=None, writer=None, log=None, epoch=self.args["epochs"] if "epochs" in self.args else 1, batch_size=self.args["batch_size"] if "batch_size" in self.args else 50, display_step=100000, save_step=10000, groups=None, structure=0, init=self.retain_ws == 0)

    def close(self):
        close(self.sess)
