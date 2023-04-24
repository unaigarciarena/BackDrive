import numpy as np


def kullback_leibler(prob1, prob2):
    return np.sum(prob2*np.log(prob1/prob2))


# Given two lists, returns a list with all the possible combinations

def combinations(list1, list2):
    result = []
    for l1 in list1:
        for l2 in list2:
            result += [[l1, l2]]
    return result


# Given a list of fitness values of solutions and a temperature, this function returns the
# boltzmann probability for each solution.

def boltzmann(fits, t):
    """
    :param fits: Fitness values
    :param t: t parameter
    :return: The boltzmann distribution probability
    """
    nums = []
    for fit in fits:
        nums += [np.exp(fit/t)]
    nums = np.array(nums)
    reg = np.sum(nums)
    nums = nums/reg
    return nums


def deceptive(ind, *args):  # Evaluation function
    size = args[0]["size"]
    val = 0
    for xi in range(0, len(ind), size):
        val += sub_deceptive(ind[xi:xi+size])
    return val


def sub_deceptive(subind):  # Evaluation subfunction
    sum_round = sum(np.round(subind, 0))
    if sum_round == len(subind):
        return len(subind)
    else:
        return len(subind) - sum_round - 1


def one_max(ind, _):  # Evaluation function
    return sum(np.round(ind, 0))


def trap(ind, *args):  # Multiplier is to level up the values of both functions. Can be 1 for no effect.
    size = args[0]["size"]
    multiplier = args[0]["multiplier"]
    val = 0
    for xi in range(0, len(ind), size):
        val += sub_trap(ind[xi:xi+size])
    return val*multiplier


def sub_trap(subind):
    sum_round = sum(np.round(subind, 0))
    if sum_round == len(subind):
        return 1
    if sum_round == len(subind)-1:
        return 0
    return 1-0.1*(sum_round+1)


def fitness_function(variables, functions, solution, args):

    if not len(variables) == len(functions) or max(max(variables))+1 > solution.shape[0]:
        print("perror1")
    total = 0
    for i in range(len(variables)):
        total += functions[i](solution[variables[i]], args[i])
    return total


if __name__ == '__main__':
    # a = fitness_function([[0, 1, 2], [0, 1, 2]], [deceptive, one_max], np.array([1, 1, 1]), [{"size": 3}, None])
    # a = permutations([1,2,3,4,5], [6,7,8,9,10])
    a = fitness_function([[0, 1, 2]], [trap], np.array([0, 0, 0]), [{"size": 3, "multiplier": 1}])
    print(a)
