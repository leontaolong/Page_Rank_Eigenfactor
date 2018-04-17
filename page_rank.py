##########################  Library Imports  #######################

import numpy as np
import pandas as pd
import math
import time

############################  Constants  ###########################

ALPHA_CONST = 0.85
EPSILON_CONST = 0.00001
NUM_OF_NODES = 10748
FILE_NAME = 'links.txt'

############################  Functions  ###########################


def create_matrix_and_journal_keys():
    m = np.zeros((NUM_OF_NODES, NUM_OF_NODES))
    journal_keys = []
    with open(FILE_NAME) as f:
        for _, line in enumerate(f):
            line = line.split(',')
            m[int(line[1])][int(line[0])] = int(line[2])
            journal_keys.append(line[1])

    return m, journal_keys


def normalize_matrix(matrix):
    is_dangling_node = []
    for idx, row in enumerate(matrix.T):
        sum = np.sum(row)
        if sum != 0:
            matrix.T[idx] = row / sum
            is_dangling_node.append(0)
        else:
            is_dangling_node.append(1)
    return matrix, is_dangling_node


def calc_pi(pi_vec, artical_vec, d, H):
    # compute pi^(k+1)
    vec = np.multiply(ALPHA_CONST, H)
    vec = np.dot(vec, pi_vec)

    alpha_d = np.multiply(ALPHA_CONST, d)
    dot_pi = np.dot(alpha_d, pi_vec)
    dot_pi = dot_pi + (1 - ALPHA_CONST)
    dot_a = np.multiply(dot_pi, artical_vec)

    # get the new pi-vector
    return vec + dot_a


def converged(res):
    not_less_than = list(filter(lambda x: x >= EPSILON_CONST, res))
    return len(not_less_than) == 0


def iterate(pi_vec, artical_vec, d, H):
    # initial residuals
    res = [1] * NUM_OF_NODES
    count = 0
    while not converged(res):
        pi_new = calc_pi(pi_vec, artical_vec, d, H)
        res = pi_new - pi_vec
        pi_vec = pi_new
        count += 1
    return pi_vec, count


def main():
    # benchmarking timer start
    start = time.time()
    print("Start running...")

    ##############################   Data Input  ##############################

    Z_matricx, journal_keys = create_matrix_and_journal_keys()

    np.fill_diagonal(Z_matricx, 0)
    H_matricx, d = normalize_matrix(Z_matricx)
    A_tot = np.array([1] * NUM_OF_NODES)
    artical_vec = A_tot / NUM_OF_NODES
    pi_vec = np.array([1 / NUM_OF_NODES] * NUM_OF_NODES)

    ##############################   Iteration   ##############################

    # reshape them to be 6x1 vectors
    pi_vec = np.reshape(pi_vec, (NUM_OF_NODES, 1))
    artical_vec = np.reshape(artical_vec, (NUM_OF_NODES, 1))

    pi_vec, count = iterate(pi_vec, artical_vec, d, H_matricx)

    eiganfactor = (100 * (np.dot(H_matricx, pi_vec) /
                          np.sum(np.dot(H_matricx, pi_vec))))

    ############################  Result Reporting  ###########################

    df = pd.DataFrame(
        {'Journal': journal_keys[:len(eiganfactor)], 'Eiganfactor': eiganfactor.T[0]})
    df = df.sort_values(
        by=['Eiganfactor'], ascending=False)
    df.index.name = 'Journal'

    print(df['Eiganfactor'].head(20))
    # print(eiganfactor)
    # print(journal_keys)
    print("Number of iterations: " + str(count))

    # benchmarking timer end
    end = time.time()
    print("Total time taken: " + str(end - start) + " Secs")


if __name__ == "__main__":
    main()
