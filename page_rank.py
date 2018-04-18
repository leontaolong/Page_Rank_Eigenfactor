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


# import and process data from file,
# make and return H_matrix and an array of all journals
def create_matrix_and_journal_keys():
    # initial matrix with NUM_OF_NODES nodes of 0
    matrix = np.zeros((NUM_OF_NODES, NUM_OF_NODES))
    journal_keys = []
    with open(FILE_NAME) as f:
        for _, line in enumerate(f):
            line = line.split(',')  # each line in file
            citing = int(line[0])
            cited = int(line[1])
            val = int(line[2])
            matrix[cited][citing] = val  # populate matrix,
            # with columns being the cited and rows being citing
            journal_keys.append(line[1])  # add journal to journal_keys array
    return matrix, journal_keys


# return normalized matrix and an array indicating dangling nodes
def normalize_matrix_and_get_dangling(matrix):
    is_dangling_node = []
    # transpose matrix to normalize by column
    for idx, row in enumerate(matrix.T):
        sum = np.sum(row)  # get column sum
        if sum != 0:
            # if column sum is not 0, normalize column by dividing number by row sum, and mark it as 0
            matrix.T[idx] = row / sum
            is_dangling_node.append(0)
        else:  # if column sum is 0, it means that this column is a dangling node, mark it as 1
            is_dangling_node.append(1)
    return matrix, is_dangling_node


# calculate pi ^ (k + 1) =
#   ALPHA_CONST * H * pi ^ (k)
#   + [
#        ALPHA_CONST * d * pi ^ (k) +
#        ( 1 - ALPHA_CONST )
#     ] * artical_vec
def calc_pi(pi_vec, artical_vec, d, H):
    # compute ALPHA_CONST * H * pi ^ (k)
    vec = np.multiply(ALPHA_CONST, H)
    vec = np.dot(vec, pi_vec)

    # compute ALPHA_CONST * d * pi ^ (k) + ( 1 - ALPHA_CONST )
    alpha_d = np.multiply(ALPHA_CONST, d)
    dot_pi = np.dot(alpha_d, pi_vec)
    dot_pi = dot_pi + (1 - ALPHA_CONST)

    # multiply by artical_vec
    dot_a = np.multiply(dot_pi, artical_vec)

    # get the new pi-vector by adding two parts together
    return vec + dot_a


# check if it's converged:
# T = pi ^ (k+1) - pi ^ (k) < EPSILON_CONST
def converged(residual):
    # check every node, find residual >= EPSILON_CONST
    not_less_than = list(filter(lambda x: x >= EPSILON_CONST, residual))
    # if there is any node where residual >= EPSILON_CONST, it means it's not converged yet,
    # otherwise it will be empty and return true
    return len(not_less_than) == 0


# iterate equation, return the final pi_vecter and number of iterations
def iterate(pi_vec, artical_vec, d, H):
    # initial residual
    res = [1] * NUM_OF_NODES
    count = 0
    while not converged(res):
        # calculate new pi_vector
        pi_new = calc_pi(pi_vec, artical_vec, d, H)
        # calculate new residual
        res = pi_new - pi_vec
        # let calculated pi_vecter become current, for the next iteration
        pi_vec = pi_new
        # increment iteration counter
        count += 1
    return pi_vec, count


def main():
    # benchmarking timer start
    start = time.time()
    print("Start running...")

    ##############################   Data Input  ##############################

    # get initial matrix Z and array of all journals
    Z_matricx, journal_keys = create_matrix_and_journal_keys()
    # zero diagnals so it doesn't count self-citing
    np.fill_diagonal(Z_matricx, 0)
    # get normalized matrix H and dangling node indicator array d
    H_matricx, d = normalize_matrix_and_get_dangling(Z_matricx)
    # Total number of articles, in this case we can assume that be all 1s
    A_tot = np.array([1] * NUM_OF_NODES)
    # get article vecter
    artical_vec = A_tot / NUM_OF_NODES
    # initial pi vecter: every node is 1 / NUM_OF_NODES
    pi_vec = np.array([1 / NUM_OF_NODES] * NUM_OF_NODES)

    ##############################   Iteration   ##############################

    # reshape pi_vecter and article_vecter to be 6x1 vectors
    pi_vec = np.reshape(pi_vec, (NUM_OF_NODES, 1))
    artical_vec = np.reshape(artical_vec, (NUM_OF_NODES, 1))

    # start iterating, get final pi_vecter and iteration counts
    pi_vec, count = iterate(pi_vec, artical_vec, d, H_matricx)

    # calculating eiganfactor
    eiganfactor = (100 * (np.dot(H_matricx, pi_vec) /
                          np.sum(np.dot(H_matricx, pi_vec))))

    ############################  Result Reporting  ###########################

    # create dataframe with Journal numbers and Eiganfactor associated with it
    df = pd.DataFrame(
        {'Journal': journal_keys[:len(eiganfactor)], 'Eiganfactor': eiganfactor.T[0]})
    # sort df by Eiganfactor
    df = df.sort_values(
        by=['Eiganfactor'], ascending=False)
    # rename index
    df.index.name = 'Journal'

    # report result
    print('Top 20 journals with greatest eiganfactor: ')
    print(df['Eiganfactor'].head(20))
    print("Number of iterations: {0}".format(count))

    # benchmarking timer end
    end = time.time()
    print("Total time taken: {0:.4f} Secs".format(end - start))


if __name__ == "__main__":
    main()

############################  Sample Results  ###########################

# Start running...
# Top 20 journals with greatest eiganfactor:
# Journal
# 4408    1.446274
# 4801    1.410543
# 6610    1.233689
# 2056    0.678919
# 6919    0.664298
# 6667    0.633399
# 4024    0.576042
# 6523    0.480174
# 8930    0.477196
# 6857    0.439395
# 5966    0.429420
# 1995    0.385487
# 1935    0.384908
# 3480    0.379425
# 4598    0.372269
# 2880    0.329962
# 3314    0.326856
# 6569    0.319103
# 5035    0.316170
# 1212    0.311132
# Name: Eiganfactor, dtype: float64
# Number of iterations: 17
# Total time taken: 24.0142 Secs
