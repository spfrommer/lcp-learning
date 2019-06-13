import sys
sys.path.append('..')

from argparse import ArgumentParser
import pdb

import numpy as np
from scipy.optimize import linprog

import sliding.traditional.dynamics as dynamics

def main():
    parser = ArgumentParser()
    parser.add_argument('--datapath', default='out/data.npy')
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false')
    parser.set_defaults(normalize=True)
    opts = parser.parse_args()

    data = dynamics.unmarshal_data(np.load(opts.datapath))
    n = data.poslambdas.shape[0]

    lambdas = np.vstack((data.poslambdas, data.neglambdas,
                         data.gammas)).transpose()
    ds = np.vstack((data.xdots, data.us,
                    np.ones(data.xdots.shape))).transpose()
    
    # R = M lambda + q
    if opts.normalize:
        R = np.zeros((3 * n + 18 + 18, 18 + 18))
    else:
        R = np.zeros((3 * n, 18))

    for i in range(n):
        # Including zeros (no bias in last layer) fixes weirdness
        R[3*i, 0:3] = lambdas[i, :]
        R[3*i, 9:12] = ds[i, :]
        R[3*i+1, 11] = 0

        R[3*i+1, 3:6] = lambdas[i, :]
        R[3*i+1, 12:15] = ds[i, :]
        R[3*i+1, 14] = 0

        R[3*i+2, 6:9] = lambdas[i, :]
        R[3*i+2, 15:18] = ds[i, :]


    if opts.normalize:
        R[3*n:3*n + 18, 0:18] = np.eye(18)
        R[3*n+18:3*n + 18 + 18, 0:18] = -np.eye(18)

        R[3*n:3*n + 18, 18:36] = np.eye(18)
        R[3*n+18:3*n + 18 + 18, 18:36] = np.eye(18)
    
    lambdas_vec = np.expand_dims(np.ndarray.flatten(lambdas), 0)

    #b_ub = -(0.1) * np.ones((R.shape[0], 1))
    b_ub = np.zeros((R.shape[0], 1))
    A_ub = -R

    if opts.normalize:
        c = np.dot(lambdas_vec, R[0:3*n, 0:18])
    else:
        c = np.dot(lambdas_vec, R)

    if opts.normalize:
        c = np.hstack((c, 0.001 * np.ones_like(c)))
    
    #bounds = ([(-1, 1)] * 18) + ([(None, None)] * 18)
    bounds = [(None, None)] * R.shape[1]

    sol = linprog(c, A_ub = A_ub, b_ub = b_ub, bounds = bounds, method="interior-point", options={"tol": 1e-06})
    #sol = linprog(c, A_ub = A_ub, b_ub = b_ub, bounds = bounds, method="simplex", options={"tol": 1e-06})
    #sol = linprog(c, A_ub = A_ub, b_ub = b_ub, options={"tol": 1e-06})

    print("Got results: " + sol.message)
    print("Objective value: " + str(sol.fun))

    print("M matrix: ")
    print(sol.x[0:9].reshape((3,3)))

    print("W matrix: ")
    print(sol.x[9:18].reshape((3,3)))

    if opts.normalize:
        print("Slack variables: ")
        print(sol.x[18:36].reshape((18,1)))

if __name__ == "__main__": main()
