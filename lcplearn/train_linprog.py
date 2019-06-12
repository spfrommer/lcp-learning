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
    opts = parser.parse_args()

    data = dynamics.unmarshal_data(np.load(opts.datapath))
    n = data.poslambdas.shape[0]

    lambdas = np.vstack((data.poslambdas, data.neglambdas,
                         data.gammas)).transpose()
    ds = np.vstack((data.xdots, data.us,
                    np.ones(data.xdots.shape))).transpose()
    
    # R = M lambda + q
    R = np.zeros((3 * n, 18))

    for i in range(n):
        R[3*i, 0:3] = lambdas[i, :]
        R[3*i, 9:12] = ds[i, :]

        R[3*i+1, 3:6] = lambdas[i, :]
        R[3*i+1, 12:15] = ds[i, :]

        R[3*i+2, 6:9] = lambdas[i, :]
        R[3*i+2, 15:18] = ds[i, :]
    
    lambdas_vec = np.expand_dims(np.ndarray.flatten(lambdas), 0)

    #b_ub = -(0.1) * np.ones((3 * n, 1))
    b_ub = np.zeros((3 * n, 1))
    A_ub = -R
    c = np.dot(lambdas_vec, R)

    bounds = [(None, None)] * 18
    #bounds[3] = (None, 0)
    #bounds[4] = (0, None)

    sol = linprog(c, A_ub = A_ub, b_ub = b_ub, bounds = bounds, method="interior-point", options={"tol": 1e-06})
    #sol = linprog(c, A_ub = A_ub, b_ub = b_ub, options={"tol": 1e-06})

    print("Got results: " + sol.message)
    print("Objective value: " + str(sol.fun))

    print("M matrix: ")
    print(sol.x[0:9].reshape((3,3)))

    print("W matrix: ")
    print(sol.x[9:18].reshape((3,3)))

if __name__ == "__main__": main()
