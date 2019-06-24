import sys
sys.path.append('..')
sys.path.append('../..')

from argparse import ArgumentParser
import pdb

from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
import numpy as np

import sliding.traditional.dynamics as dynamics

def main():
    parser = ArgumentParser()
    parser.add_argument('--datapath', default='../out/data.npy')
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false')
    parser.set_defaults(normalize=True)
    opts = parser.parse_args()

    data = dynamics.unmarshal_data(np.load(opts.datapath))
    n = data.poslambdas.shape[0]

    lambdas = np.vstack((data.poslambdas, data.neglambdas,
                         data.gammas))
    ds = np.vstack((data.xdots, data.us,
                    np.ones(data.xdots.shape)))

    mp = MathematicalProgram()
    M = mp.NewContinuousVariables(3, 3, "M")
    W = mp.NewContinuousVariables(3, 3, "W")
    W[0, 2] = 0
    W[1, 2] = 0
    
    R = M.dot(lambdas) + W.dot(ds)
    elementwise_positive_constraint(mp, R) 

    lambdas_vec = np.expand_dims(np.ndarray.flatten(lambdas), 0)
    R_vec = np.ndarray.flatten(R)
    mp.AddLinearCost(lambdas_vec.dot(R_vec)[0])

    result = Solve(mp)
    print(result.is_success())

    Msol = result.GetSolution(M)
    
    # When mix 0's in with W, need to evaluate expression
    evaluator = np.vectorize(lambda x: x.Evaluate())
    Wsol = evaluator(result.GetSolution(W))
    print(Msol)
    print(Wsol)

def elementwise_positive_constraint(mp, array):
    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            mp.AddLinearConstraint(array[row, col] >= 0)

if __name__ == "__main__": main()
