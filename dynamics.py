import sys
import math
import numpy as np
from collections import namedtuple
import pdb

import lemkelcp as lcp

# x corresponds to height
FallingSimParams = namedtuple('falling_params', 'g x0 xdot0 lambda0 dt time_steps')
# x corresponds to sliding velocity, u external force
SlidingSimParams = namedtuple('sliding_params', 'g x0 xdot0 mu dt u time_steps')

def main():
    if sys.argv[1] == "falling":
        print('Solving falling sim...')
        p = FallingSimParams(x0=20.0,  xdot0=4.0,  lambda0=0.0,
                      g=1.0,    dt=1.0,     time_steps=30)
        xs, xdots, lambdas = falling_box_sim(p)
        print('x, xdot, lambda')
        print(np.hstack((xs, xdots, lambdas)))

    if sys.argv[1] == "sliding":
        print('Solving sliding sim...')
        p = SlidingSimParams(x0=0.0,  xdot0=1.0,  mu=0.0,
                      g=1.0,   dt=1.0,     time_steps=30,
                      u = np.ones(30) * 0)
        xs, posxdots, negxdots, poslambdas, neglambdas = sliding_box_sim(p)
        print('Full output: ')
        print('x, xdot+, xdot-, lambda+, lambda-')
        print(np.hstack((xs, posxdots, negxdots, poslambdas, neglambdas)))
        print('Summed output: ')
        print('x, xdot, lambda')
        print(np.hstack((xs, posxdots - negxdots, poslambdas - neglambdas)))

# Takes in parameters as a FallingSimParams
def falling_box_sim(p):
    xs = np.zeros((p.time_steps, 1))
    xs[0] = p.x0
    xdots = np.zeros((p.time_steps, 1))
    xdots[0] = p.xdot0
    lambdas = np.zeros((p.time_steps, 1))
    lambdas[0] = p.lambda0

    for t in range(p.time_steps-1):
        M = np.array([[(1/p.dt)**2]])
        q = np.array([-(1/(p.dt**2)) *
            (xs[t] + xdots[t] * p.dt - p.g * (p.dt**2))])
        xsol, exit_code, _ = lcp.lemkelcp(M, q)
        assert(exit_code == 0)
        
        xs[t+1] = xsol

        # Solver doesn't output slack variable
        # So we calculate it explicitely (otherwise just zero)
        if np.isclose(xsol, 0):
            lambdas[t+1] = M[0,0] * xsol + q[0]
            
        xdots[t+1] = xdots[t] + (-p.g + lambdas[t+1]) * p.dt

    return xs, xdots, lambdas

# Takes in parameters as a SlidingSimParams
def sliding_box_sim(p):
    xs = np.zeros((p.time_steps, 1))
    xs[0] = p.x0
    posxdots = np.zeros((p.time_steps, 1))
    posxdots[0] = p.xdot0 * (np.sign(p.xdot0) > 0)
    negxdots = np.zeros((p.time_steps, 1))
    negxdots[0] = (-p.xdot0) * (np.sign(p.xdot0) < 0)
    # Lambda corresponds to unused friction force, see notes
    # First time step all friction force is unused
    poslambdas = np.zeros((p.time_steps, 1))
    poslambdas[0] = p.mu * p.g
    neglambdas = np.zeros((p.time_steps, 1))
    neglambdas[0] = p.mu * p.g

    for t in range(p.time_steps-1):
        M = (1/p.dt) * np.array([[0,1],[1,0]])
        q = np.array([(1/p.dt) * posxdots[t] - (1/p.dt) * negxdots[t] + p.u[t+1] + p.mu * p.g,
                      (1/p.dt) * negxdots[t] - (1/p.dt) * posxdots[t] - p.u[t+1] + p.mu * p.g])
        xdotsol, exit_code, _ = lcp.lemkelcp(M, q)
        assert(exit_code == 0)
        
        posxdots[t+1] = xdotsol[0]
        negxdots[t+1] = xdotsol[1]
        xs[t+1] = xs[t] + p.dt * (posxdots[t+1] - negxdots[t+1])

        # Solver doesn't output slack variable
        # So we calculate it explicitely
        if np.isclose(xdotsol[0], 0):
            poslambdas[t+1] = np.dot(M[0,:], xdotsol) + q[0]
        if np.isclose(xdotsol[1], 0):
            neglambdas[t+1] = np.dot(M[1,:], xdotsol) + q[1]

    return xs, posxdots, negxdots, poslambdas, neglambdas

if __name__ == "__main__": main()
