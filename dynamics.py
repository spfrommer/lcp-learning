import math
import numpy as np
import lemkelcp as lcp
from collections import namedtuple

SimParams = namedtuple('params', 'g x0 xdot0 lambda0 dt time_steps')

def main():
    p = SimParams(x0=20.0,  xdot0=4.0,  lambda0=0.0,
                  g=-1.0,    dt=1.0,     time_steps=20)
    xs, xdots, lambdas = falling_box_sim(p)
    print(np.hstack((xs, xdots, lambdas)))

# Takes in parameters as a named tuple
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
            (xs[t] + xdots[t] * p.dt + p.g * (p.dt**2))])
        xsol, exit_code, _ = lcp.lemkelcp(M,q)
        assert(exit_code == 0)
        
        xs[t+1] = xsol
        if np.isclose(xs[t+1], 0):
            lambdas[t+1] = M[0,0] * xs[t+1] + q[0]
        else:
            lambdas[t+1] = 0
        xdots[t+1] = xdots[t] + (p.g + lambdas[t+1]) * p.dt

    return xs, xdots, lambdas

if __name__ == "__main__": main()
