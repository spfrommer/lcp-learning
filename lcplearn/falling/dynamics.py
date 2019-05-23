import numpy as np
from collections import namedtuple
import pdb

import lemkelcp

# x corresponds to height
PhysicsParams = namedtuple('falling_physics_params', 'g dt')
SimParams = namedtuple('falling_sim_params', 'x0 xdot0 lambda0 time_steps')
SimSolution = namedtuple('falling_solution', 'xs, xdots, lambdas')
SimData = namedtuple('falling_data', 'xs, xdots, lambdas')

MARSHALLED_SIZE = 3
HAS_PROCESSING = False

def lcp(x, xdot, pp):
    M = np.array([[(1/pp.dt)**2]])
    q = np.array([-(1/(pp.dt**2)) *
        (x + xdot * pp.dt - pp.g * (pp.dt**2))])
    return M,q

def slack_calc(xsol, M, q):
    # Solver doesn't output slack variable
    # So we calculate it explicitely (otherwise just zero)
    x_zero = np.isclose(xsol, 0) 
    return M[0,0] * xsol + q[0] if x_zero else 0

def dynamics_step(xdot, slack_lambda, pp):
    return xdot + (-pp.g + slack_lambda) * pp.dt

def sim(pp, sp):
    xs = np.zeros((sp.time_steps, 1))
    xs[0] = sp.x0
    xdots = np.zeros((sp.time_steps, 1))
    xdots[0] = sp.xdot0
    lambdas = np.zeros((sp.time_steps, 1))
    lambdas[0] = sp.lambda0

    for t in range(sp.time_steps-1):
        M, q = lcp(xs[t], xdots[t], pp)
        xsol, exit_code, _ = lemkelcp.lemkelcp(M, q)
        assert(exit_code == 0)
        
        xs[t+1] = xsol
        lambdas[t+1] = slack_calc(xsol, M, q)
        xdots[t+1] = dynamics_step(xdots[t], lambdas[t+1], pp)
    
    return SimSolution(xs, xdots, lambdas)

# Lines up xs/xdots from one time step to lambdas of the next
# And concatenates them into a matrix
# Goes from n entries -> n - 1
# Used to prep data for input into ML
def marshal_data(ss):
    return np.hstack((ss.xs[:-1], ss.xdots[:-1], ss.lambdas[1:]))

# numpy array -> sd
def unmarshal_data(data):
    return SimData(data[:, 0], data[:, 1], data[:, 2])

def main():
    print('Solving falling sim...')
    pp = FallingPhysicsParams  (g=1.0,         dt=1.0)
    sp = FallingSimParams      (x0=20.0,       xdot0=4.0,
                                lambda0=0.0,   time_steps=30)
    sol = sim(pp, sp)
    print('x, xdot, lambda')
    print(np.hstack((sol.xs, sol.xdots, sol.lambdas)))

if __name__ == "__main__": main()
