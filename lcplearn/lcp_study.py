import sys
sys.path.append('..')

import numpy as np
from collections import namedtuple
from enum import Enum
from argparse import ArgumentParser
import pdb

import lemkelcp

import sims

class SolType(Enum):
    UNSOLVABLE      = 'unsolvable'
    SAME_XDOT       = 'same_xdot'
    SAME_LAMBDAS    = 'same_lambdas'
    DIFFERENT       = 'different'
    
    def __str__(self):
        return self.value

# If perturbing m, elem is an index tuple (i, j)
# Otherwise it's just an int
def study_pertubation(M, q, elem, perturb_M, perturb_range, xdotsolver):
    # Generate pertubations
    pertubations = np.linspace(perturb_range[0], perturb_range[1], 21)
    pertubations = pertubations + (M[elem] if perturb_M else q[elem])
    
    # Get correct, unperturbed data
    sol, exit_code, _ = lemkelcp.lemkelcp(M, q)
    assert(exit_code == 0)
    poslambda = sol[0]
    neglambda = sol[1]
    xdot = xdotsolver(poslambda, neglambda)
    
    solutions = [None] * pertubations.size

    Mprime = M.copy()
    qprime = q.copy()
    for i,p in np.ndenumerate(pertubations):
        i = i[0]
        if perturb_M:
            Mprime[elem] = p
        else:
            qprime[elem] = p
        
        sol, exit_code, _ = lemkelcp.lemkelcp(Mprime, qprime)
        if exit_code != 0:
            solutions[i] = SolType.UNSOLVABLE
        else:
            poslambdaprime = sol[0]
            neglambdaprime = sol[1]
            xdotprime = xdotsolver(poslambdaprime, neglambdaprime)

            if np.isclose(poslambda, poslambdaprime) and \
                    np.isclose(neglambda, neglambdaprime):
                solutions[i] = SolType.SAME_LAMBDAS
            elif np.isclose(xdot, xdotprime):
                solutions[i] = SolType.SAME_XDOT
            else:
                solutions[i] = SolType.DIFFERENT
    
    return solutions, pertubations

def render_solution(solutions, pertubations):
    sys.stdout.write(str(pertubations[0]) + ' ')
    for p in pertubations:
        if np.isclose(p % 0.5, 0):
            sys.stdout.write('|')
        else:
            sys.stdout.write(' ')

    sys.stdout.write(' ' + str(pertubations[-1]))

    print

    sys.stdout.write(str(pertubations[0]) + ' ')
    for s in solutions:
        if s == SolType.UNSOLVABLE:
            sys.stdout.write('U')
        elif s == SolType.SAME_LAMBDAS:
            sys.stdout.write('L')
        elif s == SolType.SAME_XDOT:
            sys.stdout.write('X')
        elif s == SolType.DIFFERENT:
            sys.stdout.write('D')
        else:
            sys.stdout.write('E')
    sys.stdout.write(' ' + str(pertubations[-1]))

    print

def study_pertubations(M, q, xdotsolver):
    perturb_range = [-1, 1]
    
    print('============== q ==============\n')
    for i,_ in np.ndenumerate(q):
        print('Perturbing q[{}]'.format(i))
        solutions, pertubations = study_pertubation(M, q, i,
                        False, perturb_range, xdotsolver)
        render_solution(solutions, pertubations)

    print('\n============== M ==============\n')
    for i,_ in np.ndenumerate(M):
        print('Perturbing M[{}]'.format(i))
        solutions, pertubations = study_pertubation(M, q, i,
                        True, perturb_range, xdotsolver)
        render_solution(solutions, pertubations)

def main():
    parser = ArgumentParser()
    parser.add_argument('--datapath', default='out/data.npy')
    parser.add_argument('--modelpath', default='out/model.pt')
    opts = parser.parse_args()

    dynamics = sims.dynamics_module(sims.SimType.SLIDING_TRADITIONAL)

    pp = dynamics.PhysicsParams(us=None, g=1.0, mu=1.0)
    xdot0 = 0
    u = 1

    def xdotsolver(poslambda, neglambda):
        _, xdot = dynamics.dynamics_step(0, xdot0, poslambda, neglambda, u)
        return xdot

    M, q = dynamics.lcp(xdot=xdot0, u=u, pp=pp)
    
    study_pertubations(M, q, xdotsolver)

if __name__ == "__main__": main()
