import sys
sys.path.append('..')

import numpy as np
from collections import namedtuple
from enum import Enum
from argparse import ArgumentParser
import pdb

import lemkelcp

import sims

soltype_lookup = ('U', 'X', 'L', 'D')

class SolType(Enum):
    UNSOLVABLE      = 'U'
    SAME_XDOT       = 'X'
    SAME_LAMBDAS    = 'L'
    DIFFERENT       = 'D'
    
    def __str__(self):
        return self.value

def main():
    parser = ArgumentParser()
    parser.add_argument('studytype', choices=['sweep', 'noise'])
    opts = parser.parse_args()

    dynamics = sims.dynamics_module(sims.SimType.SLIDING_TRADITIONAL)

    pp = dynamics.PhysicsParams(us=None, g=1.0, mu=1.0)
    xdot0 = 0
    u = 1

    def xdotsolver(poslambda, neglambda):
        _, xdot = dynamics.dynamics_step(0, xdot0, poslambda, neglambda, u)
        return xdot

    M, q = dynamics.lcp(xdot=xdot0, u=u, pp=pp)
    
    if opts.studytype == 'sweep':
        element_sweep_study(M, q, xdotsolver)
    elif opts.studytype == 'noise':
        matrix_noise_study(M, q, xdotsolver)

# =============== ELEMENT SWEEPS =============== 

class Selector(Enum):
    M = 0
    Q = 1
    BOTH = 2

def element_sweep_study(M, q, xdotsolver):
    perturb_range = [-1, 1]
    print('============== q ==============\n')
    for i,_ in np.ndenumerate(q):
        print('Perturbing q[{}]'.format(i))
        solutions, pertubations = element_sweep(M, q, i,
                        Selector.Q, perturb_range, xdotsolver)
        render_element_solution(solutions, pertubations)

    print('\n============== M ==============\n')
    for i,_ in np.ndenumerate(M):
        print('Perturbing M[{}]'.format(i))
        solutions, pertubations = element_sweep(M, q, i,
                        Selector.M, perturb_range, xdotsolver)
        render_element_solution(solutions, pertubations)

# If perturbing m, elem is an index tuple (i, j)
# Otherwise it's just an int
def element_sweep(M, q, elem, selector, perturb_range, xdotsolver):
    # Generate pertubations
    pertubations = np.linspace(perturb_range[0], perturb_range[1], 21)
    pertubations = pertubations + (M[elem] if selector == Selector.M
                                           else q[elem])
    
    # Get correct, unperturbed data
    sol, exit_code, _ = lemkelcp.lemkelcp(M, q)
    assert(exit_code == 0)
    
    solutions = [None] * pertubations.size

    Mprime = M.copy()
    qprime = q.copy()
    for i,p in np.ndenumerate(pertubations):
        i = i[0]
        if selector == Selector.M:
            Mprime[elem] = p
        else:
            qprime[elem] = p
        
        solutions[i], _ = compare_solution(Mprime, qprime,
                                        xdotsolver, sol)
    
    return solutions, pertubations

def render_element_solution(solutions, pertubations):
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
        sys.stdout.write(str(s))

    sys.stdout.write(' ' + str(pertubations[-1]))

    print

# =============== MATRIX NOISE =============== 

def matrix_noise_study(M, q, xdotsolver):
    stds = np.linspace(0, 1, 21)
    print("Testing gaussian stds:")
    print(stds)

    soltype_results_M, noise_results_M = \
        compile_noise_results(M, q, xdotsolver, stds, Selector.M)
    print("=============== Noise on M ===============")
    print_results(soltype_results_M, noise_results_M)

    print("=============== Noise on q ===============")
    soltype_results_q, noise_results_q = \
        compile_noise_results(M, q, xdotsolver, stds, Selector.Q)
    print_results(soltype_results_q, noise_results_q)

    print("=============== Noise on both ===============")
    soltype_results_both, noise_results_both = \
        compile_noise_results(M, q, xdotsolver, stds, Selector.BOTH)
    print_results(soltype_results_both, noise_results_both)

def print_results(soltype_results, noise_results):
    print("===== Solution proportions =====")
    print("Unsolvable, same xdot, same lambda, different xdot")
    print(soltype_results)
    print("===== Mean xdot errors =====")
    print(noise_results)

def compile_noise_results(M, q, xdotsolver, stds, selector):
    # Each row has solution counts for each type of solution type
    soltype_results = np.zeros((stds.size, 4))
    noise_results = np.zeros((stds.size, 2))

    for i, std in np.ndenumerate(stds):
        i = i[0]
        for _ in range(1000):
            sol, xdoterr = matrix_noise(M, q, xdotsolver, std, selector)
            soltype_results[i, soltype_lookup.index(str(sol))] += 1
            if xdoterr != None:
                noise_results[i, 0] += np.absolute(xdoterr)
                noise_results[i, 1] += 1
    
    soltype_proportions = soltype_results / 1000.0
    noise_maes = noise_results[:, 0] / noise_results[:, 1] 

    return soltype_proportions, noise_maes[:, None]

def matrix_noise(M, q, xdotsolver, std, selector):
    # Get correct, unperturbed data
    sol, exit_code, _ = lemkelcp.lemkelcp(M, q)
    assert(exit_code == 0)
    
    Mprime = M
    qprime = q
    if selector in (Selector.M, Selector.BOTH):
        Mprime = M + np.random.normal(0, std, M.shape)
    if selector in (Selector.Q, Selector.BOTH):
        qprime = q + np.random.normal(0, std, q.shape)
    
    #if std > 0 and selector == Selector.Q:
    #    pdb.set_trace()

    return compare_solution(Mprime, qprime, xdotsolver, sol)

def compare_solution(Mprime, qprime, xdotsolver, sol):
    poslambda = sol[0]
    neglambda = sol[1]
    xdot = xdotsolver(poslambda, neglambda)

    soltype = None
    xdoterr = None
    sol, exit_code, _ = lemkelcp.lemkelcp(Mprime, qprime)

    if exit_code != 0:
        soltype = SolType.UNSOLVABLE
    else:
        poslambdaprime = sol[0]
        neglambdaprime = sol[1]
        xdotprime = xdotsolver(poslambdaprime, neglambdaprime)
        xdoterr = xdotprime - xdot

        if np.isclose(poslambda, poslambdaprime) and \
                np.isclose(neglambda, neglambdaprime):
            soltype = SolType.SAME_LAMBDAS
        elif np.isclose(xdot, xdotprime):
            soltype = SolType.SAME_XDOT
        else:
            soltype = SolType.DIFFERENT
    
    return soltype, xdoterr

if __name__ == "__main__": main()
