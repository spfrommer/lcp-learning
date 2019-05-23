import sys
sys.path.append('..')

from argparse import ArgumentParser
import pdb

import sims

def main():
    parser = ArgumentParser()
    parser.add_argument('simtype', type=sims.SimType,
                                   choices=list(sims.SimType))
    opts = parser.parse_args()

    dynamics = sims.dynamics_module(opts.simtype)
    dynamics.main()

if __name__ == "__main__": main()
