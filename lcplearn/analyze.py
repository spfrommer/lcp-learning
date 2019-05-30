import sys
sys.path.append('..')

from argparse import ArgumentParser
import pdb

import torch

import sims

def main():
    parser = ArgumentParser()
    parser.add_argument('--datapath', default='out/data.npy')
    parser.add_argument('--netpath', default='out/model.pt')
    parser.add_argument('--learntype', default='pytorch',
                        choices=list('pytorch', 'custom'))
    parser.add_argument('modeltype', type=sims.ModelType,
                                     choices=list(sims.ModelType))
    opts = parser.parse_args()
    
    if opts.learntype == 'pytorch':
        analyze = sims.analyze_module(opts.modeltype)
        model = sims.model_module(opts.modeltype)

        net, _, _ = model.learning_setup()
        net.load_state_dict(torch.load(opts.netpath))
        net.eval()

        analyze.analyze(net, opts)
    elif opts.learntype == 'custom':
        print('No custom support')

if __name__ == "__main__": main()
