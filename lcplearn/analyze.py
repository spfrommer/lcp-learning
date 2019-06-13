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
    parser.add_argument('--learntype', default='model',
                        choices=['model', 'data'])
    parser.add_argument('analyzetype', type=sims.AnalyzeType,
                                     choices=list(sims.AnalyzeType))
    opts = parser.parse_args()
    
    if opts.learntype == 'model':
        analyze = sims.analyze_module(opts.analyzetype)
        opts.modeltype = sims.ModelType(str(opts.analyzetype)) 
        model = sims.model_module(opts.modeltype)

        model = sims.model_module(opts.modeltype)

        net, _, _, _ = model.learning_setup()
        net.load_state_dict(torch.load(opts.netpath))
        net.eval()

        analyze.analyze(net, opts)
    elif opts.learntype == 'data':
        analyze = sims.analyze_module(opts.analyzetype)
        analyze.analyze(opts)

if __name__ == "__main__": main()
