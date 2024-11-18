import argparse
import sys
import os
import torch
import time
from experiment.tune_and_exp import tune_and_experiment_multiple_runs, tune_and_experiment_multiple_runs_foracil
from utils.utils import Logger,boolean_string
from types import SimpleNamespace
from experiment.tune_config import config_default


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Hyper-parameter tuning')
    parser.add_argument('--data', dest='data', default='uwave', type=str,
                        choices=['har', 'uwave', 'dailysports', 'grabmyo', 'wisdm',
                                 'ninapro', 'sines'])
    parser.add_argument('--encoder', dest='encoder', default='CNN', type=str)
    parser.add_argument('--agent', dest='agent', default='TSACL', type=str)
    parser.add_argument('--norm', dest='norm', default='BN', type=str)
    parser.add_argument('--buffer_size', type=int, default=8192,
                        help='Buffer size for TS-ACL')
    parser.add_argument('--gamma', type=float, default=1,
                        help='Gamma parameter for TSACL')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs for base training in TSACL')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer in TSACL')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing for loss function in TSACL')
    parser.add_argument('--separate_decay', type=boolean_string, default=False,
                        help='Use separate weight decay for TSACL')
    parser.add_argument('--path_prefix', type=str, default='results',
    args = parser.parse_args()

    # Include unchanged general params
    args = SimpleNamespace(**vars(args), **config_default)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Set directories
    exp_start_time = time.strftime("%b-%d-%H-%M-%S", time.localtime())
    exp_path_1 = args.encoder + '_' + args.data
    exp_path_2 = args.agent + '_' + args.norm + '_' + exp_start_time
    exp_path = os.path.join(args.path_prefix, exp_path_1, exp_path_2)  # Path for running the whole experiment
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    args.exp_path = exp_path
    log_path = args.exp_path + '/log.txt'
    sys.stdout = Logger('{}'.format(log_path))
    if args.agent == 'TSACL':
        tune_and_experiment_multiple_runs_foracil(args)
    else:
        tune_and_experiment_multiple_runs(args)

    
