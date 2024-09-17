import argparse
import random
import numpy as np
import torch
import wandb
from pFedGP.utils import str2bool
def set_seed(seed, cudnn_enabled=True):
    """for reproducibility

    :param seed:
    :return:
    """

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = cudnn_enabled
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--num_clients', type=int, default=512, help="Number of clients")
    parser.add_argument('--local_epoch', type=int, default=4, help="Number of local epochs")
    parser.add_argument('--global_epoch', type=int, default=960, help="Number of global epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")

    # >>> ***GEP
    parser.add_argument('--num_public_clients', type=int, default=10,
                        help="Number of public clients for gradient embedding subspace compute")
    parser.add_argument('--basis_size', type=int, default=5, help="Embedding subspace basis size")
    parser.add_argument('--history_size', type=int, default=30, help="Previous Gradients used to"
                                                                     " span subspace")
    # <<< ***GEP

    parser.add_argument('--user_sample_rate', type=float, default=8/512, help="Sample rate for user sampling")

    parser.add_argument('--target_epsilon', type=float, default=1, help="Target privacy budget epsilon")
    parser.add_argument('--target_delta', type=float, default=1/512, help="Target privacy budget delta")
    parser.add_argument('--clipping_bound', type=float, default=0.1, help="Gradient clipping bound")

    parser.add_argument('--fisher_threshold', type=float, default=0.4, help="Fisher information threshold for parameter selection")
    parser.add_argument('--lambda_1', type=float, default=0.1, help="Lambda value for EWC regularization term")
    parser.add_argument('--lambda_2', type=float, default=0.05, help="Lambda value for regularization term to control the update magnitude")

    parser.add_argument('--device', type=int, default=0, help='Set the visible CUDA device for calculations')

    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")

    parser.add_argument('--no_clip', action='store_true')
    parser.add_argument('--no_noise', action='store_true')

    parser.add_argument('--dataset', type=str, default='CIFAR10')

    parser.add_argument('--dir_alpha', type=float, default=100)

    parser.add_argument('--dirStr', type=str, default='')

    parser.add_argument('--store', action='store_true')

    parser.add_argument('--appendix', type=str, default='')

    parser.add_argument("--seed", type=int, default=42, help="seed value")

    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=10, help="eval every X selected epochs")
    parser.add_argument("--eval-after", type=int, default=50, help="eval only after X selected epochs")

    parser.add_argument('--wandb', type=bool, default=True)

    parser.add_argument('--exp-name', type=str, default='')
    #############################
    #       GP args             #
    #############################

    parser.add_argument('--use-gp', type=str2bool, default=True, help="use gaussian process as "
                                                                       "personalization mechanism")
    parser.add_argument("--n-kernels", type=int, default=16, help="number of kernels")

    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--loss-scaler', default=1., type=float, help='multiplicative element to the loss function')
    parser.add_argument('--kernel-function', type=str, default='RBFKernel',
                        choices=['RBFKernel', 'LinearKernel', 'MaternKernel'],
                        help='kernel function')
    parser.add_argument('--objective', type=str, default='predictive_likelihood',
                        choices=['predictive_likelihood', 'marginal_likelihood'])
    parser.add_argument('--predict-ratio', type=float, default=0.5,
                        help='ratio of samples to make predictions for when using predictive_likelihood objective')
    parser.add_argument('--num-gibbs-steps-train', type=int, default=5, help='number of sampling iterations')
    parser.add_argument('--num-gibbs-draws-train', type=int, default=20, help='number of parallel gibbs chains')
    parser.add_argument('--num-gibbs-steps-test', type=int, default=5, help='number of sampling iterations')
    parser.add_argument('--num-gibbs-draws-test', type=int, default=30, help='number of parallel gibbs chains')
    parser.add_argument('--outputscale', type=float, default=8., help='output scale')
    parser.add_argument('--lengthscale', type=float, default=1., help='length scale')
    parser.add_argument('--outputscale-increase', type=str, default='constant',
                        choices=['constant', 'increase', 'decrease'],
                        help='output scale increase/decrease/constant along tree')



    args = parser.parse_args()

    if not args.exp_name:
        args.exp_name = args.dataset + '_epochs_' + str(args.global_epoch) + '_epsilon_' + str(args.target_epsilon)


    # Weights & Biases
    if args.wandb:
        wandb.init(project="emg_gp_moshe", name=args.exp_name)
        wandb.config.update(args)

    set_seed(args.seed)
    return args
