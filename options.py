import argparse
import math
import random
import numpy as np
import wandb

def set_seed(seed, cudnn_enabled=True):
    """for reproducibility

    :param seed:
    :return:
    """
    import torch

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
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")

    # >>> ***GEP
    parser.add_argument('--num_public_clients', type=int, default=50,
                        help="Number of public clients for gradient embedding subspace compute")
    parser.add_argument('--virtual_publics', type=int, default=500,
                        help="Number of virtual public clients for gradient embedding subspace compute")
    parser.add_argument('--basis_size', type=int, default=360, help="Embedding subspace basis size")
    parser.add_argument('--history_size', type=int, default=500, help="Previous Gradients used to"
                                                                     " span subspace")
    # <<< ***GEP

    parser.add_argument('--user_sample_rate', type=float, default=8/512, help="Sample rate for user sampling")

    parser.add_argument('--noise_multiplier', type=float, default=0.0,
                        help="If not zero, use this factor instead of epsilon accountant ")
    parser.add_argument('--noise_multiplier_residual', type=float, default=0.0,
                        help="If not zero, use this factor instead of epsilon accountant for residual")

    parser.add_argument('--target_epsilon', type=float, default=1, help="Target privacy budget epsilon")
    parser.add_argument('--target_delta', type=float, default=1/512, help="Target privacy budget delta")
    parser.add_argument('--clipping_bound', type=float, default=0.1, help="Gradient clipping bound")
    parser.add_argument('--clipping_bound_residual', type=float, default=0.1, help="Residual clipping bound")

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
    parser.add_argument("--log-every", type=int, default=10, help="log every X selected epochs")
    parser.add_argument("--eval-after", type=int, default=150, help="eval only after X selected epochs")

    parser.add_argument('--wandb', type=bool, default=True)

    parser.add_argument('--exp-name', type=str, default='')

    parser.add_argument('--save-model-path', type=str, default='saved_checkpoints')
    parser.add_argument('--resume-path', type=str, default='')
    parser.add_argument('--min-acc-save', type=float, default=24, help="Minimum accuracy to save model")





    args = parser.parse_args()

    if not args.exp_name:
        args.exp_name = args.dataset + '_epochs_' + str(args.global_epoch) + '_epsilon_' + str(args.target_epsilon)


    # Weights & Biases
    if args.wandb:
        total_noise=math.sqrt(args.noise_multiplier_residual**2.0 + args.noise_multiplier**2.0)
        total_noise = float(int((total_noise * 2) + 0.5)) / 2.0
        wandb.init(project=f"total_noise_multiplier_{total_noise}_runs", name=args.exp_name, config=args)

    set_seed(args.seed)
    return args
