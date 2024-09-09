import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--num_clients', type=int, default=500, help="Number of clients")
    parser.add_argument('--local_epoch', type=int, default=4, help="Number of local epochs")
    parser.add_argument('--global_epoch', type=int, default=40, help="Number of global epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")

    # >>> ***GEP
    parser.add_argument('--num_public_clients', type=int, default=10,
                        help="Number of public clients for gradient embedding subspace compute")
    parser.add_argument('--basis_size', type=int, default=5, help="Embedding subspace basis size")
    parser.add_argument('--history_size', type=int, default=30, help="Previous Gradients used to"
                                                                     " span subspace")
    # <<< ***GEP

    parser.add_argument('--user_sample_rate', type=float, default=0.1, help="Sample rate for user sampling")

    parser.add_argument('--target_epsilon', type=float, default=1, help="Target privacy budget epsilon")
    parser.add_argument('--target_delta', type=float, default=1e-1, help="Target privacy budget delta")
    parser.add_argument('--clipping_bound', type=float, default=0.5, help="Gradient clipping bound")

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



    args = parser.parse_args()
    return args
