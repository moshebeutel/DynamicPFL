import argparse
import time

from opacus.accountants.utils import get_noise_multiplier


def compute_noise_multiplier(args):

    start_time = time.time()


    noise_multiplier = get_noise_multiplier(
        # epochs=int(args.global_epoch*args.user_sample_rate),
        steps=args.global_epoch,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        sample_rate=args.user_sample_rate,
        accountant="rdp",
        alphas= [alpha/10.0 for alpha in range(11, 10000, 11)]
    )

    end_time = time.time()

    print(f'noise_multiplier {noise_multiplier} to achieve {args.target_epsilon, args.target_delta}-DP')
    print(f'Compute took {end_time - start_time:.2f} seconds')
    return noise_multiplier

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=512, help="Number of clients")
    parser.add_argument('--local_epoch', type=int, default=4, help="Number of local epochs")
    parser.add_argument('--global_epoch', type=int, default=960, help="Number of global epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--user_sample_rate', type=float, default=8/512, help="Sample rate for user sampling")
    parser.add_argument('--target_epsilon', type=float, default=2.0, help="Target privacy budget epsilon")
    parser.add_argument('--target_delta', type=float, default=1/512, help="Target privacy budget delta")

    args = parser.parse_args()

    for eps in [16, 8, 4, 2, 1]:
       args.target_epsilon = eps
       compute_noise_multiplier(args)

