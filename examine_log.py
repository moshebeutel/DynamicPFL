from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_log(log_folder: str | Path):
    log_folder = Path(log_folder) if isinstance(log_folder, str) else log_folder
    assert isinstance(log_folder, Path), f'Expected log_folser str or Path. Got {type(log_folder)}'
    assert log_folder.exists(), f'log folder {log_folder.as_posix()} does not exist'
    assert log_folder.is_dir(), f'{log_folder} is not a directory'

    dfs = []
    i = 0

    for log_file_path in log_folder.glob('*.txt'):

        print(log_file_path.name)

        if not log_file_path.exists():
            print(f'{log_file_path} does not exist')
            continue
        lines = []
        record = False
        with open(log_file_path, 'r') as f:
            l = 1
            while l:
                l = f.readline()
                if l.startswith('mean accur'):
                    record = True
                elif l.startswith('acc matrix'):
                    record = False

                if record:
                    lines.append(l)

        numbers = lines[1].split(', ')

        lnumbers = []
        for n in numbers:
            n = n.replace(']\n', '')
            n = n.replace('[', '')
            lnumbers.append(float(n))

        arr = np.array(lnumbers)
        dataset=log_folder.stem.split('_')[0]
        plt.plot(arr, label=log_file_path.name);

        df = pd.DataFrame({'dataset': dataset,
                           'method': log_file_path.name,
                           'mean': np.mean(arr),
                           'std': np.std(arr),
                           'median': np.median(arr),
                           'max': np.max(arr),
                           'best_iteration': np.argmax(arr),
                           'min': np.min(arr)}, index=[i])
        i += 1

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(log_folder / f'{log_folder.stem.replace("0.", "0_")}.csv', index=False)
    print(df)
    plt.title(f'{log_folder.stem}');
    plt.legend();
    plt.savefig(f'plots/{log_folder.name}.png')


if __name__ == '__main__':
    # read_log(log_folder='logs/basline')
    # read_log(log_folder='logs/ours_gep_first')
    # read_log(log_folder='logs/CIFAR10_10_public_100_epochs_2_eps_0.02_sample_rate_sweep_basis_size')
    parser = ArgumentParser('Examine Logs')
    parser.add_argument('log_folder', type=str)
    args = parser.parse_args()
    read_log(log_folder=args.log_folder)
