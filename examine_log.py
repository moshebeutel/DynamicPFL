from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def read_log(log_folder: str|Path):
    
    log_folder = Path(log_folder) if isinstance(log_folder, str) else log_folder
    assert isinstance(log_folder, Path), f'Expected log_folser str or Path. Got {type(log_folder)}'
    assert log_folder.exists(), f'log folder {log_folder.as_posix()} does not exist'
    assert log_folder.is_dir(), f'{log_folder} is not a directory'
    
    dfs=[]
    i=0 
    
    for log_file_path in log_folder.iterdir():
        print(log_file_path.stem)
        # log_file_path = log_folder / f'{dataset}_{method}_{suffix}.txt'
        if not log_file_path.exists():
            print(f'{log_file_path} does not exist')
            continue
        lines = []
        record=False
        with open(log_file_path, 'r') as f:
            # l=1
            # num_lines = 0
            l = f.readline()
            # ss = l.split('mean accuracy : ')
            while l:
                l = f.readline()
                print(l)
                # num_lines += 1
                if l.startswith('mean accur'):
                    record=True
                elif l.startswith('acc matrix'):
                    record=False
                
                if record:
                    lines.append(l)
        # print(f'{log_file_path} num lines {num_lines}') 
        # l=f.readline()
        # lines.append(l)
        # if not lines:
        #     continue
        
        numbers = lines[1].split(', ')
        # numbers = l.split(', ')
        
        lnumbers = []
        for n in numbers:
            n=n.replace('\n', '')
            n=n.replace(']', '')
            n=n.replace('[', '')
            lnumbers.append(float(n))
        
        arr = np.array(lnumbers)
        
        
        plt.plot(arr, label=log_file_path.stem);
        
        df = pd.DataFrame({'dataset':"CIFAR10",
                        'method':log_file_path.stem, 
                        'mean':np.mean(arr), 
                        'std':np.std(arr),
                        'median':np.median(arr),
                        'max':np.max(arr), 
                        'best_iteration':np.argmax(arr),
                        'min':np.min(arr)}, index=[i])
        i+=1

        dfs.append(df)
    
    df=pd.concat(dfs, ignore_index=True)
    print(df)
    plt.title(f'{log_folder.stem}');
    plt.legend();
    plt.savefig(f'plots/chalsssls3333333333fvmsdlkvfmv.png') 
    # plt.show()

if __name__ == '__main__':

    # read_log(log_folder='logs/basline')
    # read_log(log_folder='logs/ours_gep_first')
    # read_log(log_folder='logs/CIFAR10_10_public_10_epochs_1_eps_0.02_sample_rate_sweep_history_size')
    read_log(log_folder='logs/CIFAR10_10_public_10_epochs_1_eps_0.02_sample_rate_sweep_basis_size')