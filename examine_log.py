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
    
    opt = 'clients_100_rate_0.1_epochs_20'
    
    for dataset in ['CIFAR10']:
        for method in ['base', 'base_gep','ours', 'ours_gep']:
            for pub in ['', '_3_public', '_5_public' , '_10_public', '_10_public_5_basis']:
                
                suffix = f'{opt}{pub}'
            
                print(dataset, method, suffix)
                
                log_file_path = log_folder / f'{dataset}_{method}_{suffix}.txt'
                if not log_file_path.exists():
                    print(f'{log_file_path} does not exist')
                    continue
                lines = []
                record=False
                with open(log_file_path, 'r') as f:
                    l=1
                    while l:
                        l = f.readline()
                        if l.startswith('mean accur'):
                            record=True
                        elif l.startswith('acc matrix'):
                            record=False
                        
                        if record:
                            lines.append(l)
                
                numbers = lines[1].split(', ')
                
                lnumbers = []
                for n in numbers:
                    n=n.replace(']\n', '')
                    n=n.replace('[', '')
                    lnumbers.append(float(n))
                
                arr = np.array(lnumbers)
                
                plt.plot(arr, label=f'{dataset}_{method}_{suffix}');
                
                df = pd.DataFrame({'dataset':dataset,
                                'method':method, 
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
    plt.savefig(f'plots/{log_folder.stem}.png') 

if __name__ == '__main__':

    # read_log(log_folder='logs/basline')
    # read_log(log_folder='logs/ours_gep_first')
    read_log(log_folder='logs/gep_all_2_epochs')