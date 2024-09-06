from pathlib import Path
import numpy as np
import pandas as pd

def read_log():
    
    dfs=[]
    i=0
    
    for dataset in ['CIFAR10']:
        for method in ['base', 'ours']:
            
            print(dataset, method)
                
            log_folder = Path('logs')
            assert log_folder.exists(), f'log folder {log_folder.as_posix()} does not exist'
            assert log_folder.is_dir(), f'{log_folder} is not a directory'
            
            log_file_path = log_folder / f'{dataset}_{method}.txt'
            assert log_file_path.exists(), f'{log_file_path} does not exist'
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
            
            df = pd.DataFrame({'dataset':dataset,
                               'method':method, 
                               'mean':np.mean(arr), 
                               'std':np.std(arr),
                               'median':np.median(arr),
                               'max':np.max(arr), 
                               'min':np.min(arr)}, index=[i])
            i+=1

            dfs.append(df)
            # print(np.mean(arr), np.std(arr), np.median(arr), np.max(arr), np.min(arr))
    
    df=pd.concat(dfs, ignore_index=True)
    print(df)
    
    
    # for l in lines:
    #     print('*****')
    #     print(l)
    #     print('$$$$$')
        
    # arr = np.array(lines[1]).astype('float')
    
    # print(arr.shape)


if __name__ == '__main__':
    dataset = 'CIFAR10'
    assert dataset in ['CIFAR10', 'SHVN', 'MNIST'], f'dataset {dataset} not recognized'
    method = 'base'
    assert method in ['base', 'ours'], f'method {method} unrecognized'
    
    # read_log(dataset, method)
    read_log()