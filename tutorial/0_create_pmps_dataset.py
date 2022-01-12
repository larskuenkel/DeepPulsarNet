import pandas as pd
import numpy as np
import glob
import os

#enter the path where the pmps sample set has been unpacked here
data_path = os.getcwd()+ '/pmps_sample_set/'
train_path = data_path+'training/'
test_path = data_path+'test/'

train_files = glob.glob(train_path+'*.fil')
print(f'{len(train_files)} files found')
psr_names = ['',] * len(train_files)
periods = [np.nan,] * len(train_files)
dms = [np.nan,] * len(train_files)
labels = [2,] * len(train_files)
snrs = [np.nan,] * len(train_files)

data_dict = {'JNAME':psr_names, 'P0':periods, 'DM':dms, 'Label':labels, 'FileName':train_files, 
             'SNR': snrs}
df = pd.DataFrame(data=data_dict)
os.system(f'mkdir ../deeppulsarnet/datasets/')
df.to_csv(f'../deeppulsarnet/datasets/noiseset_pmps_sample_train.csv')
print(f"Created: ../deeppulsarnet/datasets/noiseset_pmps_sample_train.csv")
print(f"To use the set use the option: --path_noise noiseset_pmps_sample_train.csv")

test_files = glob.glob(test_path+'*.fil')
print(f'{len(test_files)} files found')
psr_names = ['',] * len(test_files)
periods = [np.nan,] * len(test_files)
dms = [np.nan,] * len(test_files)
labels = [2,] * len(test_files)
snrs = [np.nan,] * len(test_files)

data_dict = {'JNAME':psr_names, 'P0':periods, 'DM':dms, 'Label':labels, 'FileName':test_files, 
             'SNR': snrs}
df = pd.DataFrame(data=data_dict)
df.to_csv(f'../deeppulsarnet/datasets/noiseset_pmps_sample_test.csv')
print(f"Created: ../deeppulsarnet/datasets/noiseset_pmps_sample_test.csv")
