# For use with the provided Dockerfile

# We first expand the provided empty filterbank file. Into this file we inject the simulated pulsars.


import filtools
import numpy as np
import glob 
import os
import pandas as pd
import random


data_folder='./tutorial_data/'
sample_file = 'zero_100sample.fil'
new_length = 840089
#length in pmpps 840089 but length may cause trouble when injecting
asc_folder = './all_asc/'

os.system(f'rm -r {data_folder}')
os.mkdir(data_folder)
if not os.path.isdir(asc_folder):
    os.system('wget https://github.com/larskuenkel/SKA-TestVectorGenerationPipeline/raw/master/ASC/ASC.zip')
    os.system('unzip -n -q ASC.zip \
   -d ./all_asc/')
    os.system('rm ASC.zip')

zero_file = 'zero.fil'

inputfile = filtools.FilterbankIO()
inputfile.read_header(sample_file)
f_range = inputfile.frequency_table()
tstart = inputfile.header['tstart']
data_old = inputfile.read_block(inputfile.total_samples())

data_new = np.zeros([new_length, data_old.shape[1]])
new_file = inputfile.clone_header()
new_file.write_header(zero_file)
new_file.write_block(data_new)

# Change for this for more simulated files
n_files = 50
print(f'{n_files} files will be created.')

#This defines the distributions of the simulated pulsars
# Uniform distributions with start_value:intervall_length
snr_dist = np.random.uniform(70.,70.,n_files)
dm_dist = np.random.uniform(200.,300.,n_files)
p0_dist = np.random.uniform(0.1,1.5,n_files)

all_asc = glob.glob(asc_folder+'*.asc')
print(f'{len(all_asc)} Profiles available')
used_asc = random.sample(all_asc, n_files)

data_path = f'{data_folder}data'

os.mkdir(data_path)

data_dict = {'snr':snr_dist, 'dm':dm_dist, 'p0':p0_dist, 'prof':used_asc}
df = pd.DataFrame(data_dict)

for (index, row) in df.iterrows():
    # old_name = FakePulsar_5407_1.630740_283.1_70.0_ASC_J1640+2224_1472.fil
    f0 = 1/row['p0']
    out_name = f"{data_path}/FakePulsar_{index}_{row['p0']:.6f}_{row['dm']:.1f}_{row['snr']:.1f}_{row['prof'].split('/')[-1].split('.')[0]}.fil"
    os.system(f"ft_inject_pulsar -o {out_name} --f0 {f0} -D {row['dm']} --rms 1 -p {row['prof']}  {zero_file}")
    df.at[index, 'FileName'] = out_name
df.to_csv(f'{data_folder}psr_para.csv')
print('Finished')