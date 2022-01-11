# For use with the provided Dockerfile

# We first expand the provided empty filterbank file. Into this file we inject the simulated pulsars.


import filtools
import numpy as np
import glob 
import os

data_folder='./tutorial_data/'
sample_file = 'zero_100sample.fil'
new_length = 840089
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
n_files = 20

#This defines the distributions of the simulated pulsars
# Uniform distributions with start_value:intervall_length
snr_para = '--snr uniform --snrparams 70:0'
dm_para = '--dm uniform --dmparams 200:100'
p0_para = '--p0 uniform --p0params 0.1:1.4'


set_path = f'{data_folder}'
pars_path = f'{data_folder}/pars'
preds_path = f'{data_folder}/pred'
cmd_path = f'{data_folder}/cmd'
data_path = f'{data_folder}data'
#fake_prof_path = f'{data_folder}/prof_fake'


# In[11]:


#!mkdir {data_path}


# In[12]:


# Reset Folders 
os.system(f'rm -r {set_path}')
os.system(f'mkdir {set_path}')
os.system(f'mkdir {pars_path}')
os.system(f'mkdir {preds_path}')
os.system(f'mkdir {cmd_path}')
os.system(f'mkdir {data_path}')
#os.system(f'mkdir {fake_prof_path}')


# 1. Creates Pars
# 2. Create Preds
# 3. Create command list
# 4. Execute command list

#pipeline_path = '/home/soft/SKA-TestVectorGenerationPipeline/code/pulsar_injection_pipeline/v1.0/src/'
pipeline_path = '/home/soft/SKA-TestVectorGenerationPipeline/'

command_string = f'-d {pars_path} -w {set_path}/out_pars.txt -s {n_files} {snr_para} {p0_para} {dm_para}'

os.system(f'python {pipeline_path}PARS/CandidateParGenerator.py {command_string}')

#asc_folder = f'/home/soft/SKA-TestVectorGenerationPipeline/resources/ASC/all_ASC/'
asc_folder = f'/home/soft/SKA-TestVectorGenerationPipeline/ASC/all_ASC/'

chunks = n_files // 1000
f_high = int(np.ceil(max(f_range)))
f_low = int(np.floor(min(f_range)))
command_string = f'--tel PARKES  -p {pars_path} -d {preds_path} -s 2000 --f1 {f_low} --f2 {f_high} --mjd1 {tstart} --mjd2 {tstart+0.2}'
print(command_string)
for chunk in range(chunks+1):
   os.system(f'python {pipeline_path}PREDS/GeneratePredictorFiles.py {command_string}')


command_string = f'--asc {asc_folder} --pred {preds_path} --out {cmd_path} --noise {zero_file} --batch {n_files}'

os.system(f'python {pipeline_path}INJECT/InjectPulsarCommandCreator.py  {command_string}')


cmd_files = glob.glob(cmd_path+ '/*.txt')
print(cmd_files)


# In[18]:


for cmd_file in cmd_files:
    os.system(f'python {pipeline_path}INJECT/InjectPulsarAutomator.py --cmd {cmd_file} --out {data_path}')
