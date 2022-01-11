import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from sigpyproc.Readers import FilReader as reader
import pandas as pd

data_folder = '/home/DeepPulsarNet/tutorial/fake_data/data/'
set_name = 'tutorial_pmps'
created_files = glob.glob(f'{data_path}/*')

out_folder = f'{set_path}/data_peaks/'

os.system(f'rm -r {out_folder}')
os.system(f'mkdir {out_folder}')

example_fil = reader(created_files[0])
tsamp = example_fil.header['tsamp']


correct_files = []
bad_files = []
approx = []
dedis_names = []
max_dedis_vals = []

# dm_shift replaces the shift argument in train_pulsar_net.py
# The pulses are shifted in the output based on the dm in a way that the output peak is at the same position where the pulse is in the middle channel of the input
dm_shift = True

for file in created_files[:]:
    name = file.split('/')[-1]
    new_name = name.split('.fil')[0]
    name_split = name.split('_')
    full_path = out_folder + new_name +'.npy'
    dm = float(name_split[3])
    period = float(name_split[2])
    approx_dist = 3/4. * period / tsamp
    fil = reader(file)
    dedis = fil.dedisperse(dm, gulp=10000000)
    max_val = np.max(dedis)
    #down = dedis.downsample(down_fac)
    down = dedis
    down -= np.median(down)
    down /= np.max(down)
    peaks, prop = scipy.signal.find_peaks(down, height=0.3, distance=approx_dist)
    
    if dm_shift:
        dm_delays = fil.header.getDMdelays(dm)
        middle_delay = dm_delays[len(dm_delays)//2]
        peaks += middle_delay
        if max(peaks) > 
    y_val = np.ones_like(peaks[:3]) 

    dummy = np.zeros_like(down).astype(bool)
    dummy[peaks] =1
    acf = scipy.signal.fftconvolve(dummy,dummy[::-1])
    acf /= np.max(acf)
    middle = int(len(acf)/2)
    try:
        height = np.max(acf[middle+25: middle+5000]) * 0.3
        acf_peaks, _ = scipy.signal.find_peaks(acf[middle-5: middle+5000], height=height)
        #print(acf_peaks)
        per_calc = (acf_peaks[1] -acf_peaks[0]) * tsamp
        approx.append((period, per_calc))
        if np.abs(period - per_calc)/period>0.15:
            bad_files.append(file)
        else:
            correct_files.append(file)
            dedis_names.append(full_path)
            max_dedis_vals.append(max_val)
    except IndexError:
        print(file + 'did not work')
    np.save(full_path, dummy)

approx_arr = np.asarray(approx)
np.save(f'{set_path}/approx_periods_{down_fac}.npy', approx_arr)
plt.scatter(approx_arr[:,0], approx_arr[:,1])
plt.savefig('approximated_periods.png')
print('Number of good files:', len(correct_files))

# Now create the csv files containing the information about the training set

raw_file_paths = correct_files
raw_file_names = [i.split('/')[-1] for i in raw_file_paths]
psr_names = ['',]*len(raw_file_paths)
periods = [float(i.split('_')[2]) for i in raw_file_names]
duty_cycles = [float(i.split('_')[3]) for i in raw_file_names]
dms = [float(i.split('_')[3]) for i in raw_file_names]
snrs = [float(i.split('_')[4]) for i in raw_file_names]
print(len(snrs), len(max_dedis_vals))

data_dict = {'JNAME':psr_names, 'P0':periods, 'DM':dms, 'Label':np.ones_like(psr_names), 'FileName':raw_file_paths, 
             'SNR': snrs, 'MaskName': dedis_names, 'MaxVal': max_dedis_vals, 'DutyCycle': duty_cycles}
df = pd.DataFrame(data=data_dict)

# Empty lines are appended, in emty lines no simulated psr is loaded
dummy_line = {'JNAME':'Noise', 'P0':np.nan, 'DM':np.nan, 'Label':0, 'FileName':'', 
             'SNR': np.nan, 'MaskName': '', 'MaxVal': np.nan, 'DutyCycle': np.nan}

df_noise = df.copy()
for i in range(len(df)):
    df_noise = df_noise.append(dummy_line, ignore_index=True)

os.system(f'mkdir ,,/deepulsarnet/datasets/')
df.to_csv(f'../datasets/simset_{set_name}.csv')
df_noise.to_csv(f'../datasets/simset_{set_name}_noise.csv')
