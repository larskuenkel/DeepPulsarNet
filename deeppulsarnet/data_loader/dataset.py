import torch
import torch.utils.data as data_utils
import pandas as pd
from sigpyproc.Readers import FilReader as reader
import numpy as np
import matplotlib.pyplot as plt


class FilDataset(data_utils.Dataset):
        # Dataset which contains the filterbanks
    def __init__(self, df, df_noise, channels, length, mode, edge=0, enc_shape=(1, 1000), test=False, down_factor=4, 
                 test_samples=11, nulling=(0, 0, 0, 0, 0, 0, 0),
                 dmsplit=False, net_out=1, dm_range=(0,10000), dm_overlap = 1/4,
                 set_based=False, sim_prob=0.5, discard_labels=False):
        self.df = df
        self.df.reset_index(drop=True, inplace=True)
        #self.df.sort_values('Unnamed: 0', inplace=True)
        self.data_files = self.df['FileName']
        self.mode = mode
        self.length = length
        self.channels = channels

        nan_value = float("NaN")
        self.psr_sim = self.df.replace("", nan_value)
        self.psr_sim.dropna(subset=["FileName"], inplace=True)

        if len(edge)==1:
            self.edge = [edge[0],edge[0]]
        elif len(edge)==2:
            self.edge = edge
        # print(self.edge)
        self.test = test

        self.noise_df = df_noise.copy()

        self.noise_df['Ini Label'] = self.noise_df['Label'].copy()

        noise_psr = np.count_nonzero((self.noise_df['Label'].to_numpy()%2)==1)
        noise_nopsr = np.count_nonzero((self.noise_df['Label'].to_numpy()%2)==0)

        if discard_labels:
            self.noise_df['Label'] = 2

        print(f"Obs set: Noise: {noise_nopsr}; PSR: {noise_psr}")

        #self.noise_df = self.noise_df.loc[self.noise_df['Label'] == 2]
        self.noise = (0.3, 2)
        self.enc_shape = enc_shape
        self.down_factor = down_factor
        # self.shift = shift
        self.test_samples = test_samples
        self.nulling = nulling

        self.net_out = net_out
        self.dm_range = dm_range
        self.dmsplit = dmsplit

        self.set_based = set_based
        self.sim_prob = sim_prob

        if self.dmsplit:
            print(dm_overlap)
            self.dm_parts = np.linspace(dm_range[0], dm_range[1], self.net_out+1)
            dm_overlap_total = int((self.dm_parts[1]- self.dm_parts[0])*dm_overlap)
            self.dm_ranges = np.zeros((2, self.net_out))
            for section in range(self.net_out):
                self.dm_ranges[0,section] = self.dm_parts[section]-dm_overlap_total 
                self.dm_ranges[1,section] = self.dm_parts[section+1]+dm_overlap_total
            self.dm_ranges[0,0] = 0
            self.dm_ranges[1,-1] = 10000
        else:
            self.dm_ranges = None
        print('DM Ranges:')
        print(self.dm_ranges)

        if 'MaskName' in self.df.columns:
            self.use_precomputed_output = 1
        else:
            self.use_precomputed_output = 0

    def __getitem__(self, idx):
        if not self.set_based:
            labels, name = grab_labels(self.df.iloc[idx], index=idx)
            sim_file = self.df.iloc[idx]['FileName']
            if len(self.noise_df)>1:
                noise_file_index = self.noise_df.sample().index
                noise_row = self.noise_df.loc[noise_file_index]
                labels = np.append(labels, noise_row['Unnamed: 0'])
                noise_file = noise_row['FileName'].item()
            else:
                labels = np.append(labels, -1)
                noise_file = ''
                noise_file_index = -1
            if self.use_precomputed_output:
                # if not 'J' in name:
                target_file = self.df.iloc[idx]['MaskName']
                # else:
                # target_file = ''
            else:
                target_file = ''
        else:
            labels, name = grab_labels(self.noise_df.iloc[idx], index=idx, set_based=self.set_based)
            noise_file = self.noise_df.iloc[idx]['FileName']
            obs_label = int(labels[2])
            if obs_label % 2 == 0:
                roll = np.random.uniform()
                choice = 1 if roll < self.sim_prob else 0
            else:
                choice = 0

            if choice:
                sim_file_index = self.psr_sim.sample().index
                sim_row = self.psr_sim.loc[sim_file_index]
                labels = np.append(labels, sim_row['Unnamed: 0'])
                labels[0] = sim_row['P0']
                labels[1] = sim_row['DM']
                labels[2] = 1
                sim_file = sim_row['FileName'].values[0]

                # target_file = self.psr_sim.iloc[sim_file_index]['MaskName'].values[0]
                target_file = sim_row['MaskName'].values[0]
            else:
                labels = np.append(labels, -1)
                sim_file = ''
                target_file = ''
            noise_file_index = idx

        labels = np.append(labels, noise_file_index)

        if self.dmsplit:
            dm_indexes = check_range(self.dm_ranges, labels[1])
        else:
            dm_indexes = [0]
            self.net_out = 1


        if len(dm_indexes)<1:
            labels = np.concatenate((labels, -1, -1), axis=None)
        else:
            labels = np.concatenate((labels, dm_indexes[0], dm_indexes[-1]), axis=None)

        noisy_data, target_data = load_filterbank(
            sim_file, self.length, self.mode, target_file, noise_file, self.noise, edge=self.edge, test=self.test, labels=labels, enc_length=self.enc_shape[
                1], down_factor=self.down_factor,
            dm=labels[1], test_samples=self.test_samples, name=name, nulling=self.nulling,
            dmsplit=self.dmsplit, dm_indexes=dm_indexes, net_out=self.net_out)
        # print(noisy_data.shape, target_data.shape)
        return noisy_data, target_data, labels

    def __len__(self):
        if not self.set_based:
            return len(self.df)
        else:
            return len(self.noise_df)


def load_filterbank(file, length, mode, target_file='', noise=np.nan, noise_val=(1, 1, 1), edge=[0,0], start_val=2000, test=False,
                    labels=[0,0,0], enc_length=1875, down_factor=1, dm=0, test_samples=11, name='', nulling=(0, 0, 0, 0, 0, 0, 0, 0),
                    dmsplit=False, dm_indexes=0, net_out=1):
        # Load filterbank from disk with sigpyproc
    # print(file, noise, down_factor)
    # print(target_file)
    if not test:
        if not (pd.isna(file) or file == ''):
            current_file = reader(file)
            start, nsamps = choose_start(mode, current_file, length, start_val)
            current_data = current_file.readBlock(start, nsamps)
            orig_array = np.asarray(current_data)#.T
            data_array = orig_array
            # if nulling[0]:
            #     if nulling[0] > 0:
            #         nulled_chunks = np.random.randint(nulling[0])
            #     else:
            #         # For easier testing directly give chunk number
            #         nulled_chunks = - nulling[0]
            #     for i in range(nulled_chunks):
            #         null_length = int(
            #             np.abs(np.random.normal(nulling[1], nulling[2])))
            #         null_start = int(np.random.randint(data_array.shape[1]))
            #         data_array[:, null_start:null_start + null_length] = 0
        if not pd.isna(noise) and noise_val != 0:
            current_noise_file = reader(noise)
            start_noise, nsamps = choose_start(
                mode, current_noise_file, length, start_val, down_factor=down_factor)
            current_noise = current_noise_file.readBlock(start_noise, nsamps)

            if pd.isna(file) or file == '':
                data_array = np.asarray(current_noise)#.T
                #  no multiplication needed due to later normalisation
                orig_array = np.zeros_like(data_array)
            else:
                if mode:
                    noise_val_used = np.random.uniform(noise_val[0], np.min(
                        (noise_val[1], noise_val[0] + noise_val[2] * 2)))
                    # noise_val = np.random.triangular(noise_val[0], noise_val[0], noise_val[1])
                    # added_val = np.random.randint(
                    #     -noise_val[4], noise_val[4] + 1)
                    # added_val = np.random.uniform(-noise_val[4], noise_val[4])
                else:
                    noise_val_used = noise_val[0]
                    added_val = 0
                if noise_val_used == 0:
                    data_array = orig_array
                else:
                    data_array = orig_array / noise_val_used + \
                        np.asarray(current_noise)# + added_val
                    # print(noise_val_used, current_noise)
            # if nulling[4]:
            #     data_array = spec_augment(spec=data_array, num_mask=nulling[5], freq_masking=nulling[6],
            #                               time_masking=nulling[7], value=data_array.mean())
                # plt.imshow(data_array,aspect='auto')
                # plt.show()
        enc_down = int(length / down_factor)
        if target_file != '' and not pd.isna(file):
            current_target = np.load(target_file)
            max_length = current_target.shape[0] - current_target.shape[0]%down_factor
            current_target = current_target[:max_length].reshape(-1, down_factor).max(1)

            start_down = int(start / down_factor)

            current_target = current_target[start_down: start_down + enc_down]

            target_array = np.zeros((net_out, len(current_target)), dtype='float32')
            # print(dm_indexes, target_array.shape)
            for index in dm_indexes:
                target_array[index, :] = current_target
        else:
            target_array = np.zeros((net_out, enc_down), dtype='float32')
            if labels[2] == 3 or labels[2] == 5:
                target_array.fill(np.nan)

        if not edge[1] and not edge[0]:
            return data_array, target_array
        else:
            if not edge[1]:
                return data_array[edge[0]:, :], target_array
            else:
                return data_array[edge[0]:-edge[1], :], target_array
    else:
        current_file = reader(noise)
        samples = test_samples
        current_data = current_file.readBlock(0, int(current_file.header['nsamples']))#.T
        file_length = current_data.shape[1]
        actual_length = min(file_length, length)
        data_array = np.zeros((samples, current_data.shape[0], actual_length))
        start_vals = np.linspace(
            0, file_length - actual_length, samples, dtype=int)
        for (i, start) in zip(range(samples), start_vals):
            # if file_length != actual_length:
            #     start = np.random.randint(file_length - actual_length)
            # else:
            #     start = 0
            data_array[i, :, :] = current_data[:, start:start + actual_length]
        if not edge[1] and not edge[0]:
            if samples == 1:
                return data_array[0,:,:], data_array[0,:,:]
            else:
                return data_array, data_array
        else:
            if not edge[1]:
                if samples == 1:
                    return data_array[0,edge[0]:,:], data_array[0,edge[0]:,:]
                else:
                    return data_array[:,edge[0]:, :], data_array[:,edge[0]:, :]
            else:
                if samples == 1:
                    return data_array[0,edge[0]:-edge[1],:], data_array[0,edge[0]:-edge[1],:]
                else:
                    return data_array[:,edge[0]:-edge[1], :], data_array[:,edge[0]:-edge[1], :]


def grab_labels(row, index=0, set_based=0):
    # Grab the target labels for the regressor
    if not set_based:
        if row.loc['Label']!=3:
            label_array = np.asarray(
            (row.loc['P0'], row.loc['DM'], row.loc['Label'], row.loc['Unnamed: 0'], index)).astype('float32')
        else:
            label_array = np.asarray(
            (row.loc['PSR P0'], row.loc['PSR DM'], row.loc['Label'], row.loc['Unnamed: 0'], index)).astype('float32')
    else:
        if 'PSR P0' not in row:
            label_array = np.asarray(
            (row.loc['P0'], row.loc['DM'], row.loc['Label'], row.loc['Unnamed: 0'], index)).astype('float32')
        else:
            label_array = np.asarray(
            (row.loc['PSR P0'], row.loc['PSR DM'], row.loc['Label'], row.loc['Unnamed: 0'], index)).astype('float32')
    name = row.loc['JNAME']
    return label_array, name


def load_noise(noise, length, edge):
    # Load a noisy file
    current_file = reader(noise)
    start, nsamps = choose_start(1, current_file, length, 0)
    current_data = current_file.readBlock(start, nsamps)
    data_array = np.asarray(current_data).T
    if not edge:
        return data_array
    else:
        return data_array[edge:-edge, :]


def choose_start(mode, current_file, length, start_val, down_factor=1):
    if length:
        if mode:
            samples = current_file.header['nsamples']
            start = int(np.random.randint(500,
                                          int(samples - length - 500) / down_factor) * down_factor)
            nsamps = length
        else:
            start = start_val
            nsamps = length
    else:
        start = 500
        nsamps = -1
    return start, nsamps


def check_range(dm_range, val):
    indexes = []
    for k in range(dm_range.shape[1]):
        if val >=dm_range[0,k] and val <= dm_range[1,k]:
            indexes.append(k)
    return indexes

def load_file_for_prediction(file, length, edge, chunks=2, reverse=True):
    # Load a pulsar_observation
    
    batch_size = chunks
    if reverse == True:
        batch_size *= 2

    current_file = reader(file)
    start, nsamps = choose_start(1, current_file, length, 0)
    current_data = current_file.readBlock(0, int(current_file.header['nsamples']))#.T

    if edge[1]:
        current_data = current_data[edge[0]:edge[1], :]
    else:
        current_data = current_data[edge[0]:, :]

    file_length = current_data.shape[1]
    actual_length = min(file_length, length)
    
    data_array = np.zeros((batch_size, current_data.shape[0], actual_length))
    start_vals = np.linspace(
            0, file_length - actual_length, chunks, dtype=int)
    for (i, start) in zip(range(chunks), start_vals):
        data_array[i, :, :] = current_data[:, start:start + actual_length]
        if reverse:
            data_array[i+chunks, :, :] = data_array[i, :, ::-1]

    return torch.Tensor(data_array)
