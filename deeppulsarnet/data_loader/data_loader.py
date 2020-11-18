import pandas as pd
import data_loader.dataset as dataset
import numpy as np
import torch.utils.data as data_utils
import sys


def create_loader(csv_file, csv_noise, samples, length, batch, edge=0, mean_period=0, mean_dm=0, mean_freq = 0, val_frac=0.2, test=False, enc_shape=(1,1000), down_factor=4,
    snr_range=[0,0], shift=False, test_samples=11, nulling=(0,0,0,0,0,0,0,0), shuffle_valid=True,val_test=False, df_val_test=None, kfold=-1,
    dmsplit=False, net_out=1, dm_range=[0,2000], dm_overlap=0.25):
        # Create train and validation loader
    if df_val_test is None:
        df = load_csv(csv_file, samples, snr_range, dm_range)
    else:
        df = df_val_test

    df_noise = load_csv(csv_noise, 0, noise_set=True)

    df_noise_noise = df_noise[df_noise['Label'] == 2]
    df_noise_psr = df_noise[df_noise['Label'] == 3]

    example_shape = load_example(df, length, edge)
    if mean_period and mean_dm:
        print("Using existing mean period and dm: {} {}".format(
            mean_period, mean_dm))
    else:
        # means = df.mean()
        # mean_period = means['P0']
        # mean_dm = means['DM']
        mean_period = df['P0'].mean()
        mean_dm = df['DM'].mean()
        print("Mean period and dm: {} {}".format(mean_period, mean_dm))
    # df['P0'] = df['P0'] / mean_period
    # df['DM'] = df['DM'] / mean_dm

    df['f0'] = 1 /df['P0']
    if not mean_freq:
        mean_freq = df['f0'].mean()
    # df['f0'] = df['f0'] / mean_freq
    # print(df['f0'][:10])

    if test == False:
        dm_min = df['DM'].min()
        dm_max = df['DM'].max()
        dm_range = (dm_min, dm_max)
    else:
        dm_range = (0,10000)

    print(csv_file)
    train_indices, valid_indices = create_indices(len(df), val_frac, kfold=kfold)
    train_noise_indices, valid_noise_indices = create_indices(len(df_noise), val_frac, kfold=kfold)

    if val_frac!=1:
        train_dataset = dataset.FilDataset(
        df.iloc[train_indices], df_noise.iloc[train_noise_indices], example_shape[0], length, 1, edge, enc_shape, 
        down_factor=down_factor, shift=shift, nulling=nulling, dmsplit=dmsplit, net_out=net_out, dm_range=dm_range, dm_overlap=dm_overlap)
        train_loader = data_utils.DataLoader(train_dataset, shuffle=True,
                                             batch_size=batch, num_workers=2, drop_last=True)
    else:
        train_loader = None
    valid_dataset = dataset.FilDataset(
        df.iloc[valid_indices], df_noise.iloc[valid_noise_indices], example_shape[0], length, 0, edge, enc_shape, down_factor=down_factor, test=test, shift=shift,
        test_samples=test_samples, dmsplit=dmsplit, net_out=net_out, dm_range=dm_range, dm_overlap=dm_overlap)
    if test:
        shuffle_valid = False
    # else:
    #     shuffle_valid = True
    valid_loader = data_utils.DataLoader(
        valid_dataset, batch_size=batch, num_workers=1, shuffle=shuffle_valid, drop_last=False)

    if val_test:
        # print(f"Val/Test Noise: {len(valid_noise_indices)}, Test PSR: {len(df_noise_psr)}")
        # df_for_test = pd.concat([df_noise.iloc[valid_noise_indices], df_noise_psr], axis=0)
        print(f"Test PSR: {len(df_noise_psr)}")
        df_for_test = df_noise_psr
    else:
        df_for_test = None
    return train_loader, valid_loader, mean_period, mean_dm, mean_freq, example_shape, df_for_test


def load_csv(csv_file, samples, snr_range=[0,0], dm_range=[0,2000], noise_set=False):
        # Load csv and truncate to given number of samples
    if csv_file is not None:
        data_frame = pd.read_csv('./datasets/{}'.format(csv_file), comment='#')
        data_frame = data_frame[(data_frame['DM'] > 0) |
                                (data_frame['DM'].isnull())]
        data_frame = data_frame[(data_frame['P0'] > 0) |
                                (data_frame['P0'].isnull())]

        if snr_range[0] or snr_range[1]:
            if 'SNR' in data_frame and noise_set:
                data_frame = data_frame[data_frame['SNR'] > -snr_range[0]]
                data_frame = data_frame.loc[((snr_range[1] < data_frame['SNR'])&(data_frame['SNR']<snr_range[2]))
                |(data_frame['SNR']<0)]

        if dm_range!=[0,2000]:
            old_size = len(data_frame)
            data_frame = data_frame[(data_frame['DM'] > dm_range[0]) |
                                (data_frame['DM'].isnull())]
            data_frame = data_frame[(data_frame['DM'] < dm_range[1]) |
                                (data_frame['DM'].isnull())]
            new_size = len(data_frame)
            cut = old_size - new_size
            if cut !=0:
                data_frame = data_frame[:-cut]
            print(old_size, new_size, len(data_frame), cut)


        if samples:
            data_frame = data_frame.sample(samples)
        print("Number of samples: {}".format(len(data_frame)))
    else:
        data_frame = pd.DataFrame(data={'FileName':[np.nan], 'Label':[0]})

    return data_frame


def create_indices(size, val_frac, kfold=-1):
        # Create train and validation indices
    all_indices = np.arange(size)
    val_size = int(val_frac * len(all_indices))
    if val_frac == 1:
        train_indices = []
        valid_indices = all_indices
    elif kfold == -1:
        valid_indices = np.random.choice(all_indices, val_size, replace=False)
        train_indices = np.delete(all_indices, valid_indices)
    else:
        if kfold >= 5:
            print('Choose kfold <5')
            sys.exit()
        half_size = size // 2
        half_val_size = val_size // 2
        valid_indices = np.concatenate((all_indices[kfold*half_val_size:kfold*half_val_size+half_val_size],
            all_indices[half_size + kfold*half_val_size:half_size + kfold*half_val_size+half_val_size]))
        train_indices = np.delete(all_indices, valid_indices)
    # print(valid_indices, len(valid_indices), val_size, len(train_indices), train_indices)
    # np.save(f'train_indices_{size}.npy', train_indices)
    # np.save(f'valid_indices_{size}.npy', valid_indices)
    return train_indices, valid_indices


def load_example(df, length, edge):
    if len(edge)==1:
        edge = [edge[0],edge[0]]
    for i in range(20):
        file = df.iloc[i]['FileName']
        if not pd.isna(file):
            example_file = dataset.load_filterbank(file, length, 0, edge=edge)[0]
            break
    return example_file.shape
