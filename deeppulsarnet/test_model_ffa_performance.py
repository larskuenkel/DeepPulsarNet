import matplotlib
matplotlib.use('Agg')
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
import argparse
import riptide
import scipy.signal
import pandas as pd
import datetime
import warnings
import sys
from sigpyproc.Readers import FilReader as reader
from sigpyproc import TimeSeries as sigpyproc_tseries
from sigpyproc import Header
warnings.filterwarnings("ignore")

import trainer
# from riptide.clustering import cluster_1d

from riptide import TimeSeries, ffa_search, peak_detection
# import riptide.pipelines as pipeline
#warnings.simplefilter("ignore", UserWarning)


def sigma_fit(stats, polydeg=2):
    x = stats['logpmid']
    y = stats['sigma']
    poly = np.poly1d( np.polyfit(x, y, polydeg) )
    def func(period):
        return poly(np.log(period))
    return func, poly.coefficients


def median_fit(stats, polydeg=2):
    x = stats['logpmid']
    y = stats['median']
    poly = np.poly1d( np.polyfit(x, y, polydeg) )
    def func(period):
        return poly(np.log(period))
    return func, poly.coefficients


def detrend_pgram(pgram):
    boundaries = peak_detection.segment(pgram.periods, pgram.tobs)
    snr = pgram.snrs.max(axis=1)
    stats = peak_detection.segment_stats(snr, boundaries)
    tfunc_m, polyco_m = median_fit(stats, polydeg=2)
    medians = tfunc_m(pgram.periods)
    tfunc_s, polyco_s = sigma_fit(stats, polydeg=2)
    sigmas = tfunc_s(pgram.periods)
    snrs = (snr - medians) / sigmas
    return snrs


def dethresh_pgram(pgram, snr_min=6.5, nsigma=6.5, polydeg=2):
    boundaries = peak_detection.segment(pgram.periods, pgram.tobs)
    snr = pgram.snrs.max(axis=1)
    stats = peak_detection.segment_stats(snr, boundaries)
    tfunc_m, polyco_m = median_fit(stats, polydeg=polydeg)
    medians = tfunc_m(pgram.periods)
    tfunc_s, polyco_s = sigma_fit(stats, polydeg=polydeg)
    sigmas = tfunc_s(pgram.periods)
    threshold = np.maximum(medians+ nsigma*sigmas, snr_min)
    #print(np.max(threshold))
    snrs = snr - threshold
    return snrs


def check_real_val(x_val, y_val, period, range_val=0.001):
    valid_yvals = y_val[(period-range_val < x_val) & (period+range_val > x_val)]
    if len(valid_yvals) > 1:
        snr_val = np.max(valid_yvals)
    else:
        snr_val = 0
    return snr_val


def check_max_val(x_val, y_val):
    max_pos = np.argmax(y_val)
    highest_peak_pos = x_val[max_pos]
    highest_amplitude = y_val[max_pos]
    return highest_peak_pos, highest_amplitude


def create_detec(output_im, period,resolution=0.00250, min_b=10, max_b=100, renorm=False, dethresh=False):
    tseries = TimeSeries.from_numpy_array(output_im, resolution)
    ts, pgram = ffa_search(
       tseries, period_min=0.09, period_max=2.5, bins_min=min_b, bins_max=max_b)
    # if period > 0.8:
    #     ts, plan, pgram = ffa_search(tseries,period_min=0.8,period_max=1.5,bins_min=80, bins_max=84)
    # elif period > 0.2:
    #     ts, plan, pgram = ffa_search(tseries,period_min=0.2,period_max=0.8,bins_min=20, bins_max=24)
    # else:
    #     ts, plan, pgram = ffa_search(tseries,period_min=0.1,period_max=0.2,bins_min=10, bins_max=14)
    #ts, plan, pgram = ffa_search(tseries,period_min=0.03,period_max=1.1,bins_min=10, bins_max=100)
#print(plan)
    pgram.metadata['dm'] = 0
    detec, _ = riptide.find_peaks(pgram)
    if renorm:
        snr = detrend_pgram(pgram)
    elif dethresh:
        snr = dethresh_pgram(pgram)
    else:
        snr = pgram.snrs.max(axis=1)
    periods = pgram.periods
    real_snr = check_real_val(periods, snr, period)
    highest_peak, highest_amplitude = check_max_val(pgram.periods, snr)
    return pgram, detec, snr, real_snr, highest_peak, highest_amplitude


# def filter_harmonics(detec, pgram):
#     periods = np.asarray([det.period for det in detec])
#     tobs = np.median([det.metadata['tobs'] for det in detec])
#     dbi = tobs / periods

#     clrad = 0.01
#     cluster_indices = cluster_1d(dbi, clrad)

#     clusters = [
#         pipeline.pipeline.DetectionCluster([detec[ii] for ii in indices])
#         for indices in cluster_indices
#         ]
#     cparams = list(map(pipeline.pipeline.DetectionCluster.to_dict, clusters))
#     cparams = pipeline.harmonic_filtering.flag_harmonics(cparams, tobs=pgram.metadata['tobs'], max_denom=50, max_distance=10.0, snr_tol=10)
#     #print(cparams)
#     fundamentals = []
#     for cl, par in zip(clusters, cparams):
#         print
#         if par["is_harmonic"]:
#             fund = clusters[par["fundamental_index"]]
#             frac = par["fraction"]
#             msg = "{!s} is a harmonic of {!s} with period ratio {!s}".format(cl, fund, frac)
#         else:
#             fundamentals.append(cl)

#     num_harmonics = len(clusters) - len(fundamentals)
#     # print("Flagged {:d} harmonics".format(num_harmonics))
#     clusters = fundamentals
#     # print("Retained {:d} final Candidates".format(len(clusters)))
#     return clusters 

def center_maximum(array):
    if len(array.shape) > 1:
        array_scrunched = np.mean(array, axis=0)
    else:
        array_scrunched = array
    shift = len(array_scrunched) //2 - np.argmax(array_scrunched)
    array = np.roll(array, shift, axis=-1)
    return array

def plot_candnum(df, plot_path, now=datetime.datetime.now()):
    plt.figure(figsize=(6,6))
    plt.scatter(df['unrelated_candidates_dm'], df['unrelated_candidates'], edgecolors='black', c=(0, 0, 0, 0), marker='*')
    max_val = np.max((df['unrelated_candidates_dm'].max(), df['unrelated_candidates'].max()))+5
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.xlabel('Candidates Dedispersion', fontsize=16)
    plt.ylabel('Candidates Neural Net', fontsize=16)
    line = np.linspace(0, max_val, 10)
    plt.plot(line, line, c='r')
    plt.savefig(f'{plot_path}candnumber_plot_{now}.png')
    # plt.show()


def plot_psr_sn(df, plot_path, now=datetime.datetime.now()):
    plt.figure(figsize=(6,6))
    plt.scatter(df['-1_real_dm'], df['-1_real'],edgecolors='black', c=(0, 0, 0, 0), marker='*')
    max_val = np.max((df['-1_real_dm'].max(), df['-1_real'].max()))+5
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.xlabel('S/N Pulsar Dedispersion', fontsize=16)
    plt.ylabel('S/N Pulsar Neural Net', fontsize=16)
    line = np.linspace(0, max_val, 10)
    plt.plot(line, line,c='r')
    plt.savefig(f'{plot_path}sn_plot_{now}.png')
    # plt.show()


def down_scrunch_array(array, factor):
    scrunched = array.sum(1)
    old_length = len(scrunched)
    new_length = old_length // factor
    trunc_length = int(new_length * factor)
    trunced = scrunched[:trunc_length]

    trunced_reshaped = trunced.reshape(-1, factor)

    summed = trunced_reshaped.sum(1)
    # print(array.shape, scrunched.shape, trunced_reshaped.shape, summed.shape)
    return summed


def main(obs_file,factor, verbose=1, dm0=False, length=-1, min_b=8, max_b=20, renorm=False, plot=False, rmdm0=False, dethresh=False, samples=100,
    model=None, edge=[0,0]):
# args.f, args.d, verbose=1, dm0=args.dm0, length=args.l



    #if 'htru' in file:
    #    resolution = 0.000436906666666667
    #else:
    #    resolution = 0.000640
    resolution = 0.00250

    df = pd.read_csv(obs_file)
    # df = df[df['Label']==1]
    if samples:
        df = df[:samples]



    column_names = ['file', 'pulsar', 'dm', 'period', 'FFA Peak', 
    'Detected', 'candidates_dm', 'unrelated_candidates_dm']
    df_perf = pd.DataFrame(columns=column_names)
    # df_peak = pd.DataFrame(columns=['file', 'period'] + start_val_range.tolist())
    # df_pred = pd.DataFrame(columns=['file', 'period'] + start_val_range.tolist())

    total_trials = 0
    total_detec = 0
    total_whole_detec = 0
    non_detection = []

    if model is not None:
        net = torch.load(f'./trained_models/{model}')
        output_resolution = net.output_resolution
        print(f'Loaded model. Output-Resolution: {output_resolution}')
        # for m in net.modules():
        #     if 'Conv' in str(type(m)):
        #         setattr(m, 'padding_mode', 'zeros')
        net.set_mode('dedisperse')
        cuda = 1
        device = torch.device("cuda:0" if cuda else "cpu")
        fake_loader = argparse.Namespace()
        fake_loader.batch_size = 1
        fake_loader.dataset = argparse.Namespace()
        # if 'htru' in file:
        #     fake_loader.dataset.edge = [32,0]
        fake_loader.dataset.edge = net.edge
        train_net = trainer.trainer(net, fake_loader, None, None, None, device, 0, 0, [
                                    0, 0, 0, 0, 0], 10, [0, 0, 0])
        train_net.mode = 'validation'

    if plot:
        if model is None:
            base_plot_folder = f'./ffa_dm_plot/'
        else:
            base_plot_folder = f'./ffa_model_plot/'
        try:
            os.mkdir(base_plot_folder)
        except FileExistsError:
            pass
        if model is None:
            plot_folder = f'./ffa_dm_plot/dm_{length}_{factor}/'
        else:
            plot_folder = f'./ffa_model_plot/{model}/'
        try:
            os.mkdir(plot_folder)
        except FileExistsError:
            pass

    for row in df.iterrows():
        file = row[1]['FileName']
        pulsar = row[1]['PSR JNAME']
        index = row[1]['Unnamed: 0']
        period = float(row[1]['PSR P0'])
        if period > 0.05:
            if not dm0:
                dm = float(row[1]['PSR DM'])
            else:
                dm = 0
            # label = row[1]['Label']
            df_perf.at[index, 'file'] = index
            df_perf.at[index, 'period'] = period
            df_perf.at[index, 'dm'] = dm
            df_perf.at[index, 'pulsar'] = pulsar

            if model is None:
                fil = reader(file)
                resolution = float(fil.header['tsamp'])
                # if not rmdm0:
                #     dedis_fil = fil.readBlock(0,length).dedisperse(dm)
                #     tseries_dm = down_scrunch_array(dedis_fil, factor)
                # else:
                data = fil.readBlock(0,length)
                delays = fil.header.getDMdelays(dm)
                if edge[1]==0:
                    data = data[edge[0]:,:]
                    delays = delays[edge[0]:]
                else:
                    data= data[edge[0]:-edge[1],:]
                    delays = delays[edge[0]:-edge[1]]
                if rmdm0:
                    data = data - data.mean(0)[None,:]
                new_length = data.shape[1]-delays[-1]
                new_length = new_length - new_length%factor
                new_array = np.zeros(new_length)
                for delay_index, delay in enumerate(delays):
                    new_array += data[delay_index, delay:delay+new_length]
                tseries_dm = new_array[:length].reshape(-1,factor).mean(1)
                output_resolution = resolution*factor
                pgram_dm, detec_dm, snr_dm, real_snr_dm, highest_peak_dm, highest_amplitude_dm = create_detec(
                tseries_dm, period, resolution=output_resolution, min_b=min_b, max_b=max_b, renorm=renorm, dethresh=dethresh)

                max_peak = real_snr_dm
                best_detected = 0
            else:
                output_im, _, _,_ = train_net.test_target_file(
                file, 0, start_val=0)
                output_im = output_im.detach().cpu().numpy()

                peaks = []
                detected_chans = []
                for chan in range(output_im.shape[1]):
                    detected_dm = 0
                    tseries_dm = output_im[0,chan,:]
                    pgram_dm, detec_dm, snr_dm, real_snr_dm, highest_peak_dm, highest_amplitude_dm = create_detec(
                    tseries_dm, period, resolution=output_resolution, min_b=min_b, max_b=max_b, renorm=renorm, dethresh=dethresh)

                    for single_detec in detec_dm:
                        # print(single_detec)

                        if np.abs(single_detec.period - period) < 0.001:
                            detected_dm = 1

                    peaks.append(real_snr_dm)
                    detected_chans.append(detected_dm)

                    if plot:
                        plt.figure(figsize=(8,4))
                        plt.axvline(period,lw=1.5, c='r', alpha=0.2)
                        if len(pgram_dm.periods) != len(snr_dm):
                            used_periods = pgram_dm.periods[:len(snr_dm)]
                        else:
                            used_periods = pgram_dm.periods
                        plt.plot(used_periods, snr_dm, marker='o', markersize=2, alpha=0.5)
                        plt.xlim(used_periods.min(), used_periods.max())
                        plt.xlabel('Trial Period (s)', fontsize=16)
                        plt.ylabel('S/N', fontsize=16)
                        plt.title(f'{index}: {pulsar} P0: {period} DM: {dm}')
                        plt.savefig(f'{plot_folder}{index}_{chan}.png')

                max_peak = np.max(peaks)
                best_detected = np.max(detected_chans)
                df_perf.at[index, 'Detected'] = best_detected

            # best_detected = np.max(detected_chans)


            df_perf.at[index, 'FFA Peak'] = max_peak

            # detected_dm = 0

            # for single_detec in detec_dm:

            #     if np.abs(single_detec.period - period) < 0.001:
            #         detected_dm = 1

            # if detected_dm == 1:
            #     df_perf.at[index, 'Detected'] = 1

            print(f"PSR: {index}, {max_peak}, {best_detected}")
            df.at[index, 'FFA']  = max_peak
            
            # if plot:
            #     plt.figure(figsize=(8,4))
            #     plt.axvline(period,lw=1.5, c='r', alpha=0.2)
            #     if len(pgram_dm.periods) != len(snr_dm):
            #         used_periods = pgram_dm.periods[:len(snr_dm)]
            #     else:
            #         used_periods = pgram_dm.periods
            #     plt.plot(used_periods, snr_dm, marker='o', markersize=2, alpha=0.5)
            #     plt.xlim(used_periods.min(), used_periods.max())
            #     plt.xlabel('Trial Period (s)', fontsize=16)
            #     plt.ylabel('S/N', fontsize=16)
            #     plt.savefig(f'{plot_folder}{index}.png')


    mean_snr = df_perf['FFA Peak'].mean()
    mean_detec = df_perf['Detected'].mean()
    print(mean_snr, mean_detec)
    now = datetime.datetime.now()

    set_name = obs_file.split('/')[-1].split('.csv')[0]
    if edge[0]!=0 or edge[1]!=0:
        edge_string = f'e_{edge[0]}_{edge[1]}_'
    else:
        edge_string = ''
        if model is None:
            base_plot_folder = f'./ffa_dm_plot/'
        else:
            base_plot_folder = f'./ffa_model_plot/'
        try:
            os.mkdir(base_plot_folder)
        except FileExistsError:
            pass
    if model is None:
        f_name = f'./ffa_dm/dm_{set_name}_{length}_{factor}_{min_b}_{max_b}_{samples}_{edge_string}{now.strftime("%Y-%m-%d_%H:%M")}.csv'
        try:
            os.mkdir( f'./ffa_dm/')
        except FileExistsError:
            pass
    else:
        f_name = f'./ffa_model/{model}_{set_name}_{min_b}_{max_b}_{samples}_{now.strftime("%Y-%m-%d_%H:%M")}.csv'
        try:
            os.mkdir( f'./ffa_model/')
        except FileExistsError:
            pass
    # f_name = f'./ffa_performance/bintest_{min_b}_{max_b}_{length}_{factor}_{now.strftime("%Y-%m-%d_%H:%M")}.csv'
    df_perf.to_csv(f_name)
    df=df.drop(columns='Unnamed: 0')
    # df=df.sort_values('FFA', ascending=False)
    # df=df.reset_index()
    # df = df.rename({'index': 'old_sorted'}, axis=1)
    #df.to_csv('Test.csv')
    print(f_name)


    return f_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grab parameters.')
    parser.add_argument('-f', type=str, help='Obs file')
    parser.add_argument('-m', type=str, help='Model')
    parser.add_argument('-d', type=int, default=1, help='Downsample factor')
    parser.add_argument('--samples', type=int, default=0, help='Downsample factor')
    parser.add_argument('-l', type=int, default=780000, help='Used time steps of filterbank.')
    parser.add_argument('--dm0', action='store_true', help='Use DM=0')
    parser.add_argument('--min', type=int, default=8, help='bins_min argument for ffa')
    parser.add_argument('--max', type=int, default=20, help='bins_max argument for ffa')
    parser.add_argument('--renorm', action='store_true', help='Renorm the FFA')
    parser.add_argument('--rmdm0', action='store_true', help='Renorm the FFA')
    parser.add_argument('--plot', action='store_true', help='Plot the FFA')
    parser.add_argument('--dethresh', action='store_true', help='Subtract the threshold.')
    parser.add_argument('--edge', type=int, default=(0,0), nargs=2, help='Edge value.')

    args = parser.parse_args()
    csv_name = main(args.f, args.d, verbose=1, dm0=args.dm0, length=args.l, min_b=args.min, max_b=args.max, renorm=args.renorm, plot=args.plot, rmdm0=args.rmdm0, dethresh=args.dethresh, samples=args.samples, model=args.m, edge=args.edge)
