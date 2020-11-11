#!/usr/bin/env python

# Finding pulsars in filterbank data

import torch
import numpy as np
import argparse
import utils
# from torchsummary import summary
import sys
import torch.nn as nn
import data_loader.data_loader as data_loader
from model.model_pulsar_net import pulsar_net
import trainer
import logger
import pandas as pd
import json
#import ffa_test_whole_perf


def main():
    # Parameters

    cuda = 1  # use cuda (1) or not (0)
    # torch.backends.cudnn.benchmark=True
    print(f"Cuda available: {torch.cuda.is_available()}")

    parser = argparse.ArgumentParser(description='Grab parameters.')
    parser.add_argument('-l', type=float, nargs='+',
                        default=[0.2, 2, 0.2, 1], help='Learn rate. [lr before first noise update, \
                        lr after first noise update, factor bewtween first part and second part of network].')
    parser.add_argument('-e', type=int, default=50, help='Epochs')
    parser.add_argument(
        '-d', type=float, default=[0.0, 0.0, 0.0, 0.], nargs='+', help='Dropout')
    parser.add_argument('-p', action='store_true',
                        help='Plot using visdom.')
    parser.add_argument('--samples', type=int, default=0,
                        help='Number of samples. Default: All')
    parser.add_argument('-c', action='store_true', help='Use lr scheduler.')
    parser.add_argument('--length', type=int,
                        default=60000, help='Length of data.')
    parser.add_argument('--kernel', type=int,
                        default=[2], nargs='+', help='Kernel size.')
    parser.add_argument('--channels', type=int, nargs='+',
                        default=[16, 16, 16, 16], help='Channels in the first encoder.')
    parser.add_argument('--dec_channels', type=int, nargs='+',
                        default=[0], help='Channels in the decoder.')
    parser.add_argument('--kernel2', type=int, default=0, help='Kernel size.')
    parser.add_argument('--channels2', type=int, nargs='+',
                        default=[0], help='Channels in the second encoder.')
    parser.add_argument('--batch', type=int, default=20, help='Batch size.')
    parser.add_argument('--decay', type=float,
                        default=0, help='Weight decay. Default: 0')
    parser.add_argument('--groups', type=int, default=[2, 1, 1, 1], nargs='+',
                        help='Channels per group in the group norm.')
    parser.add_argument('--path', type=str, default='zero_all.csv',
                        help='Set the path of the data csv.')
    parser.add_argument('--path_noise', type=str, default='palfa_test_noise_normal_minus8.csv',
                        help='Set the path of the noise csv.')
    # parser.add_argument('--weights', type=str, default='', help='Load saved weights.')
    parser.add_argument('--model', type=str, default='',
                        help='Load saved model.')
    parser.add_argument('--name', type=str, default='',
                        help='Name of the saved weights and model.')
    parser.add_argument('--noise', type=float, nargs='+',
                        default=[0.01, 1, 50, 0], help='Define how much noise is added. [start_value, step_size, max, use_saved]')
    parser.add_argument('--linear', type=int, nargs='+',
                        default=[128, 32], help='Define the hidden layers in the regressor')
    parser.add_argument('--stride', type=int, default=2,
                        help='Controls the stride of the convolution in the autoencoder.')
    parser.add_argument('--pool', type=int, default=1,
                        help='Use pooling  with this kernel in the second encoder.')
    parser.add_argument('--loops', type=int, default=1,
                        help='Loops while training on one batch')
    parser.add_argument('--bandpass', action='store_true',
                        help='Use bandpass.npy to make the signal more realistic.')
    parser.add_argument('--threshold', type=float, default=[0.004, 0, 0], nargs=3,
                        help='Increase the noise when loss drops below this value.')
    parser.add_argument('--binary', type=float, default=0.7,
                        help='Use binary output with chosen threshold. Also uses nn.BCEWithLogitsLoss() as loss.')
    parser.add_argument('--mode', type=str, default='full',
                        help='Set mode of training. Possible: autoencoder, full.')
    parser.add_argument('--rnn', type=int, nargs='+', default=[0, 0, 0],
                        help='Use lstm as regressor with this many layers.')
    parser.add_argument('--pool_multi', type=int, default=4,
                        help='Use this downsampling when using tcn_multi in deeper layers.')
    parser.add_argument('--freeze', type=int, default=-1,
                        help='Freeze everything but the classifying layers until this many epochs.')
    # parser.add_argument('--coordinates', action='store_true',
    #                     help='Add the coordinates as additonal channel.')
    parser.add_argument('--path_test', type=str, default='',
                        help='Set the path of the test_csv.')
    parser.add_argument('--layer', type=int, default=0,
                        help='Only train this many blocks in the autoencoder.')
    parser.add_argument('--edge', type=int, default=[0], nargs='+',
                        help='Ignore this many channels at the edges.')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Use a bidirectional LSTM.')
    parser.add_argument('--residual', action='store_true',
                        help='Do use residual connections.')
    parser.add_argument('--no_pad', action='store_true',
                        help='Do not use any padding in the convolutional net.')
    parser.add_argument('--pre_pool', type=int, default=0,
                        help='Subtract the mean with a kernel size of this parameter from the input.')
    # parser.add_argument('--pool_decoder', action='store_true',
    #                     help='Use pseudo-unpooling in the decoder. Does not really work nicely.')
    parser.add_argument('--overwrite_rnn', action='store_true',
                        help='Overwrite classifier.')
    parser.add_argument('--overwrite_classifier', action='store_true',
                        help='Overwrite classifier.')
    parser.add_argument('--add_classifier', action='store_true',
                        help='add classifier.')
    parser.add_argument('--overwrite_length', action='store_true',
                        help='Overwrite length.')
    # parser.add_argument('--tcn_dir', type=int, nargs=3,
    #                     default=[0, 0, 1], help='Defines how many tcn are used in the encoded state.')
    parser.add_argument('--tcn_kernel', type=int, nargs=4,
                        default=(1, 3, 5, 19), help='[ini, tcn[0], tcn[1]]')
    parser.add_argument('--tcn_layers', type=int, nargs=3,
                        default=(5, 4, 2), help='Layers in the tcn. [Blocks, initial layers, layers in block]')
    parser.add_argument('--tcn_channels', type=int, nargs=3,
                        default=[16, 4, 0], help='Channels in the tcn.')
    parser.add_argument('--tcn_dilation', type=int, nargs=3,
                        default=[20, 5, 2], help='Dilation in the tcn. [Start, add per layer, dilation of ini layers].')
    parser.add_argument('--add_chan', type=int,
                        default=0, help='Additional channels in the recovered state.')
    parser.add_argument('--no_reg', action='store_true',
                        help='Only use classification loss, not the regression loss.')
    parser.add_argument('--tcn_class', type=int, nargs='+',
                        default=[6, 5, 4, 1000, 30, 20], help='Use a tcn classifier. Para: [layers, kernel, channels, pool, [layers in lin]]')
    parser.add_argument('--acf_class', type=int, nargs='+',
                        default=[1, 1500, 5, 20, 10], help='Use a acf classifier. Para: [Use, acf_size, pool, [layers in lin]]')
    parser.add_argument('--fft_class', type=int, nargs='+',
                        default=[1, 10000, 2000, 5, 0, 20, 10], help='Use a fft classifier. Para: [mode, fft_size, trunc, pool, rnn, harm, [layers in lin]]')
    parser.add_argument('--stft', type=float, nargs='+',
                        default=[10000], help='Use a fft classifier. Para: [stft_size, stft_crop, rrn_chan, pool, num_layers, blocks, hop_length_fraction]')
    parser.add_argument('--ffa', type=int, nargs='+',
                        default=[5, 2, 8, 11, 1, 0, 0], help='Parameters of ffa classifier. norm currently in ffa_args.\
                        Para: [pool, layers, chan, kernel, norm, use amplitudes, pytorch_ffa]')
    # parser.add_argument('--simple_class', type=int,
    #                     default=0, help='Use a acf classifier. Para: [Use, fft_size, pool, [layers in lin]]')
    parser.add_argument('--crop', type=int,
                        default=0, help='Crop output by this much at both edges.')
    parser.add_argument('--ffa_test', type=str, default='',
                        help='Use the ffa test every 2 iterations.')
    parser.add_argument('--ada', action='store_true',
                        help='Use AdaBound instead of Adam.')
    parser.add_argument('--snr_range', type=float, nargs=3,
                        default=[0, 0, 0], help='Range of SNR. First number gives the number of noise samples')
    parser.add_argument('--dm_range', type=float, nargs=2,
                        default=[0, 2000], help='Range of DM.')
    parser.add_argument('--norm', action='store_true',
                        help='Normalise input dynamically with group norm.')
    parser.add_argument('--block_mode', type=str, default='cat',
                        help='Choose Mode for the combination of the block. add, cat, pool')
    parser.add_argument('--reduce_mode', type=str, default='mlp',
                        help='Choose Mode for the reduction at the end of each block. avg, max, mlp')
    parser.add_argument('--tcn_mode', type=str,
                        help='Choose Mode for the tcn-like layers. tcn, tcn_multi, seq, tied')
    parser.add_argument('--filter_size', type=int, nargs=2,
                        default=[0, 0], help='Choose size and Dilation of the mean filter for the input.')
    parser.add_argument('--clamp', type=int, nargs=3,
                        default=[65, -100, 100], help='Bias value and clamp range.')
    parser.add_argument('--reset_pre', action='store_true',
                        help='Reset the preprocessing option when loading a model.')
    parser.add_argument('--dec_mode', type=str, default='ups',
                        help='Choose mode for the decoder. conv, ups')
    parser.add_argument('--enc_mode', type=str, default='conv',
                        help='Choose mode for the multi layer encoder. conv, pool')
    parser.add_argument('--shift', action='store_true',
                        help='Shift the target according to the DM')
    parser.add_argument('--loss_weights', type=float, nargs=4,
                        default=(0.001, 0.001, 1, 1), help='Loss weights. [regression, classification, autoencoder')
    parser.add_argument('--class_mode', type=str, nargs='+',
                        help='Choose mode for the classification. acf, fft, simple, tcn, rnn, none')
    parser.add_argument('--multi_class', action='store_true',
                        help='Combine multiple classifiers')
    parser.add_argument('--train_single', action='store_true',
                        help='Learn individual classifiers in multi class')
    parser.add_argument('--gauss', type=float, nargs=4,
                        default=(27, 15 / 4, 1, 1), help='Parameters of the Gaussian that is used for the smoothing of the target.')
    parser.add_argument('--fft_loss', type=float,
                        default=0, help='Use fft loss with given weight.')
    parser.add_argument('--acf_loss', type=float,
                        default=0, help='Use acf loss with given weight.')
    parser.add_argument('--untie_weights', action='store_true',
                        help='If model with a tied weights is used, these will be untied.')
    # parser.add_argument('--aa', action='store_true',
    #                     help='Use aliased downsampling in the encoder.')
    parser.add_argument('--dm0_mode', type=str, default='none',
                        help='Choose how to handle dm0 signals. none cat sub')
    parser.add_argument('--add_test_to_train', type=int, default=0,
                        help='Add the pulsars of the test_set to the train_set x times.')
    parser.add_argument('--test_samples', type=float, nargs=2,
                        default=[5, 0.25], help='Samples that are used for the test classification \
                        and fraction that is needed for positive classification.')
    parser.add_argument('--nulling', type=float, nargs=8,
                        default=[0, 0, 0, 0, 0, 0, 0, 0], help='Null a part of the training signal. [chunk_max, length, length deviation,\
                        use_specaug_psr,use_specaug_combined,num,freq_para,time_para]')
    parser.add_argument('--use_val_as_test', action='store_true',
                        help='Use validation noise also for test set.')
    parser.add_argument('--acc_grad', type=int,
                        default=1, help='Accumulate gradient over this many batches.')
    parser.add_argument('--cmask', action='store_true',
                        help='Use channel masking layer based on the mean and std dev.')
    parser.add_argument('--rfimask', action='store_true',
                        help='Use rfi masking.')
    parser.add_argument('--net_chunks', type=int, nargs=2,
                        default=[1,0], help='Calculate tcn in chunks to save memory. [Chunks, overlap]')
    parser.add_argument('--crop_augment', type=float, default=0.0,
                        help='Change cropping in stft by set value.')
    parser.add_argument('--kfold', type=int, default=-1,
                        help='Choose which 5-fold to use. -1 means random validation/train set.')
    parser.add_argument('--dmsplit', action='store_true',
                        help='Split dm in output channels.')
    parser.add_argument('--focal', type=float,
                        default=0, help='Focal loss parameter.')
    parser.add_argument('--progress', action='store_false',
                        help='Do not print progress. (Nicer output with slurm).')
    parser.add_argument('--dmoverlap', type=float,
                        default=0.25, help='DM overlap when using dmsplit.')
    parser.add_argument('--break_grad', action='store_true',
                        help='Do not propagate gradients between dedispersion and classification.')
    parser.add_argument('--ffa_args', type=float, nargs=5,
                        default=[0.03, 1.1, 10, 12, 1], help='Set the arguments of the ffa calculation.\
                        [min_p0, max_p0, min_bin, max_bin, renorm]')
    parser.add_argument('--dm0_class', action='store_true',
                        help='Feed the DM 0  time series to the classifier.')
    parser.add_argument('--loss_pool_mse', action='store_true',
                        help='Do not use single channels as the input the the loss, instead use a maxpooled version.')
    parser.add_argument('--model_config', type=str, default='default_model_config.json',
                        help='Path to the file containing the hyperparameters of the model.')
    parser.add_argument('--model_parameter', type=str, action='append',
                        help='Change parameters of the model. Usage: --model_parameter "tcn_1_layer 4, encoder_conv_groups 2".\
                        Separate multiple parameters with a comma and use the quotation marks.')
    parser.add_argument('--class_configs', type=str, default=['class_stft.json'], nargs='+',
                        help='Name of the config files containing the hyperparameters of the classifiersin the model_configs folder.\
                        --class_configs class_1.json class_2.json')

    args = parser.parse_args()

    print(args.model_parameter)
    # Reading model parameters
    with open(f"./model_configs/{args.model_config}") as json_data_file:
        model_para_dict = json.load(json_data_file)

    # Converting to Namespace for easier parsing
    model_para = argparse.Namespace(**model_para_dict)

    # Changing parameters based on --model_parameter
    new_model_para = args.model_parameter
    if not new_model_para is None:
        new_model_para_split = []
        if isinstance(new_model_para, list):
            for new_para in new_model_para:
                new_para = new_para.split(',')
                new_model_para_split.extend(new_para)
        else:
            new_model_para_split = new_model_para.split(',')

        if not isinstance(new_model_para_split, list):
            new_model_para_split = [new_model_para_split]

        for changed_parameter in new_model_para_split:

            changed_parameter = changed_parameter.strip()
            split_para = changed_parameter.split()
            para_name = split_para[0].lstrip()
            if not hasattr(model_para, para_name):
                print(f"Parameter {para_name} does not yet exist in model config.")
                setattr(model_para, para_name, split_para[1:])
            else:
                old_type = type(getattr(model_para, para_name))
                if old_type is int:
                    setattr(model_para, para_name, int(split_para[1].strip()))
                elif old_type is float:
                    setattr(model_para, para_name, float(split_para[1].strip))
                elif old_type is list:
                    setattr(model_para, para_name, split_para[1:])
                elif old_type is str:
                    setattr(model_para, para_name, ' '.join(split_para[1:]))
                elif old_type is bool:
                    setattr(model_para, para_name, split_para[1:])
                else:
                    print(f"Parameter type not implemented yet for {changed_parameter}")
            #print(model_para[para_name])
            #model_para[para_name] = split_para[1:]

    torch.set_num_threads(4)
    print(torch.get_num_threads())
    # if args.simple_class:
    #     args.no_reg = 1

    # torch.backends.cudnn.enabled == True

    # args.channels2 = args.channels2 if args.channels2[0] else args.channels
    # args.pool2 = args.pool2 or args.pool
    # args.stride2 = args.stride2 or args.stride
    # args.kernel2 = args.kernel2 or args.kernel[0]

    if args.channels[0] != 0:
        down_factor = int((args.stride * args.pool) ** len(args.channels))
    else:
        down_factor = 1

    length, enc_length = args.length, args.length
    # args.kernel2, length = utils.test_parameters(
    #     length, args.kernel2, args.channels2, args.stride2, args.pool2, args.no_pad,part=1)

    enc_shape = (args.tcn_channels[-1], enc_length)
    # if len(args.kernel) > 1:
    #     args.kernel[-1], length = utils.test_parameters(
    #         length, args.kernel[1], args.channels, args.stride, args.pool, args.no_pad, part=2)

    # if args.tcn_layers:
    #     tcn_range = utils.calc_tcn_depth(args.tcn_kernel, args.tcn_layers)

    if args.pool != 1 and args.no_pad:
        print('No_pad and pool != 1 are incompatible.')
        sys.exit()

    # if args.kernel[0] != args.kernel[-1] and args.no_pad:
    #     print(' The option no_pad and two different kernel sizes are incompatible.')
    #     sys.exit()

    np.random.seed(2)
    torch.manual_seed(1)  # reproducible
    if cuda:
        torch.cuda.manual_seed(1)  # reproducible
    device = torch.device("cuda:0" if cuda else "cpu")

    #  Setup loggin, plotting and create data
    logging = logger.logger(args.p, args.name)

    train_loader, valid_loader, mean_period, mean_dm, mean_freq, example_shape, df_for_test = data_loader.create_loader(
        args.path, args.path_noise, args.samples, length, args.batch, args.edge, enc_shape=enc_shape, down_factor=down_factor,
        snr_range=args.snr_range, shift=args.shift, nulling=args.nulling, val_test=args.use_val_as_test, kfold=args.kfold,
        dmsplit=args.dmsplit, net_out=args.tcn_channels[2], dm_range=args.dm_range, dm_overlap=args.dmoverlap)

    if args.path_test != '' or args.use_val_as_test:
        _, test_loader, _, _, _, _, _ = data_loader.create_loader(
            args.path_test, None, 0, length, 1, args.edge,
            mean_period=mean_period, mean_freq=mean_freq, mean_dm=mean_dm, val_frac=1, test=True, test_samples=int(args.test_samples[0]),
            df_val_test=df_for_test)
        if args.add_test_to_train:
            df_test = test_loader.dataset.df
            df_test = df_test[df_test['Label'] == 1]
            df_test['MaskName'] = ''
            for k in range(args.add_test_to_train):
                train_loader.dataset.df = train_loader.dataset.df.append(
                    df_test)
                print(len(train_loader.dataset.df))
    else:
        test_loader = None

    # print('Data shape: {}'.format(example_shape))
    (channels, real_length) = example_shape
    example_shape_altered = example_shape
    if real_length != length:
        print('Example file short than expected! No padding implemented yet.')

    print('Train samples: {}'.format(len(train_loader.dataset)))

    if args.model:
        if args.untie_weights:
            net_ini = torch.load('./trained_models/{}'.format(args.model))
            net = utils.untie_weights(net_ini).to(device)
            del(net_ini)
        else:
            net = torch.load(
                './trained_models/{}'.format(args.model)).to(device)
        net.set_mode(args.mode)
        example_shape = net.input_shape
        example_shape_altered = example_shape
        # if args.overwrite_rnn:
        #     n_inputs = net.encoder_channels[-1]
        #     n_hidden = args.rnn[0]
        #     n_outputs = 4
        #     layers = args.rnn[1]
        #     bidirectional = args.bidirectional
        #     net.classifier = regressor_rnn(
        #         n_inputs, n_hidden, n_outputs, layers, bidirectional, drnn=args.rnn[2]).to(device)

        #     # torch.save(net, "./trained_models/{}.pt".format(args.name))
        #     # net = torch.load("./trained_models/{}.pt".format(args.name)).to(device)
        if args.overwrite_length:
            new_length = args.length
            # kernel, new_length, enc_length = utils.test_parameters(
            #     args.length, net.kernel_encoder, net.encoder_channels, net.stride, net.pool, args.no_pad, part=1)
            net.input_shape = (net.input_shape[0], new_length)
            # new_enc_shape = (args.tcn_channels[-1], enc_length)
            # train_loader.dataset.enc_shape = new_enc_shape
            # valid_loader.dataset.enc_shape = new_enc_shape
            net.create_loss_func()
            net.to(device)

        if args.overwrite_classifier or args.add_classifier:
            net.create_classifier_levels(args.class_mode, args.multi_class,
                                         args.no_reg, args.fft_class, args.acf_class, args.rnn, args.stft, dropout=args.d[3:],
                                         crop_augment=args.crop_augment, tcn_class=args.tcn_class, ffa_class=args.ffa,
                                         overwrite=args.overwrite_classifier, ffa_args=args.ffa_args, dm0_class=args.dm0_class)
            net.to(device)

        net.reset_optimizer(args.l, decay=args.decay,
                            freeze=args.freeze, init=1, ada=args.ada)
        train_loader.dataset.length = net.input_shape[1]
        valid_loader.dataset.length = net.input_shape[1]
        if test_loader is not None:
            test_loader.dataset.length = net.input_shape[1]

        if args.reset_pre:
            net.set_preprocess(net.input_shape, args.norm, args.filter_size,
                               bias=args.clamp[0], clamp=args.clamp[1:], cmask=args.cmask, rfimask=args.rfimask)
            net.to(device)

        for m in net.modules():
            if 'Conv' in str(type(m)):
                setattr(m, 'padding_mode', 'zeros')
        # if args.untie_weights:
        #     net = utils.untie_weights(net)
        #     paras = []
        #     for child in net.tcn.children():
        #         for single_module in child.modules():
        #             if single_module.__module__ == 'torch.nn.modules.conv':
        #                 #print(single_module.__module__)
        #                 #single_module.weight.data = single_module.weight.data
        #                 #single_module.bias.data = single_module.bias.data
        #                 dil = single_module.dilation[0]
        #                 # single_module.weight = copy.deepcopy(single_module.weight)
        #                 # single_module.bias = copy.deepcopy(single_module.bias)
        #                 if dil == 15 or dil ==17:
        #                     print(dil)
        #                     #print(single_module.weight)
        #                     paras.append(single_module.weight)
        #     if paras[1] is paras[0]:
        #         print('Parameters still tied')
        #     else:
        #         print('Parameters not tied anymore')
        for child in net.encoder.modules():
            if isinstance(child, nn.Dropout):
                child.p = args.d[0]
        if hasattr(net, 'tcn'):
            for child in net.tcn.modules():
                if isinstance(child, nn.Dropout):
                    child.p = args.d[1]
        for child in net.output_layer.modules():
            if isinstance(child, nn.Dropout2d):
                child.p = args.d[2]
        # if hasattr(net, 'classifier_stft_0'):
        #     for child in net.classifier_stft_0.modules():
        #         if isinstance(child, nn.LSTM):

        #             child.dropout = args.d[3]
        for (child_name, child) in net.named_modules():
            if child_name.startswith('classifier_stft'):
                if hasattr(child, 'rnn'):
                    child.rnn.dropout = args.d[4]
                if hasattr(child, 'ini_dropout'):
                    child.ini_dropout.p = args.d[3]
                # for chil in child.children():
                #     if isinstance(chil, nn.LSTM):
                #         chil.dropout = args.d[3]
                for chil in child.modules():
                    if isinstance(chil, nn.Dropout2d):
                        child.p = args.d[4]

        if args.noise[3] == 1:
            args.noise = net.noise
    else:
        net = pulsar_net(model_para, example_shape, 4, args.channels, args.kernel, args.l,
                         args.d, pool=args.pool, stride=args.stride, binary=args.binary,
                         rnn=args.rnn, list_channels_lin=args.linear,
                         groups=args.groups, bidirectional=args.bidirectional, residual=args.residual,
                         no_pad=args.no_pad, mode=args.mode,
                         tcn_kernel=args.tcn_kernel, tcn_layers=args.tcn_layers,
                         tcn_channels=args.tcn_channels, tcn_dilation=args.tcn_dilation,
                         dec_channels=args.dec_channels, add_chan=args.add_chan,
                         no_reg=args.no_reg, tcn_class=args.tcn_class, acf_class=args.acf_class, crop=args.crop,
                         fft_class=args.fft_class, multi_class=args.multi_class,
                         norm=args.norm, block_mode=args.block_mode, reduce_mode=args.reduce_mode, tcn_mode=args.tcn_mode,
                         filter_size=args.filter_size, clamp=args.clamp, dec_mode=args.dec_mode,
                         class_mode=args.class_mode,
                         gauss=args.gauss, enc_mode=args.enc_mode, pool_multi=args.pool_multi,
                         stft=args.stft, dm0=args.dm0_mode, cmask=args.cmask, rfimask=args.rfimask,
                         crop_augment=args.crop_augment, ffa=args.ffa,
                         ffa_args=args.ffa_args, dm0_class=args.dm0_class,
                         class_configs=args.class_configs).to(device)
        net.edge = train_loader.dataset.edge
        net.reset_optimizer(args.l, decay=args.decay,
                            freeze=args.freeze, init=1, ada=args.ada)

    net.create_loss_func(args.focal)
    # print(net)
    # net.set_mode(args.mode)
    print('Data shape: {}'.format(net.input_shape))
    # if args.mode == 'autoencoder' and args.layer != 0:
    #     net.set_layer(args.layer)
    print(net)

    net.net_chunks = args.net_chunks

    net.break_grad = args.break_grad
    # print(net)
    # from tensorboardX import SummaryWriter
    # dummy_input = torch.ones(example_shape).to(device)
    # dummy_input = torch.stack((dummy_input, dummy_input), dim=0)
    # with SummaryWriter(comment='Net1') as w:
    #     w.add_graph(net, (dummy_input,))
    net.train()
    net.save_noise(args.noise[:])
    net.save_mean_vals(mean_period, mean_dm, mean_freq)
    print('Noise: {}'.format(net.noise))
    train_loader.dataset.noise = net.noise
    valid_loader.dataset.noise = net.noise

    train_net = trainer.trainer(net, train_loader, valid_loader, test_loader, logging,
                                device, args.bandpass, args.binary, args.noise, args.threshold, args.l, pre_pool=args.pre_pool,
                                crop=args.crop, loss_weights=args.loss_weights, train_single=args.train_single,
                                fft_loss=args.fft_loss, acf_loss=args.acf_loss, test_frac=args.test_samples[1],
                                acc_grad=args.acc_grad,
                                loss_pool_mse=args.loss_pool_mse)

    command_string = 'python ' + ' '.join(sys.argv[:])

    train_net.logger.log_command(command_string)

    class_ok = 0

    for epoch in range(args.e):
        freeze_val = args.freeze - epoch
        train_net.logger.time_meter.reset()
        train_net.net.save_epoch(epoch)
        train_net.logger.epoch = epoch
        loss_train = train_net.run('train', args.loops, only_class=args.no_reg, print_progress=args.progress)
        reg_loss_train = train_net.logger.loss_meter.value()[0]
        clas_loss_train = train_net.logger.loss_meter_2.value()[0]
        im_loss_train = train_net.logger.loss_meter_3.value()[0]
        loss_valid = train_net.run(
            'validation', args.loops, only_class=args.no_reg, print_progress=args.progress)
        reg_loss = train_net.logger.loss_meter.value()[0]
        clas_loss = train_net.logger.loss_meter_2.value()[0]
        im_loss = train_net.logger.loss_meter_3.value()[0]
        if test_loader is not None:
            loss_test = train_net.run('test', 1, only_class=1, print_progress=args.progress)
        else:
            loss_test = None
        if hasattr(net.scheduler, 'get_lr'):
            train_net.logger.log_loss(epoch, net.scheduler.get_lr()[0], train_net.net.noise,
                                      loss_train, loss_valid, loss_test, reg_loss_train,
                                      clas_loss_train, im_loss_train, reg_loss, clas_loss, im_loss)
        else:
            train_net.logger.log_loss(epoch, net.optimizer.param_groups[0]['lr'], train_net.net.noise,
                                      loss_train, loss_valid, loss_test, reg_loss_train,
                                      clas_loss_train, im_loss_train, reg_loss, clas_loss, im_loss)


        if hasattr(train_net.logger, 'last_test_mcc'):
            if not train_net.logger.last_test_mcc == 'None':
                if train_net.logger.last_test_mcc > 0.9:
                    torch.save(train_net.net,
                               "./trained_models_best/{}_{}_{:.2f}.pt".format(args.name, epoch, train_net.logger.last_test_mcc))
        if args.c:
            if hasattr(net.scheduler, 'get_lr'):
                train_net.net.scheduler.step()
            else:
                train_net.net.scheduler.step(loss_valid)
        train_net.logger.save_best_values(
            epoch, loss_train, loss_valid, loss_test)
        train_net.update_noise(
            epoch, args.name, decay=args.decay, ada=args.ada, freeze=freeze_val)
        if epoch % 1 == 0:
            if args.name:
                # torch.save(train_net.net.state_dict(),
                #            "./trained_models/{}_weights.pt".format(args.name))
                torch.save(train_net.net,
                           "./trained_models/{}.pt".format(args.name))
            else:
                torch.save(train_net.net, './trained_models/last_model.pt')
                # torch.save(train_net.net.state_dict(),
                #            './trained_models/last_model_weights.pt')

        # if train_net.net.mode=='full' or train_net.net.mode=='classifier':
        #     if not class_ok and epoch % 1 == 0 and len(train_net.net.classifiers):
        #         class_ok = train_net.check_classifier()

        if epoch % 1 == 0 and args.ffa_test != '':
            # ffa_count, non_detec, csv_ffa = ffa_test_whole_perf.main(
            #     args.ffa_test, args.name, 4, crop=train_net.crop, verbose=0, train_net=train_net)
            # train_net.logger.save_ffa_values(ffa_count, non_detec)
            pass

    with open("../log_training_runs.txt", "a") as myfile:
        myfile.write("\n #{} \n {:.2f}   {} {}  {:} FFA: {}".format(
            command_string, train_net.net.noise[0], np.array2string(
                logging.best_values, precision=5)[1:-1],
            np.array2string(logging.best_test_values, precision=5)[
                1:-1], np.array2string(logging.conf_val, precision=2)[1:-1],
            np.array2string(logging.best_ffa_values)[1:-1]))
    return train_net.logger.best_values


if __name__ == "__main__":
    print('Start')
    best_values = main()
