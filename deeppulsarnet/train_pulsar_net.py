#!/usr/bin/env python

# Finding pulsars in filterbank data

import torch
import numpy as np
import argparse
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

    # cuda = 1  # use cuda (1) or not (0)
    # torch.backends.cudnn.benchmark=True
    print(f"Cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        cuda = 1
    else:
        cuda = 0

    parser = argparse.ArgumentParser(description='Grab parameters.')
    parser.add_argument('--model_config', type=str, default='default_model_config.json',
                        help='Path to the file containing the hyperparameters of the model.')
    parser.add_argument('--model_parameter', type=str, action='append',
                        help='Change parameters of the model. Usage: --model_parameter "tcn_1_layer 4, encoder_conv_groups 2".\
                        Separate multiple parameters with a comma and use the quotation marks.')
    parser.add_argument('--class_configs', type=str, default=['class_stft.json'], nargs='+',
                        help='Name of the config files containing the hyperparameters of the classifiers in the model_configs folder.\
                        --class_configs class_1.json class_2.json')
    parser.add_argument('-l', type=float, nargs='+',
                        default=[1e-3, 5e-4, 1], help='Learn rate. [lr before first noise update, \
                        lr after first noise update, factor bewtween first part and second part of network].')
    parser.add_argument('-e', type=int, default=50, help='Epochs')
    parser.add_argument('--noise', type=float, nargs='+',
                        default=[0.01, 20, 2, 0], help='Define how much noise is added. [start_value, max, step_size, use noise from loaded model]')
    parser.add_argument('-p', action='store_true',
                        help='Plot using visdom.')
    parser.add_argument('--samples', type=int, default=0,
                        help='Number of samples. Default: All')
    parser.add_argument('-c', action='store_false',
                        help='Disable lr scheduler.')
    parser.add_argument('--length', type=int,
                        default=60000, help='Length of data.')
    parser.add_argument('--batch', type=int, default=20, help='Batch size.')
    parser.add_argument('--decay', type=float,
                        default=0, help='Weight decay. Default: 0')
    parser.add_argument('--path', type=str,
                        help='Set the path of the data csv.')
    parser.add_argument('--path_noise', type=str,
                        help='Set the path of the noise csv.')
    parser.add_argument('--model', type=str, default='',
                        help='Load saved model.')
    parser.add_argument('--name', type=str, default='',
                        help='Name of the saved weights and model.')
    # parser.add_argument('--bandpass', action='store_true',
    #                     help='Use bandpass.npy to make the signal more realistic.')
    parser.add_argument('--threshold', type=float, default=[0.1, 0, 2], nargs=3,
                        help='Increase the noise when loss drops below this value.\
                        [Value, added value with noise, which metric (0:total loss, 1: classification loss, 2: mcc)]')
    parser.add_argument('--mode', type=str, default='full',
                        help='Set mode of training. Possible: dedisperse, full, classifier.')
    parser.add_argument('--freeze', type=int, default=-1,
                        help='Freeze everything but the classifying layers until this many epochs.')
    # parser.add_argument('--coordinates', action='store_true',
    #                     help='Add the coordinates as additonal channel.')
    parser.add_argument('--path_test', type=str, default='',
                        help='Set the path of the test_csv.')
    parser.add_argument('--edge', type=int, default=[0], nargs='+',
                        help='Ignore this many channels at the edges.')
    parser.add_argument('--loops', type=int, default=1,
                        help='Loops while training on one batch')
    parser.add_argument('--overwrite_classifier', action='store_true',
                        help='Overwrite classifier.')
    parser.add_argument('--add_classifier', action='store_true',
                        help='add classifier.')
    parser.add_argument('--overwrite_length', action='store_true',
                        help='Overwrite length.')
    parser.add_argument('--no_reg', action='store_false',
                        help='Do not only use classification loss, also use regression loss. currently broken.')
    parser.add_argument('--ffa_test', type=str, default='',
                        help='Use the ffa test every 2 iterations.')
    parser.add_argument('--snr_range', type=float, nargs=3,
                        default=[0, 0, 0], help='Range of SNR. First number gives the number of noise samples')
    parser.add_argument('--dm_range', type=float, nargs=2,
                        default=[0, 2000], help='Range of DM.')
    # parser.add_argument('--shift', action='store_true',
    #                     help='Shift the target according to the DM. currently broken.')
    parser.add_argument('--loss_weights', type=float, nargs=5,
                        default=(0.001, 0.001, 1, 1, 1), help='Loss weights. [regression, classification, reconstruction, single_classifiers, candidates]\
                        regression not used currently')
    parser.add_argument('--train_single', action='store_false',
                        help='Do not learn individual classifiers in multi class, only learn the combined result.')
    parser.add_argument('--gauss', type=float, nargs=4,
                        default=(27, 15 / 4, 1, 1), help='Parameters of the Gaussian that is used for the smoothing of the target.')
    parser.add_argument('--add_test_to_train', type=int, default=0,
                        help='Add the pulsars of the test_set to the train_set x times.')
    parser.add_argument('--test_samples', type=float, nargs=2,
                        default=[5, 0.25], help='Samples that are used for the test classification \
                        and fraction that is needed for positive classification.')
    parser.add_argument('--nulling', type=float, nargs=8,
                        default=[0, 0, 0, 0, 0, 0, 0, 0], help='Null a part of the training signal. Not implemented currently .[chunk_max, length, length deviation,\
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
                        default=[1, 0], help='Calculate tcn in chunks to save memory. [Chunks, overlap]')
    parser.add_argument('--kfold', type=int, default=-1,
                        help='Choose which 5-fold to use. -1 means random validation/train set.')
    parser.add_argument('--dmsplit', action='store_true',
                        help='Split dm in output channels.')
    parser.add_argument('--progress', action='store_false',
                        help='Do not print progress. (Nicer output with slurm).')
    parser.add_argument('--dmoverlap', type=float,
                        default=0.25, help='DM overlap when using dmsplit.')
    parser.add_argument('--break_grad', action='store_true',
                        help='Do not propagate gradients between dedispersion and classification.')
    parser.add_argument('--dm0_class', action='store_true',
                        help='Feed the DM 0  time series to the classifier.')
    parser.add_argument('--loss_pool_mse', action='store_true',
                        help='Do not use single channels as the input the the loss, instead use a maxpooled version.')
    parser.add_argument('--clamp', type=int, nargs=3,
                        default=[0, 0, 10000], help='Bias value and clamp range.')
    parser.add_argument('--crop', type=int,
                        default=0, help='Crop output by this much at both edges.')
    parser.add_argument('--reset_pre', action='store_true',
                        help='Reset the preprocessing option when loading a model.')
    parser.add_argument('--set_based', action='store_true',
                        help='Cycle through the whole observation set during one epoch and add random simulations.')
    parser.add_argument('--sim_prob', type=float, default=0.5,
                        help='Simulation probability when set_based is used.')
    parser.add_argument('--relabel_set', action='store_true',
                        help='Correct labels in the observation set when the network thinks it sees a pulsar.')
    parser.add_argument('--relabel_set_slow', action='store_true',
                        help='Alternative mode for relabelling.')
    parser.add_argument('--relabel_thresholds', type=int, nargs=2,
                        default=[0.85, 0.5], help='Relabel threshold. If the softmax is above the first value, the label changes to pulsar;\
                        If it is above the second value when the reverse is used it is not labelled as a pulsar.')
    parser.add_argument('--discard_labels', action='store_true',
                        help='Discard all labels in the observation set.')
    parser.add_argument('--reverse_batch', action='store_true',
                        help='Reverse batch after each batch and try to predict negative.')
    parser.add_argument('--class_weight', type=float, nargs=2,
                        default=[1, 1], help='Weight of the classes.')
    parser.add_argument('--added_cands', type=int,
                        default=0, help='Number of additional candidates per file per classifier.')
    parser.add_argument('--psr_cands', action='store_true',
                        help='Also use a candidate at the position of the pulsar period during training.')
    parser.add_argument('--cands_threshold', type=float,
                        default=0, help='Threshold under which candidates are filtered.')

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
                print(
                    f"Parameter {para_name} does not yet exist in model config.")
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
                    print(
                        f"Parameter type not implemented yet for {changed_parameter}")
            # print(model_para[para_name])
            #model_para[para_name] = split_para[1:]

    torch.set_num_threads(4)
    print(torch.get_num_threads())

    # torch.backends.cudnn.enabled == True

    down_factor = (model_para.encoder_stride *
                   model_para.encoder_pooling) ** len(model_para.encoder_channels)

    length, enc_length = args.length, args.length / down_factor

    enc_shape = (model_para.output_channels, enc_length)

    np.random.seed(2)
    torch.manual_seed(1)  # reproducible
    if cuda:
        torch.cuda.manual_seed(1)  # reproducible
    device = torch.device("cuda:0" if cuda else "cpu")

    #  Setup loggin, plotting and create data
    logging = logger.logger(args.p, args.name)

    train_loader, valid_loader, mean_period, mean_dm, mean_freq, example_shape, df_for_test, data_resolution = data_loader.create_loader(
        args.path, args.path_noise, args.samples, length, args.batch, args.edge, enc_shape=enc_shape, down_factor=down_factor,
        snr_range=args.snr_range, nulling=args.nulling, val_test=args.use_val_as_test, kfold=args.kfold,
        dmsplit=args.dmsplit, net_out=model_para.output_channels, dm_range=args.dm_range, dm_overlap=args.dmoverlap,
        set_based=args.set_based, sim_prob=args.sim_prob, discard_labels=args.discard_labels)

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
        net = torch.load(
            './trained_models/{}'.format(args.model)).to(device)
        net.device = device
        net.set_mode(args.mode)
        example_shape = net.input_shape
        example_shape_altered = example_shape
        net.create_loss_func(args.class_weight)

        if args.overwrite_length:
            new_length = args.length
            net.input_shape = (net.input_shape[0], new_length)
            net.create_loss_func()
            net.to(device)

            # Adding new classifiers does not curently
        if args.overwrite_classifier or args.add_classifier:
            net.create_classifier_levels(args.class_configs, no_reg=args.no_reg,
                                         overwrite=args.overwrite_classifier, dm0_class=args.dm0_class)
            net.to(device)

        net.reset_optimizer(args.l, decay=args.decay,
                            freeze=args.freeze, init=1)
        train_loader.dataset.length = net.input_shape[1]
        valid_loader.dataset.length = net.input_shape[1]
        net.out_length = net.input_shape[1] // net.down_fac - net.crop * 2
        if test_loader is not None:
            test_loader.dataset.length = net.input_shape[1]

        if args.reset_pre:
            net.set_preprocess(net.input_shape, model_para.initial_norm,
                               bias=args.clamp[0], clamp=args.clamp[1:], cmask=args.cmask, rfimask=args.rfimask, groups=model_para.initial_norm_groups)
            net.to(device)

        # This section allowed the changing of dropout values in later training steps but is not supported currently
        # for m in net.modules():
        #     if 'Conv' in str(type(m)):
        #         setattr(m, 'padding_mode', 'zeros')
        # for child in net.encoder.modules():
        #     if isinstance(child, nn.Dropout):
        #         child.p = args.d[0]
        # if hasattr(net, 'tcn'):
        #     for child in net.tcn.modules():
        #         if isinstance(child, nn.Dropout):
        #             child.p = args.d[1]
        # for child in net.output_layer.modules():
        #     if isinstance(child, nn.Dropout2d):
        #         child.p = args.d[2]
        # for (child_name, child) in net.named_modules():
        #     if child_name.startswith('classifier_stft'):
        #         if hasattr(child, 'rnn'):
        #             child.rnn.dropout = args.d[4]
        #         if hasattr(child, 'ini_dropout'):
        #             child.ini_dropout.p = args.d[3]
        #         # for chil in child.children():
        #         #     if isinstance(chil, nn.LSTM):
        #         #         chil.dropout = args.d[3]
        #         for chil in child.modules():
        #             if isinstance(chil, nn.Dropout2d):
        #                 child.p = args.d[4]

        if args.noise[3] == 1:
            args.noise = net.noise
    else:
        net = pulsar_net(model_para, example_shape, args.l, mode=args.mode,
                         clamp=args.clamp, gauss=args.gauss,
                         cmask=args.cmask, rfimask=args.rfimask, dm0_class=args.dm0_class,
                         class_configs=args.class_configs, data_resolution=data_resolution,
                         crop=args.crop, edge=args.edge, class_weight=args.class_weight,
                         added_cands=args.added_cands, psr_cands=args.psr_cands,
                         cands_threshold=args.cands_threshold).to(device)
        net.edge = train_loader.dataset.edge
        net.device = device
        net.reset_optimizer(args.l, decay=args.decay,
                            freeze=args.freeze, init=1)

    net.create_loss_func()
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
                                device, args.noise, args.threshold, args.l,
                                loss_weights=args.loss_weights, train_single=args.train_single,
                                test_frac=args.test_samples[
                                    1],
                                acc_grad=args.acc_grad,
                                loss_pool_mse=args.loss_pool_mse,
                                relabel_set=args.relabel_set,
                                relabel_thresholds=args.relabel_thresholds)

    command_string = 'python ' + ' '.join(sys.argv[:])

    train_net.logger.log_command(command_string)

    class_ok = 0

    for epoch in range(args.e):
        freeze_val = args.freeze - epoch
        train_net.logger.time_meter.reset()
        train_net.net.save_epoch(epoch)
        train_net.logger.epoch = epoch
        loss_train = train_net.run(
            'train', args.loops, only_class=args.no_reg, print_progress=args.progress,
            reverse_batch=args.reverse_batch)
        if args.relabel_set_slow and epoch > 0 and epoch % 4==0:
            train_net.label_set('train', print_progress=args.progress)
        reg_loss_train = train_net.logger.loss_meter.value()[0]
        clas_loss_train = train_net.logger.loss_meter_2.value()[0]
        im_loss_train = train_net.logger.loss_meter_3.value()[0]
        loss_valid = train_net.run(
            'validation', args.loops, only_class=args.no_reg, print_progress=args.progress)
        reg_loss = train_net.logger.loss_meter.value()[0]
        clas_loss = train_net.logger.loss_meter_2.value()[0]
        im_loss = train_net.logger.loss_meter_3.value()[0]
        if test_loader is not None:
            loss_test = train_net.run(
                'test', 1, only_class=1, print_progress=args.progress)
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
            epoch, args.name, decay=args.decay, freeze=freeze_val)
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
        if args.relabel_set or args.relabel_set_slow:
            if train_net.valid_loader is not None:
                val_df = train_net.valid_loader.dataset.noise_df
            else:
                val_df = None
            train_net.logger.log_relabel(train_net.train_loader.dataset.noise_df, val_df,
                args.name)

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
