import torch
from torch import nn
from torch import optim
from model.model_preprocessing import Preprocess
from model.model_encoder import pulsar_encoder
from model.model_tcn import TemporalConvNet
from model.model_tcn_multi import TemporalConvNet_multi
from model.model_output import OutputLayer
from model.model_multiclass import MultiClass
from model.model_classifier import classifier
from model.model_regressor_ffa import regressor_ffa
from model.model_regressor_stft import regressor_stft_comb
from model.model_classifier_filter import classifier_filter
from torch.utils.checkpoint import checkpoint
import numpy as np
import sys
import torch.nn.functional as F
import scipy.signal
import matplotlib.pyplot as plt
import argparse
import json


class pulsar_net(nn.Module):
    # Whole pulsar net. By default it contains an autoencoder as well as as a regressor
    def __init__(self, model_para, input_shape, output_size, list_channels_conv, kernel_size, lr, dropout, pool=1, stride=2,
                 binary=0, list_channels_lin=[128, 32], rnn=[0, 0], groups=4, bidirectional=False,
                 residual=True, no_pad=False, tcn_kernel=5, tcn_layers=5, tcn_dilation=[40, 10, 5],
                 mode='full', tcn_class=[0], acf_class=[0],
                 tcn_channels=0, dec_channels=[0], add_chan=0, no_reg=0, rnn_dec_args=[[0], 0, 0, 0], bce_weight=1, crop=0,
                 fft_class=[0], simple_class=0, norm=False, block_mode='add', reduce_mode='avg', tcn_mode='tcn_multi',
                 filter_size=0, clamp=[65, -10, 10], dec_mode='conv', class_mode='simple', multi_class=False,
                 gauss=(27, 15 / 4, 1, 1), enc_mode='conv', pool_multi=4, aa=False, stft=[10000], dm0='none',
                 cmask=False, rfimask=False, crop_augment=0., ffa=[5, 2, 8, 10], ffa_args=[0.07, 1.1, 15, 80, 1],
                 dm0_class=False, class_configs=[''], data_resolution=1):
        super().__init__()

        print('Creating neural net.')

        self.model_para = model_para

        self.set_mode(mode)

        self.input_shape = input_shape
        self.encoder_channels = list_channels_conv

        self.crop = crop

        if dec_channels[0]:
            self.dec_channels = dec_channels
        else:
            self.dec_channels = list_channels_conv[::-1]
        self.stride = model_para.encoder_stride
        self.pool = model_para.encoder_pooling

        self.kernel_encoder = kernel_size[0]
        self.kernel_decoder = kernel_size[-1]

        self.no_pad = no_pad
        self.binary = binary
        self.tcn_class = tcn_class
        self.down_fac = (
            self.stride * self.pool) ** len(model_para.encoder_channels)

        self.int_chan = self.input_shape[0] + add_chan
        self.input_shape_2 = (self.int_chan, self.input_shape[1])

        self.output_chan = model_para.output_channels

        self.out_length = self.input_shape[1] // self.down_fac - self.crop * 2

        self.data_resolution = data_resolution
        self.output_resolution = data_resolution * self.down_fac

        self.set_preprocess(self.input_shape, norm,
                            filter_size, bias=clamp[0], clamp=clamp[1:], dm0=model_para.subtract_dm0,
                            groups=groups, cmask=cmask, rfimask=rfimask)

        # self.preprocess = Preprocess(self.input_shape, norm, filter_size)
        self.tcn_mode = tcn_mode

        if list_channels_conv[0] != 0:
            if not model_para.concat_dm0:
                input_encoder = self.input_shape
            else:
                input_encoder = (self.input_shape[0] + 1, self.input_shape[1])
            self.encoder = pulsar_encoder(input_encoder, model_para,
                                          no_pad=no_pad)
            tcn_input = list_channels_conv[-1]
        else:
            tcn_input = self.input_shape[0]

        if model_para.tcn_2_layers:
            self.use_tcn = 1
            self.tcn = TemporalConvNet_multi(model_para.tcn_2_channels, model_para.tcn_2_channels_increase,
                                             model_para.tcn_2_layers, model_para.tcn_2_groups, dilation=model_para.tcn_2_dilation,
                                             levels=model_para.tcn_2_downsample_levels,
                                             downsample_factor=model_para.tcn_2_downsample_factor)
            dec_input = self.tcn.output_chan

        else:
            self.use_tcn = 0
            dec_input = tcn_channels[0]

        self.dec_input = dec_input
        self.use_output_layer = 1

        self.output_layer = OutputLayer(
            dec_input, model_para.output_intermediate_channels, model_para.output_final_nonlin,
            dropout=model_para.output_dropout, kernel=model_para.output_kernel,
            output_channels=self.output_chan)

        rnn_input = tcn_channels[-1]

        self.create_classifier_levels(class_configs, class_mode, multi_class, no_reg, fft_class, acf_class, rnn, stft, dropout=dropout[3:],
                                      crop_augment=crop_augment, tcn_class=tcn_class, ffa_class=ffa, ffa_args=ffa_args, dm0_class=dm0_class)

        self.create_loss_func()

        self.enc_layer = len(self.encoder_channels)
        self.dec_layer = 0
        self.freeze = 0
        # (int(111 / self.down_fac), int(15 / self.down_fac))
        self.gauss_para = gauss

        if self.gauss_para[0] % 2 == 0:
            self.gauss_para[0] += 1

        self.gaussian_kernel = torch.Tensor(scipy.signal.gaussian(
            *self.gauss_para[:2])).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0) * self.gauss_para[2]

        self.gaussian_kernel = torch.clamp(
            self.gaussian_kernel, 0, self.gauss_para[3])

        # self.crop = self.tcn.biggest_pad

    def create_classifier_levels(self, class_configs, class_mode, multi_class, no_reg, fft_class, acf_class, rnn, stft, bidirectional=True, dropout=[0, 0],
                                 crop_augment=0., tcn_class=[0], ffa_class=[5, 2, 8, 10], overwrite=True, ffa_args=[0.07, 1.1, 15, 80, 1], dm0_class=False):
        self.class_mode = class_mode
        self.dm0_class = dm0_class
        # if hasattr(self, 'classifiers'):
        #     for classifier in self.classifiers:
        #         print(classifier)
        #         del classifier
        if overwrite:
            if hasattr(self, 'classifiers'):
                del self.classifiers
                del_list = []
                for (child_name, child) in self.named_modules():
                    if child_name.startswith('classifier'):
                        del_list.append(child_name)
                for del_element in del_list:
                    try:
                        self.__delattr__(del_element)
                    except AttributeError:
                        pass

            self.classifiers = []
            added = ''
        else:
            added = '_1'
            for single_classifier in self.classifiers:
                single_classifier.dm0_class = dm0_class
        self.no_reg = no_reg

        # if not hasattr(self, 'out_length'):
        self.out_length = self.input_shape[1] // self.down_fac - self.crop * 2

        if self.no_reg:
            self.final_output = 3
        else:
            self.final_output = 3

        if not hasattr(self, 'output_chan'):
            self.output_chan = 1

        for config in class_configs:
            with open(f"./model_configs/{config}") as json_data_file:
                class_para_dict = json.load(json_data_file)
            class_para = argparse.Namespace(**class_para_dict)

            # if 'ffa' in self.class_mode:
            if class_para.class_type == 'ffa':
                setattr(self, "classifier_ffa%s" % added, regressor_ffa(self.output_resolution, no_reg=True, dm0_class=False,
                                                                        pooling=class_para.pooling, nn_layers=class_para.nn_layers, channels=class_para.channels,
                                                                        kernel=class_para.kernel, norm=class_para.norm, use_ampl=class_para.only_use_amplitude,
                                                                        min_period=class_para.min_period, max_period=class_para.max_period, bins_min=class_para.bins_min,
                                                                        bins_max=class_para.bins_max,
                                                                        remove_threshold=class_para.remove_dynamic_threshold))
                self.classifiers.append(
                    getattr(self, "classifier_ffa%s" % added))

            # if 'stft_comb' in self.class_mode:
            if class_para.class_type == 'stft':
                setattr(self, f"classifier_{class_para.name}", regressor_stft_comb(self.out_length, self.output_resolution, height_dropout=class_para.height_dropout, norm=class_para.norm,
                                                                                   harmonics=class_para.harmonics, nn_layers=class_para.nn_layers, stft_count=class_para.stft_count,
                                                                                   dm0_class=dm0_class, crop_factor=class_para.crop_factor, channels=class_para.channels,
                                                                                   kernel=class_para.kernel))
                self.classifiers.append(
                    getattr(self, f"classifier_{class_para.name}"))
        # else:
        #     self.classifier = None

        self.used_classifiers = len(self.classifiers)

        if multi_class or self.used_classifiers > 1:
            self.use_multi_class = 1
            self.multi_class = MultiClass(self.used_classifiers, self.no_reg)
        else:
            self.use_multi_class = 0

    def forward(self, x):
        # y = x - tile(self.pool(x)[:,:,:], 2, 1000)
        # x.requires_grad=False
        # return checkpoint(self.apply_net, x)
        return self.apply_net(x)

    def save_epoch(self, epoch):
        self.epoch = epoch

    def save_noise(self, noise):
        self.noise = noise

    def save_mean_vals(self, mean_period, mean_dm, mean_freq):
        self.mean_vals = (mean_period, mean_dm, mean_freq)

    def set_mode(self, mode):
        self.mode = mode
        if mode != 'autoencoder' and mode != 'full' and mode != 'classifier' and mode != 'short':
            print('Unkown mode!')
            sys.exit()

    def calc_tcn_out(self, x):

        chunks = self.net_chunks[0]
        overlap = self.net_chunks[1]
        split_val = np.linspace(0, x.shape[2], chunks + 1, dtype=int)
        output_tensor = torch.zeros(
            x.shape[0], self.dec_input, x.shape[2] // self.down_fac).cuda()
        for chunk in range(chunks):
            ini_start_val = split_val[chunk]
            ini_end_val = split_val[chunk + 1]
            start_val = np.max((ini_start_val - overlap, 0))
            end_val = np.min((ini_end_val + overlap, x.shape[2]))
            actual_start_overlap = (ini_start_val - start_val) // self.down_fac
            actual_end_value = (actual_start_overlap +
                                ini_end_val - ini_start_val) // self.down_fac
            out_start = split_val[chunk] // self.down_fac
            out_end = split_val[chunk + 1] // self.down_fac
            # output_tensor[:,:,out_start:out_end] = self.calc_tcn_chunk(
            #     x[:,:,start_val:end_val])[:,:,actual_start_overlap:actual_end_value]
            output_tensor[:, :, out_start:out_end] = checkpoint(self.calc_tcn_chunk,
                                                                x[:, :, start_val:end_val])[:, :, actual_start_overlap:actual_end_value]
        return output_tensor

    def calc_tcn_chunk(self, x):
        if hasattr(self, 'encoder'):
            y = self.encoder(x)
        else:
            y = x
        if self.use_tcn:
            out = self.tcn(y)
        else:
            out = y
        return out

    def apply_net(self, input):
        # input = input.permute(0,2,1)
        # input = F.layer_norm(input, [input.shape[2]]).permute(0,2,1)
        # input = F.group_norm(input, 10).permute(0,2,1)
        # if self.use_norm:
        #     input = self.norm(input)

        # input = torch.clamp(input, -5,5)
        # input = input+0.1
        # means = F.avg_pool2d(input, (14, 4001), stride=(1,1), padding=(0,2000))
        # input -= means[:,0,:][:,None,:]

        input = self.preprocess(input)
        # if hasattr(self, 'encoder'):
        #     input = self.encoder(input)
        encoded = self.calc_tcn_out(input)
        if self.crop:
            encoded = encoded[:, :, self.crop:-self.crop].contiguous()

        class_tensor = torch.zeros(
            (input.shape[0], self.used_classifiers, self.final_output)).cuda()
        # switch = 0
        j = 0
        encoded = self.output_layer(encoded)
        if self.mode == 'autoencoder':
            return encoded, torch.empty(0, requires_grad=True), torch.empty(0, requires_grad=True)

        if hasattr(self, 'break_grad'):
            if self.break_grad:
                # if self.train:
                encoded_ = encoded.detach()
            else:
                encoded_ = encoded
        else:
            encoded_ = encoded
        if hasattr(self, 'dm0_class'):
            if self.dm0_class:
                encoded_ = self.append_dm0(input, encoded_)
        for classifier in self.classifiers:
            # print(class_tensor.shape)
            class_tensor[:, j, :] = classifier(encoded_)
            j += 1
        if self.use_multi_class:
            classifier_output_multi = self.multi_class(
                class_tensor)
            return encoded, classifier_output_multi, class_tensor
        return encoded, class_tensor[:, 0, :], torch.empty(0, requires_grad=True)

    # def calc_encoded_length(self):
    #     # Calculates the size of the encoded state
    #     factor = self.stride2 * self.pool2
    #     channels = np.asarray(self.encoder_channels2)
    #     block_layers = len(channels[channels > 0])
    #     pool_layers = len(channels[channels < 0])
    #     whole_layers = block_layers * 2 + pool_layers
    #     out_size = (self.input_shape[1] / (factor **
    #                                        whole_layers) + 6) * abs(channels[-1])
    #     return out_size

    def apply_classifier(self, input):
        class_tensor = torch.zeros(
            (input.shape[0], self.used_classifiers, self.final_output)).cuda()
        # switch = 0
        j = 0
        # if self.output_layer.use_direct_class:
        #     input, class_tensor[:, j, :] = self.output_layer(input)
        #     # switch = 1
        #     j += 1
        #     if not self.use_multi_class:
        #         return input, class_tensor, torch.empty(0, requires_grad=True)
        # else:
        #     input = self.output_layer(input)
        # if self.mode == 'autoencoder':
        #     return input, torch.empty(0, requires_grad=True), torch.empty(0, requires_grad=True)
        for classifier in self.classifiers:
            class_tensor[:, j, :] = classifier(input)
            j += 1
        if self.use_multi_class:
            classifier_output_multi = self.multi_class(
                class_tensor)
            return input, classifier_output_multi, class_tensor
        return input, class_tensor[:, 0, :], torch.empty(0, requires_grad=True)

    def reset_optimizer(self, lr, decay=0, freeze=0, init=0, ada=False):

        # self.freeze = freeze
        if init:
            learn_rate_1 = lr[0]
        else:
            learn_rate_1 = lr[1]

        learn_rate_2 = learn_rate_1 * lr[2]
        print(learn_rate_2)

        # encoder_params = list(self.encoder.network[freeze:].parameters())
        # if self.use_tcn:
        #     encoder_params += list(self.tcn.parameters())
        #     # print(encoder_params)
        # if freeze == 0:
        #     second_params = self.decoder.parameters()
        # else:
        #     second_params = self.decoder.network[:-freeze].parameters()

        # self.optimizer = optim.Adam([{'params': encoder_params, 'lr': learn_rate_2},
        #                              {'params': second_params, 'lr': learn_rate_1}], lr=learn_rate_1, weight_decay=decay)  #, amsgrad=True

        if freeze <= 0:
            # parameters = self.parameters()
            class_params = list()
            encoder_params = list()
            if self.use_multi_class:
                class_params += list(self.multi_class.parameters())
            for classifier in self.classifiers:
                class_params += list(classifier.parameters())
            encoder_params += list(self.preprocess.parameters())
            encoder_params += list(self.encoder.parameters())
            if hasattr(self, 'tcn'):
                encoder_params += list(self.tcn.parameters())
            encoder_params += list(self.output_layer.parameters())
            self.frozen = 0

            # for para in self.parameters():
            #     print(para)
            #     if (para not in class_params):
            #         encoder_params += list(para)

        else:
            print('using freeze')
            parameters = list()
            if self.use_multi_class:
                parameters += list(self.multi_class.parameters())
            for classifier in self.classifiers:
                parameters += list(classifier.parameters())
                self.frozen = 1

        if not ada:
            if freeze <= 0:
                self.optimizer = optim.Adam([{'params': encoder_params, 'lr': learn_rate_2},
                                             {'params': class_params, 'lr': learn_rate_1}], lr=learn_rate_1, weight_decay=decay)
            else:
                self.optimizer = optim.Adam(
                    parameters, lr=learn_rate_1, weight_decay=decay)
        else:
            print('AdaBound not used currently.')
            # self.optimizer = adabound.AdaBound(
            #     parameters, lr=learn_rate_1, final_lr=0.1, weight_decay=decay)

        min_lr = learn_rate_1 / 100

        # print(parameters)
        # print(self.optimizer)

        # if lr[2] == 0:
        #     min_lr = learn_rate_1 / 100
        # else:
        #     min_lr = min(learn_rate_1, learn_rate_2) / 100
        # else:
        #     self.optimizer = optim.Adam(self.parameters(), lr=lr[0])

        # self.scheduler = CosineAnnealingLRW(
        #     self.optimizer, T_max=5, cycle_mult=4, eta_min=min_lr, last_epoch=-1)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=3, factor=0.5)

    def set_layer(self, layer):
        self.enc_layer = layer
        self.dec_layer = len(self.encoder_channels) - layer

    def create_loss_func(self, focal=0):
        # if not self.binary:
        #     self.loss_autoenc = nn.MSELoss()
        # else:
        # self.loss_autoenc = nn.BCEWithLogitsLoss(pos_weight=torch.full((3200,1), 20))
            # self.loss_autoenc = nn.BCEWithLogitsLoss()
        # self.loss_autoenc = nn.L1Loss()
        # self.loss_autoenc = nn.BCEWithLogitsLoss()
        # out_length = int(self.input_shape[1]/4 -self.crop*2)
        # # self.loss_autoenc = nn.BCEWithLogitsLoss(pos_weight=torch.full((1, out_length), self.bce_weight))
        self.loss_autoenc = nn.MSELoss()
        # self.loss_autoenc = nn.BCELoss()

        self.loss_1 = nn.MSELoss(reduction='sum')
        self.loss_2 = nn.CrossEntropyLoss()

    def gauss_smooth(self, tensor):
        pad = int((self.gaussian_kernel.shape[-1] - 1) / 2)
        if len(self.gaussian_kernel.shape) == 3:
            self.gaussian_kernel = self.gaussian_kernel.unsqueeze(1)
        # smoothed = F.conv1d(F.relu(tensor[:,:1,:]), self.gaussian_kernel,padding=pad)
        smoothed = F.conv2d(
            tensor[:, :, :].unsqueeze(1), self.gaussian_kernel, padding=(0, pad))
        #print(smoothed.shape, tensor.shape)
        # out_tensor = torch.cat((
        #     smoothed, tensor[:, 1:, :]), dim=1)
        # plt.plot(tensor[0,0,:].detach().cpu().numpy())
        # plt.plot(smoothed[0,0,:].detach().cpu().numpy())
        # plt.show()
        return smoothed[:, 0, :, :]

    def set_preprocess(self, input_shape, norm, filter_size, bias=65, clamp=[-10, 10], dm0='none',
                       groups=[1, 1, 1, 1], cmask=False, rfimask=False):
        self.preprocess = Preprocess(
            input_shape, norm, filter_size, bias=bias, clamp=clamp, dm0=dm0,
            groups=groups[2], cmask=cmask, rfimask=rfimask)

    def append_dm0(self, ini_fil, out_dedis, down_fac=4):
        dm0_series = F.avg_pool2d(ini_fil.unsqueeze(
            1), (ini_fil.shape[1], 4), 4, padding=0)[:, 0, :, :]
        new_out = torch.cat((out_dedis, dm0_series), dim=1)
        return new_out
