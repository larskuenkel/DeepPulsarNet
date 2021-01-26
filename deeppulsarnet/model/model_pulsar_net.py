import torch
from torch import nn
from torch import optim
from model.model_preprocessing import Preprocess
from model.model_encoder import pulsar_encoder
from model.model_tcn_multi import TemporalConvNet_multi
from model.model_output import OutputLayer
from model.model_multiclass import MultiClass
try:
    from model.model_classifier_ffa import classifier_ffa
except ImportError:
    print("FFA classifier not imported properly")
    pass
from model.model_classifier_stft import classifier_stft
from model.model_candidate_creator import candidate_creator
from torch.utils.checkpoint import checkpoint
import numpy as np
import sys
import torch.nn.functional as F
import scipy.signal
import matplotlib.pyplot as plt
import argparse
import json

from data_loader import dataset


class pulsar_net(nn.Module):
    # Whole pulsar net. By default it contains an classifier as well as as a classifier
    def __init__(self, model_para, input_shape, lr, no_pad=False,
                 mode='full', no_reg=0, clamp=[0, -1000, 1000],
                 gauss=(27, 15 / 4, 1, 1),
                 cmask=False, rfimask=False, 
                 dm0_class=False, class_configs=[''], data_resolution=1, crop=0,
                 edge=[0,0], class_weight=[1,1], added_cands=0, psr_cands=False,
                 cands_threshold=0):
        super().__init__()

        print('Creating neural net.')

        self.model_para = model_para

        self.set_mode(mode)

        self.input_shape = input_shape

        self.stride = model_para.encoder_stride
        self.pool = model_para.encoder_pooling

        self.no_pad = no_pad

        self.crop = crop

        self.edge = edge

        self.down_fac = (
            self.stride * self.pool) ** len(model_para.encoder_channels)


        self.output_chan = model_para.output_channels

        self.out_length = self.input_shape[1] // self.down_fac - self.crop * 2

        self.data_resolution = data_resolution
        self.output_resolution = data_resolution * self.down_fac

        self.added_cands = added_cands
        self.psr_cands = psr_cands
        self.cands_threshold = 0

        if self.added_cands or self.psr_cands:
            self.candidate_creator = candidate_creator(added_cands=self.added_cands, psr_cands=self.psr_cands, 
                candidate_threshold=self.cands_threshold)
            self.cand_based = True
        else:
            self.cand_based = False

        self.set_preprocess(self.input_shape, model_para.initial_norm,
                            bias=clamp[0], clamp=clamp[1:], dm0=model_para.subtract_dm0,
                            groups=model_para.initial_norm_groups, cmask=cmask, rfimask=rfimask)


        if not model_para.concat_dm0:
            input_encoder = self.input_shape
        else:
            input_encoder = (self.input_shape[0] + 1, self.input_shape[1])
        self.encoder = pulsar_encoder(input_encoder, model_para,
                                      no_pad=no_pad)

        if model_para.tcn_2_layers:
            self.use_tcn = 1
            self.tcn = TemporalConvNet_multi(model_para.tcn_2_channels, model_para.tcn_2_channels_increase,
                                             model_para.tcn_2_layers, model_para.tcn_2_groups, dilation=model_para.tcn_2_dilation,
                                             levels=model_para.tcn_2_downsample_levels,
                                             downsample_factor=model_para.tcn_2_downsample_factor)
            dec_input = self.tcn.output_chan

        else:
            self.use_tcn = 0
            dec_input = model_para.tcn_2_channels

        self.dec_input = dec_input
        self.use_output_layer = 1

        self.output_layer = OutputLayer(
            dec_input, model_para.output_intermediate_channels, model_para.output_final_nonlin,
            dropout=model_para.output_dropout, kernel=model_para.output_kernel,
            output_channels=self.output_chan)


        self.create_classifier_levels(class_configs, no_reg, dm0_class=dm0_class)

        self.create_loss_func(class_weight)

        self.freeze = 0
        # (int(111 / self.down_fac), int(15 / self.down_fac))
        self.gauss_para = gauss

        if self.gauss_para[0] % 2 == 0:
            self.gauss_para[0] += 1

        self.gaussian_kernel = torch.Tensor(scipy.signal.gaussian(
            *self.gauss_para[:2])).unsqueeze(0).unsqueeze(0).unsqueeze(0) * self.gauss_para[2]

        self.gaussian_kernel = torch.clamp(
            self.gaussian_kernel, 0, self.gauss_para[3])

        # self.crop = self.tcn.biggest_pad

    def create_classifier_levels(self, class_configs, no_reg=True, overwrite=True, dm0_class=False):
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
                class_name = f"classifier_ffa{added}"
                while hasattr(self, class_name):
                    class_name += '_'
                setattr(self, class_name, classifier_ffa(self.output_resolution, no_reg=True, dm0_class=False,
                                                                        pooling=class_para.pooling, nn_layers=class_para.nn_layers, channels=class_para.channels,
                                                                        kernel=class_para.kernel, norm=class_para.norm, use_ampl=class_para.only_use_amplitude,
                                                                        min_period=class_para.min_period, max_period=class_para.max_period, bins_min=class_para.bins_min,
                                                                        bins_max=class_para.bins_max,
                                                                        remove_threshold=class_para.remove_dynamic_threshold,
                                                                        name=f"classifier_ffa{added}"))
                self.classifiers.append(
                    getattr(self, "classifier_ffa%s" % added))

            # if 'stft_comb' in self.class_mode:
            if class_para.class_type == 'stft':
                class_name = f"classifier_{class_para.name}"
                while hasattr(self, class_name):
                    class_name += '_'
                setattr(self, class_name, classifier_stft(self.out_length, self.output_resolution, height_dropout=class_para.height_dropout, norm=class_para.norm,
                                                                                   harmonics=class_para.harmonics, nn_layers=class_para.nn_layers, stft_count=class_para.stft_count,
                                                                                   dm0_class=dm0_class, crop_factor=class_para.crop_factor, channels=class_para.channels,
                                                                                   kernel=class_para.kernel,
                                                                                   name=f"classifier_{class_para.name}", harmonic_downsample=class_para.harmonic_downsample,
                                                                                   train_harmonic=class_para.train_harmonic))
                self.classifiers.append(
                    getattr(self, f"classifier_{class_para.name}"))
        # else:
        #     self.classifier = None

        self.used_classifiers = len(self.classifiers)

        if self.used_classifiers > 1:
            self.use_multi_class = 1
            self.multi_class = MultiClass(self.used_classifiers, self.no_reg)
        else:
            self.use_multi_class = 0

    def forward(self, x, target=None):
        # y = x - tile(self.pool(x)[:,:,:], 2, 1000)
        # x.requires_grad=False
        # return checkpoint(self.apply_net, x)
        return self.apply_net(x, target)

    def save_epoch(self, epoch):
        self.epoch = epoch

    def save_noise(self, noise):
        self.noise = noise

    def save_mean_vals(self, mean_period, mean_dm, mean_freq):
        self.mean_vals = (mean_period, mean_dm, mean_freq)

    def set_mode(self, mode):
        self.mode = mode
        if mode != 'dedisperse' and mode != 'full' and mode != 'classifier' and mode != 'short':
            print('Unkown mode!')
            sys.exit()

    def calc_tcn_out(self, x):

        chunks = self.net_chunks[0]
        overlap = self.net_chunks[1]
        split_val = np.linspace(0, x.shape[2], chunks + 1, dtype=int)
        output_tensor = torch.zeros(
            x.shape[0], self.dec_input, x.shape[2] // self.down_fac).to(x.device)
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

    def apply_net(self, input, target=None):

        if target is not None:
            target = target.to(input.device)
            if len(target.shape)==1:
                target = target.unsqueeze(0)
        input = self.preprocess(input)

        encoded = self.calc_tcn_out(input)
        if self.crop:
            encoded = encoded[:, :, self.crop:-self.crop].contiguous()

        class_tensor = torch.zeros(
            (input.shape[0], self.used_classifiers, self.final_output)).to(input.device)
        # switch = 0
        j = 0
        encoded = self.output_layer(encoded)
        if self.mode == 'dedisperse':
            return encoded, torch.empty(0, requires_grad=True), torch.empty(0, requires_grad=True), torch.empty(0, requires_grad=True)

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
            class_tensor[:, j, :], class_data = classifier(encoded_)
            if self.cand_based:
                class_candidates, class_targets = self.candidate_creator(class_data, classifier.final_cands, target)
                if j == 0:
                    candidates = class_candidates
                    cand_targets = class_targets
                else:
                    candidates = torch.cat((candidates, class_candidates), 0)
                    cand_targets = torch.cat((cand_targets, class_targets), 0)
            else:
                candidates = torch.empty(0, requires_grad=True)
                cand_targets = torch.empty(0, requires_grad=True)

            j += 1
        if self.use_multi_class:
            classifier_output_multi = self.multi_class(
                class_tensor)
            return encoded, classifier_output_multi, class_tensor, (candidates, cand_targets)
        return encoded, class_tensor[:, 0, :], torch.empty(0, requires_grad=True), (candidates, cand_targets)

    # def apply_classifier(self, input):
    #     class_tensor = torch.zeros(
    #         (input.shape[0], self.used_classifiers, self.final_output)).to(input.device)
    #     j = 0
    #     for classifier in self.classifiers:
    #         class_tensor[:, j, :] = classifier(input, target)
    #         j += 1
    #         if self.cand_based:
    #             pass
    #     if self.use_multi_class:
    #         classifier_output_multi = self.multi_class(
    #             class_tensor)
    #         return input, classifier_output_multi, class_tensor
    #     return input, class_tensor[:, 0, :], torch.empty(0, requires_grad=True)

    def reset_optimizer(self, lr, decay=0, freeze=0, init=0):

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
        if freeze <= 0:
            self.optimizer = optim.Adam([{'params': encoder_params, 'lr': learn_rate_2},
                                         {'params': class_params, 'lr': learn_rate_1}], lr=learn_rate_1, weight_decay=decay)
        else:
            self.optimizer = optim.Adam(
                parameters, lr=learn_rate_1, weight_decay=decay)
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


    def create_loss_func(self, class_weight=[1,1]):
        # if not self.binary:
        #     self.loss_autoenc = nn.MSELoss()
        # else:
        # self.loss_autoenc = nn.BCEWithLogitsLoss(pos_weight=torch.full((3200,1), 20))
            # self.loss_autoenc = nn.BCEWithLogitsLoss()
        # self.loss_autoenc = nn.L1Loss()
        # self.loss_autoenc = nn.BCEWithLogitsLoss()
        # out_length = int(self.input_shape[1]/4 -self.crop*2)
        # # self.loss_autoenc = nn.BCEWithLogitsLoss(pos_weight=torch.full((1, out_length), self.bce_weight))
        self.loss_autoenc = nn.MSELoss().to(next(self.parameters()).device)
        # self.loss_autoenc = nn.BCELoss()

        self.loss_1 = nn.MSELoss(reduction='sum').to(next(self.parameters()).device)
        self.loss_2 = nn.CrossEntropyLoss(weight=torch.Tensor(class_weight)).to(next(self.parameters()).device)

    def gauss_smooth(self, tensor):
        self.gaussian_kernel = self.gaussian_kernel.to(tensor.device)
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

    def set_preprocess(self, input_shape, norm, bias=65, clamp=[-10, 10], dm0='none',
                       groups=1, cmask=False, rfimask=False):
        self.preprocess = Preprocess(
            input_shape, norm, bias=bias, clamp=clamp, dm0=dm0,
            groups=groups, cmask=cmask, rfimask=rfimask)

    def append_dm0(self, ini_fil, out_dedis, down_fac=4):
        dm0_series = F.avg_pool2d(ini_fil.unsqueeze(
            1), (ini_fil.shape[1], 4), 4, padding=0)[:, 0, :, :]
        new_out = torch.cat((out_dedis, dm0_series), dim=1)
        return new_out

    def test_single_file(self, noise_file, target=None, file='', noise=[0,0,0], start_val=2000,
                         verbose=0, nulling=(0, 0, 0, 0, 0, 0, 0, 0)):

        if hasattr(self, 'edge'):
            edge = self.edge
        else:
            edge = [0,0]
        data, target_array = dataset.load_filterbank(
            file, self.input_shape[1], 0, noise=noise_file,
            edge=edge, noise_val=noise, start_val=start_val, nulling=nulling)

        device = next(self.parameters()).device
        data_tensor = torch.tensor(
            data, dtype=torch.float).unsqueeze(0).to(device)
        # data_tensor = self.noise_and_norm(data_tensor, 2)
        # target = self.smooth(torch.tensor(
        #     target_array, dtype=torch.float).unsqueeze(0).to(self.device))
        # plt.imshow(target[0,:,:], aspect='auto')
        # plt.show()
        output_image, output_reg, output_single, cand_data = self(data_tensor, target=target)
        # loss = self.calc_loss(output_image_mask, ten_y_mask,
        #                           output_classifier, ten_y2)
        output = output_image.squeeze()
        # print(output)
        # plt.imshow(output, aspect='auto')
        # plt.show()
        if verbose:
            out_vals = torch.nn.Softmax(dim=1)(output_reg[:,:2])
            print(out_vals)
            print(output_single)
        # return loss
        return output_image, output_reg, output_single, cand_data