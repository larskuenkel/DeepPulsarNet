from model import model_smooth
import torch
import numpy as np
from data_loader import dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
from contextlib import nullcontext
from torch.cuda.amp import autocast, GradScaler


class trainer():
    #  class to store training and data augmentation routines
    def __init__(self, net, train_loader, valid_loader, test_loader, logger, device, noise, threshold, lr, 
                 loss_weights_ini=(0.001, 1, 1, 1), loss_weights=(0.001, 1, 1, 1), train_single=True, fft_loss=False, acf_loss=False, reduce_test=True, test_frac=0.1, acc_grad=1,
                 loss_pool_mse=False, bandpass=False, relabel_set =False, relabel_thresholds=[0.95, 0.5], relabel_validation=False, reverse_weight=1,
                 amp=False):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.logger = logger
        self.net = net
        self.device = device
        self.bandpass = 0
        self.smooth = model_smooth.smooth().to(device)
        self.noise = noise
        self.threshold = threshold
        self.last_noise_update = -1
        self.lr = lr
        self.train_mode = 0
        self.crop = self.net.crop
        self.loss_weights = loss_weights
        self.loss_weights_ini = loss_weights_ini

        self.used_loss_weights = self.loss_weights_ini
        self.train_single = train_single
        # self.fft_scale = 0.0001
        self.acf_scale = 0.01
        self.fft_loss = fft_loss
        self.acf_loss = acf_loss
        self.reduce_test = reduce_test

        self.no_reg = net.no_reg

        self.test_frac = test_frac

        self.acc_grad = acc_grad
        self.class_ok = 0

        self.loss_pool_mse = loss_pool_mse

        self.relabel_set = relabel_set

        self.threshold_pos = relabel_thresholds[0]
        self.threshold_neg = relabel_thresholds[1]

        self.relabel_validation = relabel_validation

        self.reverse_weight = reverse_weight
        self.amp = amp
        if self.amp:
            self.scaler = GradScaler()


    def run(self, mode, loops, only_class=0, print_out=0, store_stats=False, print_progress=True, store_tseries=False, reverse_batch=False):
        # Runs the net over the training or validation set
        if mode == 'test':
            torch.manual_seed(0)
        else:
            torch.manual_seed(np.random.randint(10000))
        self.logger.reset_meters()
        self.set_mode(mode)
        self.net.optimizer.zero_grad()
        for step, (x, y, y2) in enumerate(self.loader):

            if print_progress:
                print(step * x.shape[0], end="\r")

            if mode == 'test' and len(x.shape)==4:
                single_test_batch = True
                x = x[0, :, :, :]
                new_y_shape = list(y2.shape)
                new_y_shape[0] = x.shape[0]
                y2 = y2.expand(new_y_shape)
            else:
                single_test_batch = False

            if reverse_batch and mode == 'train':
                batch_loops = 2
            else:
                batch_loops = 1

            for batch_loop in range(batch_loops):

                if batch_loop==0:
                    ten_x = x.to(self.device).float()
                    ten_y = y.to(self.device).float()
                    ten_y2 = y2.to(self.device).float()
                    ten_x.requires_grad = True
                else:
                    ten_x = ten_x.flip(2)
                    ten_y = torch.zeros_like(ten_y)
                    #ten_y2 = torch.zeros_like(ten_y2)
                    null_indices = [0,1,2]
                    ten_y2[:,null_indices] = 0

                # ten_x = x.to(self.device).float()

                # ten_y = y.to(self.device).float()
                # ten_y2 = y2.to(self.device).float()

                # if self.bandpass:
                #     ten_x = self.apply_bandpass(ten_x)


                # ten_x.requires_grad = True

                # self.net.ini_target = ten_y2
                with (autocast() if self.amp else nullcontext()):
                    output_image, output_classifier, output_single_class, candidate_data = self.net(
                        ten_x, target=ten_y2)  # net output
                    loss, periods = self.calc_loss(output_image, ten_y,
                                               output_classifier, ten_y2, only_class, single_class=output_single_class)

                if store_tseries:
                    torch.save(output_image, f'tseries_{int(ten_y2[0, 3])}.pt')

                if store_stats:
                    means = output_image[:, 0, :].mean(
                        dim=1).detach().cpu().numpy()
                    stds = output_image[:, 0, :].std(dim=1).detach().cpu().numpy()
                    maxs = output_image[:, 0, :].max(
                        dim=1)[0].detach().cpu().numpy()
                    combined_vals = np.asarray((means, stds, maxs)).T.tolist()

                    self.logger.stats.extend(combined_vals)

                if self.crop:
                    ten_y, output_image = self.crop_target_output(
                        self.crop, ten_y, output_image)


                # loss, periods = self.calc_loss(output_image, ten_y,
                #                                output_classifier, ten_y2, only_class, single_class=output_single_class)

                try:
                    if candidate_data[0].shape[0]>1:
                        cand_loss = self.calc_cand_loss(candidate_data)
                        loss = loss + cand_loss
                except:
                    pass

                loss /= np.sum(self.used_loss_weights)
                loss = loss / self.acc_grad

                if mode == 'test' and self.net.mode != 'dedisperse' and self.reduce_test and single_test_batch:
                    class_result_single = output_classifier[:,
                                                            0] - output_classifier[:, 1]
                    pulsar_count = (0 > class_result_single).sum().to(torch.float)
                    pulsar_fraction = pulsar_count / len(class_result_single)

                    class_result = torch.FloatTensor(
                        [self.test_frac, pulsar_fraction]).to(self.device)

                    if self.no_reg:
                        periods = torch.mean(periods).unsqueeze(0)
                        # dummy_weights = torch.ones_like(periods)
                    else:
                        class_soft_max = F.softmax(output_classifier[:, :2],dim=1)
                        most_secure = torch.argmax(class_soft_max[:, 1])
                        periods = output_classifier[most_secure, 2].unsqueeze(0)

                    ten_y2 = ten_y2[:1]
                    class_estimate = torch.cat(
                        (class_result, periods)).unsqueeze(0)
                    output_classifier = torch.cat(
                        (class_result, periods), dim=0).unsqueeze(0)

                else:

                    if self.net.mode == 'dedisperse':
                        dummy = torch.zeros((len(periods), 2)).to(self.device)
                        dummy[:, 0] = 1
                        class_estimate = torch.cat(
                            (periods.float(), dummy), dim=1)
                    else:
                        class_estimate = output_classifier
                if print_out:

                    print(ten_y2)
                    print(output_classifier)

                if self.mode == 'train':

                    if batch_loop==1:
                        loss *= self.reverse_weight

                    if not self.amp:
                        loss.backward(retain_graph=True)
                        if step % self.acc_grad == 0:

                            self.net.optimizer.step()  # apply gradients
                            self.net.optimizer.zero_grad()
                    else:
                        self.scaler.scale(loss).backward(retain_graph=True)
                        if step % self.acc_grad == 0:
                            self.scaler.step(self.net.optimizer)
                            self.scaler.update()
                            self.net.optimizer.zero_grad()
                # stack results for scatter plot
                cand_combined = torch.cat((candidate_data[0].detach().cpu(), candidate_data[1].detach().cpu()), dim=1).numpy()
                self.logger.stack_output(class_estimate.detach().cpu().numpy().tolist(),
                                         ten_y2.detach().cpu().numpy().tolist(),
                                         output_single_class.detach().cpu().numpy().tolist(),
                                         cand_combined.tolist())


                if not self.net.mode == 'dedisperse':
                    # if not self.mode == 'train':
                        # self.logger.stack_output(output_classifier.detach().cpu().numpy().tolist(),
                        #                          ten_y2.detach().cpu().numpy().tolist())
                    try:
                        self.logger.classerr.add(
                            output_classifier[:, :2].detach().cpu(), torch.fmod(ten_y2[:, 2],2).detach().cpu())
                        self.logger.confusion_meter.add(
                            output_classifier[:, :2].detach().cpu(), torch.fmod(ten_y2[:, 2],2).detach().cpu())
                        # print(np.array2string(self.logger.confusion_meter.value().flatten(), precision=3))
                    except ValueError:
                        print('error with loss meter')
                        pass
                    except IndexError:
                        if self.net.epoch == 0:
                            print(
                                '1 sample will not be included in the logged class errors.')

                if self.net.epoch > 0 and self.train_loader.dataset.set_based and self.relabel_set:
                    if batch_loop==0:
                        self.relabel_set_no_cand(output_classifier, ten_y2)
                    else:
                        self.relabel_set_no_cand(output_classifier, ten_y2, reverse=True)
                if self.mode =='test':
                    self.add_test_predictions(output_classifier, ten_y2)

        if self.net.mode == 'dedisperse':
            final_loss = self.logger.loss_meter_3.value()[0]
        else:
            if self.mode == 'test':
                final_loss = self.logger.loss_meter_2.value()[0]
            else:
                final_loss = self.logger.loss_meter_2.value()[0]
                # if not np.isnan(self.logger.loss_meter.value()[0]) or not only_class:
                #     final_loss += self.logger.loss_meter.value()[0]
                if not self.net.mode == 'classifier' and not self.net.mode == 'short':
                    final_loss += self.logger.loss_meter_3.value()[0]

        if self.logger.plot == 1:
            self.make_plots(output_image, output_classifier,
                            ten_x, ten_y, ten_y2, final_loss)
            # print(self.logger.loss_meter.value()[0] , self.logger.loss_meter.n , self.logger.loss_meter_2.value()[0])

        # if self.mode == 'validation' and self.logger.loss_meter_3.value()[0] < self.threshold[0]:
        #     self.train_mode = 1
        #     print('Changed loss weights.')
        if not self.net.mode == 'dedisperse':
            self.logger.conf_mat[mode] = np.copy(
                self.logger.confusion_meter.conf)

        # Split mcc into real pulsars and fake pulsars during training
        if mode=='train' or mode=='validation':
            if mode=='train':
                sim_string = 'train sim'
                real_string = 'train real'
            else:
                sim_string = 'valid sim'
                real_string = 'valid real'
            epoch_output = np.asarray(self.logger.out_stack[2:])
            epoch_target = np.asarray(self.logger.target_stack[2:])
            self.logger.confusion_meter.reset()
            sim_indices = ((epoch_target[:, 2] != 3)&(epoch_target[:, 2] != 5))!=0
            if any(sim_indices):
                self.logger.confusion_meter.add(
                                torch.Tensor(epoch_output[sim_indices, :2]), torch.fmod(torch.Tensor(epoch_target[sim_indices, 2]),2))
                self.logger.conf_mat_split[sim_string] = np.copy(
                    self.logger.confusion_meter.conf)
            self.logger.confusion_meter.reset()
            real_indices = (epoch_target[:, 2] != 1)!=0
            if any(real_indices):
                self.logger.confusion_meter.add(
                                torch.Tensor(epoch_output[real_indices, :2]), np.fmod(torch.Tensor(epoch_target[real_indices, 2]),2))
                self.logger.conf_mat_split[real_string] = np.copy(
                    self.logger.confusion_meter.conf)

        return final_loss

    def set_mode(self, mode):
        # Chnage between training and validation mode
        self.mode = mode
        self.logger.out_stack = [[1, 0, 0,0,0,0,0], [0, 1, 0, 0,0,0,0]]
        self.logger.out_single_stack = []
        self.logger.target_stack = [[0, 0, 0, -1, 0, 0,0,0,0], [0, 0, 1, -1, 0, 0,0,0,0]]
        self.logger.cand_stack = []
        # if not self.no_reg:
        #     self.logger.out_stack = [[0, 1, 0,0], [0, 0, 1,0]]
        #     self.logger.target_stack = [[0, 0, 0, -1,0], [0, 0, 1, -1,0]]
        # else:
        #     self.logger.out_stack = [[0, 1, 0], [0, 0, 1]]
        #     self.logger.target_stack = [[0, 0, 0, -1,0], [0, 0, 1, -1,0]]
        if mode == 'train':
            self.loader = self.train_loader
            self.net.train()
            if self.logger.plot:
                self.logger.loss_logger = self.logger.train_loss_logger
                self.logger.err_logger = self.logger.train_err_logger
                self.logger.confusion_logger = self.logger.train_conf_logger
        elif mode == 'validation':
            self.net.eval()
            self.loader = self.valid_loader
            if self.logger.plot:
                self.logger.loss_logger = self.logger.valid_loss_logger
                self.logger.err_logger = self.logger.valid_err_logger
                self.logger.confusion_logger = self.logger.valid_conf_logger
        elif mode == 'test':
            self.net.eval()
            self.loader = self.test_loader
            # self.logger.loss_logger = self.logger.valid_loss_logger
            if self.logger.plot:
                self.logger.err_logger = self.logger.test_err_logger
                self.logger.confusion_logger = self.logger.test_conf_logger
        else:
            print('Unknown mode.')

    def apply_bandpass(self, tensor):
        # Apply the bandpass to the input data
        bpass = torch.tensor(np.load('bandpass.npy')).to(self.device)
        tensor *= bpass[None, None, :]
        return tensor


    def update_noise(self, epoch, name, decay, patience=3, freeze=-1):
        # Update the noise when the loss was under the threshold for some time

        # if self.net.mode=='full' or self.net.mode=='classifier':
        #     if not self.class_ok == 1 and epoch % 1 == 0 and len(self.net.classifiers) > 1:
        #         self.class_ok = self.check_classifier()
        self.class_ok = 1

        if self.class_ok == 0:
            print('Resetting with frozen first part')
            self.net.reset_optimizer(
                self.lr, decay, 1)
        else:
            current_noise = self.noise[0]
            noise_factor = self.threshold[1] if len(self.threshold) > 1 else 0
            # freeze_factor_val = 2
            # freeze_factor = freeze_factor_val if self.net.freeze != 0 else 1
            current_threshold = (
                self.threshold[0] + self.noise[0] * noise_factor)
            if freeze == 0 or (freeze < 0 and self.net.frozen):
                self.net.reset_optimizer(
                    self.lr, decay, freeze)
            else:
                if epoch - self.last_noise_update >= patience:
                    if self.threshold[2] == 2:
                        # use mcc of validation simulations
                        test_vals = 1 - np.asarray(
                            self.logger.values[-patience:])[:, 8]
                    elif self.threshold[2] == 1:
                        # use clas loss
                        test_vals = np.asarray(
                            self.logger.values[-patience:])[:, 3]
                    else:
                        # use loss
                        test_vals = np.asarray(
                            self.logger.values[-patience:])[:, 2]

                    if np.all(test_vals < current_threshold):

                        if current_noise < self.noise[1]:
                            if name:
                                torch.save(self.net,
                                           "./trained_models/{}_last_update.pt".format(name))
                            else:
                                torch.save(
                                    self.net, './trained_models/last_model_last_update.pt')
                            self.noise[0] += self.noise[2]
                            new_threshold = (
                                self.threshold[0] + self.noise[0] * noise_factor)
                            print('Noise: {:.2f} - {:.2f} Threshold: {}'.format(
                                self.noise[0], np.min((self.noise[1],
                                                       self.noise[0] + self.noise[2] * 2)), new_threshold))
                            self.last_noise_update = epoch
                            self.net.reset_optimizer(
                                self.lr, decay, freeze)
                            self.net.save_noise(self.noise[:])
                            self.logger.reset_best_values()
                            self.used_loss_weights = self.loss_weights
                        # for child in self.net.modules():
                        #     if isinstance(child, torch.nn.BatchNorm1d):
                        #         # child.reset_running_stats()
                        #         pass

            self.train_loader.dataset.noise = self.noise[:]
            self.valid_loader.dataset.noise = self.noise[:]

    def make_plots(self, output_image, output_classifier, ten_x, ten_y, ten_y2, final_loss):
        if self.mode != 'test':
            self.logger.loss_logger.log(
                self.net.epoch, final_loss)

        if not self.net.mode == 'dedisperse':
            self.logger.err_logger.log(
                self.net.epoch, self.logger.classerr.value()[0])
            self.logger.confusion_logger.log(
                self.logger.confusion_meter.value())

        # and not self.net.no_reg:
        # and not self.net.mode == 'dedisperse':
        if self.mode != 'train' and self.net.epoch % 1 == 0:
            # self.logger.plot_regressor(self.mode)
            # self.logger.plot_estimate(self.mode, self.no_reg)
            pass
        if self.mode == 'validation' and self.net.epoch % 1 == 0 and not self.net.mode == 'short':
            in_array = ten_x.detach().cpu().numpy()  # .squeeze()
            # if self.binary:
            #     out_array = torch.sigmoid(
            #         output_image.detach()).cpu().numpy().squeeze()
            # else:
            # output_image = torch.sigmoid(output_image)#self.net.gauss_smooth(output_image)
            ten_y = self.net.gauss_smooth(ten_y)
            # ten_y = self.net.classifier_fft.compute_fft(
            #     ten_y).unsqueeze(1).detach()[:, 0, :]
            # output_image = self.net.classifier_fft.compute_fft(
            #     output_image).unsqueeze(1).detach()[:, 0, :]
            out_array = output_image.detach().cpu().numpy()
            target_array = ten_y.cpu().numpy()
            self.logger.plot_dedisperse(
                in_array, out_array, target_array, ten_y2)

    def test_target_file(self, file, noise, noise_file='/media/lkuenkel/Uni/data/palfa/new_processed_2/p2030_53831_43886_0165_G65.12-00.39.C_0.decim.fil', start_val=2000,
                         verbose=0, nulling=(0, 0, 0, 0, 0, 0, 0, 0)):
        data, target_array = dataset.load_filterbank(
            file, self.net.input_shape[1], 0, noise=noise_file,
            edge=self.train_loader.dataset.edge, noise_val=noise, start_val=start_val, nulling=nulling)
        data_tensor = torch.tensor(
            data, dtype=torch.float).unsqueeze(0).to(self.device)
        # data_tensor = self.noise_and_norm(data_tensor, 2)
        target = self.smooth(torch.tensor(
            target_array, dtype=torch.float).unsqueeze(0).to(self.device))
        # plt.imshow(target[0,:,:], aspect='auto')
        # plt.show()
        output_image, output_reg, output_single = self.net(data_tensor)
        # loss = self.calc_loss(output_image_mask, ten_y_mask,
        #                           output_classifier, ten_y2)
        output = output_image.squeeze()
        # print(output)
        # plt.imshow(output, aspect='auto')
        # plt.show()
        if verbose:
            out_vals = torch.nn.Softmax(dim=1)(output_reg[-2:])
            print(out_vals)
            print(output_single)
        # return loss
        return output_image, output_reg, output_single

    def check_classifier(self):
        im, clas, single_class = self.test_target_file(self.train_loader.dataset.data_files[0], [self.noise[0] / 2, self.noise[0] / 2],
                                                       noise_file=self.train_loader.dataset.noise_df['FileName'][0])
        single_soft = F.softmax(single_class[0, :, :2], dim=1)
        max_pred = torch.max(single_soft[:, 1])
        good_class = 0
        if max_pred > 0.95:
            for (class_out, class_used) in zip(single_soft[:, 1], self.net.classifiers):
                if class_out < 0.7:
                    if hasattr(class_used, 'ini_conv'):
                        class_used.ini_conv()
                        print(f'{class_used.__module__} has been reset')
                else:
                    good_class += 1

            print(f'{good_class} classifiers seem to work')
            if good_class == len(self.net.classifiers):
                return_val = 1
            else:
                return_val = 0
        else:
            return_val =-1
        return return_val

    def calc_loss(self, output_im, target_im, output_clas, target_clas, only_class=0, single_class=None):

        # reg_factor = self.used_loss_weights[0]
        clas_factor = self.used_loss_weights[0]
        autoenc_factor = self.used_loss_weights[1]
        single_weight = self.used_loss_weights[2]
        output_im = output_im[:, :target_im.shape[1], :]
        output_im_smooth = output_im  # self.net.gauss_smooth(output_im)
        #periods = self.estimate_period(output_im_smooth[:, :1, :])
        if self.net.mode != 'dedisperse':
            periods = output_clas[:, 2:]
        else:
            periods = torch.zeros((output_im.shape[0],1)).to(output_im.device)


        if not self.mode == 'test' and not self.net.mode == 'classifier' and not self.net.mode == 'short':

            #non_nan_batches = target_im[:,0,0] == target_im[:,0,0]
            non_nan_batches = ~torch.isnan(target_im[:,0,0])
            # print(non_nan_batches)
            target_im = target_im[non_nan_batches,:,:]
            output_im_smooth = output_im_smooth[non_nan_batches,:,:]

            if target_im.shape[0]>1:
                target_im_smooth = self.net.gauss_smooth(target_im)

                if self.loss_pool_mse:
                    max_val_out, max_pos_out = torch.max(output_im_smooth, dim=1, keepdim=True)
                    max_val_target, max_pos_target = torch.max(target_im_smooth, dim=1, keepdim=True)
                    # output_im_smooth = F.adaptive_max_pool2d(output_im_smooth.unsqueeze(1), (1, output_im.shape[2]))[:,0,:,:]
                    # target_im_smooth = F.adaptive_max_pool2d(target_im_smooth.unsqueeze(1), (1, output_im.shape[2]))[:,0,:,:]
                    loss_in_out = max_val_out
                    loss_in_target = max_val_target
                else:
                    loss_in_out = output_im_smooth
                    loss_in_target = target_im_smooth

                loss_whole_im = self.net.loss_autoenc(
                    loss_in_out, loss_in_target)

                loss_val = loss_whole_im.data.cpu().numpy()
                try:
                    self.logger.loss_meter_3.add(loss_val)
                except ValueError:
                    print('Error with loss meter.')

                loss_whole = loss_whole_im * autoenc_factor
            else:
                loss_whole = None

            # if self.fft_loss or self.acf_loss:
            #     loss_whole_im = loss_whole_im * \
            #         np.max((0, 1 - self.fft_loss - self.acf_loss))
            # if self.fft_loss:
            #     # print(self.fft_loss)
            #     output_fft = self.net.classifier_fft.compute_fft(
            #         output_im_smooth, harmonics=0)[:, 0, :]
            #     target_fft = self.net.classifier_fft.compute_fft(
            #         target_im, harmonics=0)[:, 0, :]
            #     loss_fft = self.net.loss_autoenc(
            #         output_fft, target_fft)
            #     loss_whole_im = loss_whole_im + loss_fft * self.fft_loss

            # if self.acf_loss:
            #     acf_padding = 500
            #     output_acf = self.calc_acf(output_im_smooth, padding=acf_padding)[
            #         0, :, :] * self.acf_scale
            #     target_acf = self.calc_acf(target_im_smooth, padding=acf_padding)[
            #         0, :, :] * self.acf_scale
            #     loss_acf = self.net.loss_autoenc(
            #         output_acf, target_acf)
            #     loss_whole_im = loss_whole_im + loss_acf * self.acf_loss
        else:
            loss_whole = None

        # or (self.net.mode == 'full' and self.net.epoch != 0):
        if self.net.mode != 'dedisperse':
            output_clas_2 = output_clas[:, :2]
            ten_y_2 = torch.fmod(target_clas[:, 2],2).long()

            loss_2 = self.net.loss_2(output_clas_2, ten_y_2)
            # print(loss_2)
            loss_val_2 = loss_2.data.cpu().numpy()
            weight_2 = len(output_clas_2)
            self.logger.loss_meter_2.add(loss_val_2, weight_2)

            if len(single_class) != 0 and self.train_single and not self.mode == 'test':
                for j in range(single_class.shape[1]):
                    single_out = single_class[:, j, :2]
                    loss_2_2 = self.net.loss_2(single_out, ten_y_2)
                    # print(loss_2_2)
                    loss_2 = loss_2_2 * single_weight + loss_2

            if loss_whole is not None:
                loss_whole += loss_2 * clas_factor
            else:
                loss_whole = loss_2 * clas_factor

            if not only_class:
                output_clas_1 = output_clas[:, 2][~torch.isnan(
                    target_clas[:, 0])].view(-1, 1)
                ten_y_1 = target_clas[:, 0][~torch.isnan(
                    target_clas[:, 0])].view(-1, 1)
                if len(output_clas_1) != 0:
                    # print(output_clas_1, ten_y_1)
                    loss_1 = self.net.loss_1(output_clas_1, ten_y_1)
                    weight_1 = len(output_clas_1)
                    loss_val_1 = loss_1.data.cpu().numpy()
                    try:
                        self.logger.loss_meter.add(loss_val_1, weight_1)
                    except ValueError:
                        print('Error with loss meter.')

                    # currently no single training for regression
                    # print(single_class.shape)
                    # single_reg = single_class[~torch.isnan(target_clas[:, -1]), :,2]

                    # if len(single_reg) != 0 and self.train_single and not self.mode == 'test':
                    #     for j in range(single_reg.shape[1]):
                    #         single_reg_single = single_reg[:, j:j+1]
                    #         loss_1_2 = self.net.loss_1(single_reg_single, ten_y_1)
                    #         # print(loss_2_2)
                    #         loss_1 = loss_1_2 * single_weight + loss_1

                    # loss_whole += loss_1 / weight_1 * reg_factor

            # if not self.mode == 'test':
            #     if only_class:
            #         loss_whole /= (clas_factor + autoenc_factor)
            #     else:
            #         loss_whole /= (clas_factor + autoenc_factor)
            # else:
            #     loss_whole /= (clas_factor)
        # else:
        #     if not loss_whole == None:
        #         loss_whole = None
        return loss_whole, periods

    def calc_cand_loss(self, cand_data):

        weight_factor = self.used_loss_weights[3] * self.used_loss_weights[0]
        output = cand_data[0][:,:2]
        target = torch.fmod(cand_data[1][:,2],2).long()

        cand_loss = loss_2 = self.net.loss_2(output, target) * weight_factor

        return cand_loss

    def estimate_period(self, tensor):
        conv_factor = 0.00064 * 4
        # tensor_reshaped = tensor.permute(1, 0, 2)
        # tensor_acf = F.conv1d(tensor_reshaped, tensor,
        #                       padding=2500, groups=tensor.shape[0])
        tensor_acf = self.calc_acf(tensor)
        middle = int(tensor_acf.shape[2] / 2)
        part_acf = tensor_acf[:, :, middle + 100:]
        max_vals = torch.argmax(part_acf, dim=2).float()
        periods = (max_vals + 100) * conv_factor  # / self.net.mean_vals[0]
        return periods.permute(1, 0).float()

    def crop_target_output(self, crop, target, output):
        return target[:, :, crop:-crop], output  # [:, :, crop:-crop]

    def calc_acf(self, tensor, padding=2500):
        tensor_reshaped = tensor.permute(1, 0, 2)
        tensor_acf = F.conv1d(tensor_reshaped, tensor,
                              padding=padding, groups=tensor.shape[0])

        return tensor_acf

    def relabel_set_no_cand(self, output_labels, target, reverse=False):

        # print(target.shape, output_labels.shape)

        if self.mode =='train' or (self.mode =='validation' and self.relabel_validation):
            target_nosim = target[target[:,2]!=1]
            output_labels = output_labels[target[:,2]!=1]
            softmaxed_ini = F.softmax(output_labels[:,:2], 1)

            periods = output_labels[:,2].detach()
            target_output_labels = target_nosim[:,2]

            # cand_index = target_nosim[:, 7]
            obs_index_ini = target_nosim[:, 6]
            obs_index_np = obs_index_ini.cpu().numpy().astype(int)
            softmax_pred_np = softmaxed_ini[:,1].detach().cpu().numpy()
            # obs_index = obs_index_ini[target_output_labels!=1]
            # softmaxed = softmaxed_ini[target_output_labels!=1]

            # obs_index_psrcand = obs_index_ini[cand_index==1]
            # softmax_psrcand = softmaxed_ini[cand_index==1]
            # periods_psrcand = periods[cand_index==1]

            # obs_index_cand = obs_index_ini[cand_index==0]
            # softmax_cand = softmaxed_ini[cand_index==0]
            # periods_cand = periods[cand_index==0]

            # label positives
            if not reverse:
                # threshold = 0.95
                identified_psrs = obs_index_ini[softmaxed_ini[:,1]>self.threshold_pos].cpu().numpy().astype(int)
                identified_periods = periods[softmaxed_ini[:,1]>self.threshold_pos].cpu().numpy()
 
                label_index = self.loader.dataset.noise_df.columns.get_loc("Label")
                period_index = self.loader.dataset.noise_df.columns.get_loc("P0")
                for (psr, new_period) in zip(identified_psrs, identified_periods):
                    self.loader.dataset.noise_df.iat[psr, label_index] = 5
                    old_period = self.loader.dataset.noise_df.iat[psr, period_index]

                    self.loader.dataset.noise_df.iat[psr, period_index] = new_period

                if not 'Pulsar Prediction' in self.loader.dataset.noise_df.columns:
                    self.loader.dataset.noise_df["Pulsar Prediction"] = -1.
                pred_index = self.loader.dataset.noise_df.columns.get_loc("Pulsar Prediction")

                for (pred, obs_index) in zip(softmax_pred_np, obs_index_np):
                    self.loader.dataset.noise_df.iat[obs_index, pred_index] = pred


            if not reverse:
                # threshold_neg = 0.5
                nonidentified_psrs = obs_index_ini[softmaxed_ini[:,1]<self.threshold_pos].cpu().numpy().astype(int)
            else:
                # threshold_non = 0.5
                nonidentified_psrs = obs_index_ini[softmaxed_ini[:,1]>self.threshold_neg].cpu().numpy().astype(int)
            for nonpsr in nonidentified_psrs:
                label_index = self.loader.dataset.noise_df.columns.get_loc("Label")
                period_index = self.loader.dataset.noise_df.columns.get_loc("P0")
                self.loader.dataset.noise_df.iat[nonpsr, label_index] = 2
                self.loader.dataset.noise_df.iat[nonpsr, period_index] = np.nan

    def add_test_predictions(self, output_labels, target):
        softmaxed_ini = F.softmax(output_labels[:,:2], 1)
        softmax_pred_np = softmaxed_ini[:,1].detach().cpu().numpy()
        obs_index_ini = target[:, 6]
        obs_index_np = obs_index_ini.cpu().numpy().astype(int)
        if not 'Pulsar Prediction' in self.loader.dataset.noise_df.columns:
            self.loader.dataset.noise_df["Pulsar Prediction"] = -1.
        pred_index = self.loader.dataset.noise_df.columns.get_loc("Pulsar Prediction")
        for (pred, obs_index) in zip(softmax_pred_np, obs_index_np):
            self.loader.dataset.noise_df.iat[obs_index, pred_index] = pred


    def label_set(self, mode='train', print_progress=True):
        self.net.eval()
        if mode == 'train':
            self.loader = self.train_loader
        if mode == 'validation':
            self.loader = self.valid_loader
        if mode == 'test':
            self.loader = self.test_loader

        for index in range(len(self.loader.dataset.noise_df)):
            if print_progress:
                print(index, end="\r")
            row = self.loader.dataset.noise_df.iloc[index]
            file = row['FileName']
            # target = torch.Tensor([row.loc['P0'], row.loc['DM'], 0, row.loc['Unnamed: 0'], index, -1, index]).to(self.net.device)

            reverse = True

            ten_x = dataset.load_file_for_prediction(file, self.loader.dataset.length, self.loader.dataset.edge, chunks=2, reverse=reverse).to(self.net.device)

            output_image, output_classifier, output_single_class, candidate_data = self.net(ten_x)

            self.relabel_slow(output_classifier, index, reverse=reverse)


    def relabel_slow(self, output_labels, index, reverse=True):

        if not 'Pulsar Prediction' in self.loader.dataset.noise_df.columns:
            self.loader.dataset.noise_df["Pulsar Prediction"] = -1.


        softmaxed_ini = F.softmax(output_labels[:,:2], 1)

        if reverse:
            half_size = softmaxed_ini.shape[0] // 2
            if (softmaxed_ini[:half_size,1] > self.threshold_pos).all():
                new_label = 5
                period = output_labels[0,2].detach().cpu().numpy()
            else:
                new_label = 2
                period = np.nan

            if (softmaxed_ini[half_size:,1] > self.threshold_neg).any():
                new_label = 2
                period = np.nan

            pred = softmaxed_ini[:half_size,1].median().detach().cpu().numpy()
        else:
            if (softmaxed_ini[:,1] > self.threshold_pos).all():
                new_label = 5
                period = output_labels[0,2].detach().cpu().numpy()
            else:
                new_label = 2
                period = np.nan

            pred = softmaxed_ini[:,1].median().detach().cpu().numpy()
        # print(new_label)
        label_index = self.loader.dataset.noise_df.columns.get_loc("Label")
        period_index = self.loader.dataset.noise_df.columns.get_loc("P0")
        pred_index = self.train_loader.dataset.noise_df.columns.get_loc("Pulsar Prediction")
        self.loader.dataset.noise_df.iat[index, label_index] = new_label
        self.loader.dataset.noise_df.iat[index, period_index] = period
        self.loader.dataset.noise_df.iat[index, pred_index] = pred
