import visdom
import torchnet as tnt
from torchnet.logger import VisdomPlotLogger
from torchnet.logger import VisdomLogger
import numpy as np
import torch.nn.functional as F
import torch
import pycm
from collections import OrderedDict


class logger():
    #  class to store all relevant meter and logging objects
    def __init__(self, plot, name):
        if name:
            self.name = name
        else:
            self.name = 'last_model'
        self.loss_meter = tnt.meter.AverageValueMeter()
        self.loss_meter_2 = tnt.meter.AverageValueMeter()
        self.loss_meter_3 = tnt.meter.AverageValueMeter()
        self.time_meter = tnt.meter.TimeMeter(unit=False)

        self.classerr = tnt.meter.ClassErrorMeter(accuracy=True)
        self.confusion_meter = tnt.meter.ConfusionMeter(2, normalized=True)
        self.plot = plot
        if plot:
            port = 8097
            self.train_loss_logger = VisdomPlotLogger(
                'line', port=port, opts={'title': 'Train Loss'})

            self.valid_loss_logger = VisdomPlotLogger(
                'line', port=port, opts={'title': 'Validation Loss'})

            self.train_err_logger = VisdomPlotLogger(
                'line', port=port, opts={'title': 'Train Class Error'})
            self.valid_err_logger = VisdomPlotLogger(
                'line', port=port, opts={'title': 'Validation Class Error'})
            self.test_err_logger = VisdomPlotLogger(
                'line', port=port, opts={'title': 'Test Class Error'})

            names = list(range(2))
            self.train_conf_logger = VisdomLogger('heatmap', port=port,
                                                  opts={'title': 'Train Confusion matrix',
                                                        'columnnames': names, 'rownames': names, 'xmax': 1, 'xmin': 0})
            self.valid_conf_logger = VisdomLogger('heatmap', port=port,
                                                  opts={'title': 'Validation Confusion matrix',
                                                        'columnnames': names, 'rownames': names, 'xmax': 1, 'xmin': 0})
            self.test_conf_logger = VisdomLogger('heatmap', port=port,
                                                 opts={'title': 'Test Confusion matrix',
                                                       'columnnames': names, 'rownames': names, 'xmax': 1, 'xmin': 0})
            self.vis = visdom.Visdom()
            self.vis.close(None)
            #self.plot = 1
        self.reset_best_values()
        self.best_test_values = np.zeros(3)
        self.best_test_values[2] = 10**9
        self.last_values = np.asarray((1000, -1))
        self.values = []
        self.epoch = 0
        self.conf_val = np.zeros(2)
        self.conf_val[0] = 2
        self.conf_val[1] = -1
        self.conf_mat = {'train':[0], 'validation':[0], 'test':[0]}
        self.conf_mat_split = {'train sim':[0], 'train real':[0], 'valid sim':[0], 'valid real':[0]}
        #self.conf_mat = OrderedDict([('train',[0]), ('validation',[0]), ('test',[0])])
        self.best_ffa_values = np.zeros(2)
        self.best_ffa_values[1] = 100
        self.stats = []#[[0,0,0]]

    def reset_best_values(self):
        self.best_values = np.zeros(4)
        self.best_values[2] = 10**9

    def save_best_values(self, epoch, train_loss, val_loss, loss_test):
        if self.best_values[2] > val_loss:
            self.best_values = np.asarray(
                (epoch, train_loss, val_loss, loss_test))
        if loss_test is not None:
            if self.best_test_values[2] > loss_test:
                self.best_test_values = np.asarray((epoch, val_loss, loss_test))

    def save_ffa_values(self, ffa_count, non_detec):
        if self.best_ffa_values[0] < ffa_count:
            self.best_ffa_values[0] = ffa_count
        if self.best_ffa_values[1] > len(non_detec):
            self.best_ffa_values[1] = len(non_detec)

    def save_last_values(self, loss, epoch):
        self.last_values = np.asarray((loss, epoch))

    def reset_meters(self):
        self.loss_meter.reset()
        self.loss_meter_2.reset()
        self.loss_meter_3.reset()
        self.classerr.reset()
        self.confusion_meter.reset()

    def log_loss(self, epoch, lr, noise, loss_train, loss_valid, loss_test, reg_loss_train, 
                                  clas_loss_train, im_loss_train, reg_loss, clas_loss, im_loss):
        conf_string = ''
        conf_string_split = ''
        mat_string = ''
        if len(self.conf_mat['train']) >1:
            conf_string = 'MCC: '
            for mode in self.conf_mat:
                if len(self.conf_mat[mode])>1:
                    mat_dict = conv_mat_to_dict(self.conf_mat[mode])
                    pycm_cm = pycm.ConfusionMatrix(matrix=mat_dict)
                    mcc = pycm_cm.MCC['pulsar']
                    if mode == 'train':
                        self.last_train_mcc = mcc if not mcc == 'None' else 0
                    if mode == 'validation':
                        self.last_val_mcc = mcc if not mcc == 'None' else 0
                    if mode == 'test':
                        self.last_test_mcc = mcc

                    if mcc == 'None':
                        mcc_string = 'nan  |'
                    else:
                        mcc_string = "{:.2f}".format(mcc)+ ' |'
                    conf_string += ' ' + mcc_string + ''

            conf_string_split = 'Split: '
            for mode in self.conf_mat_split:
                if len(self.conf_mat_split[mode])>1:
                    mat_dict = conv_mat_to_dict(self.conf_mat_split[mode])
                    pycm_cm = pycm.ConfusionMatrix(matrix=mat_dict)
                    mcc = pycm_cm.MCC['pulsar']
                    if mode == 'train sim':
                        self.last_train_mcc_sim = mcc if not mcc == 'None' else 0
                    if mode == 'train real':
                        self.last_train_mcc_real = mcc if not mcc == 'None' else 0
                    if mode == 'valid sim':
                        self.last_valid_mcc_sim = mcc if not mcc == 'None' else 0
                    if mode == 'valid real':
                        self.last_valid_mcc_real = mcc if not mcc == 'None' else 0

                    if mcc == 'None':
                        mcc_string = 'nan  |'
                    else:
                        mcc_string = "{:.2f}".format(mcc)+ ' |'
                    conf_string_split += ' ' + mcc_string + ''
        if loss_test is None:
            if epoch==0:
                print(f"Epoch Count | Loss Rate  | Total Loss         Split Loss:|Train (class&MSE)||Valid(class&MSE) | MCC   Train  Valid      Train(Sim   Real) Valid(Sim   Real)")
                        # Epoch:  32 LR: 0.000500000 Train: 0.19716 Valid: 0.16227 |0.106255 0.090906||0.111256 0.051013| MCC:  0.53 | 0.75 | |Split:  0.53 | 0.68 ||Time: 26.37
            print('Epoch: {:3.0f} LR: {:.9f} Train: {:.5f} Valid: {:.5f} |{:.6f} {:.6f}||{:.6f} {:.6f}| {} |{}|Time: {:.2f}'.format(
                epoch, lr, loss_train, loss_valid, 
                                  clas_loss_train, im_loss_train, clas_loss, im_loss, conf_string,conf_string_split,  self.time_meter.value()))
            # self.last_val_mcc = 0
        else:
            if np.count_nonzero(self.confusion_meter.value()) != 0:
                conf_array = self.confusion_meter.value().flatten()
                mat_string = np.array2string(
                    conf_array, precision=3, floatmode='fixed')[1:-1]
                conf_val = 2 - conf_array[0] - conf_array[-1]
                if self.conf_val[0] > conf_val:
                    self.conf_val[0] = conf_val
                if mcc !='None':
                    if self.conf_val[1] < float(mcc):
                        self.conf_val[1] = float(mcc)
                # conf_string += ' ' + \
                #     np.array2string(conf_val, precision=2, floatmode='fixed')
            print('Epoch: {:3.0f} LR: {:.9f} Train: {:.5f} Valid: {:.5f} |{:.6f} {:.6f}||{:.6f} {:.6f}| Test: {:.5f} {} [{}] Time: {:.2f}'.format(
                epoch, lr, loss_train, loss_valid,  
                                  clas_loss_train, im_loss_train,
                                  clas_loss, im_loss, loss_test, conf_string, mat_string, self.time_meter.value()))
        self.values.append([epoch, loss_train, loss_valid, np.nansum((clas_loss)), loss_test, self.last_val_mcc, self.last_train_mcc, self.last_train_mcc_sim, self.last_valid_mcc_sim])
        #print(np.nansum((reg_loss, clas_loss)))
        with open("./logs/log_{}.txt".format(self.name), "a") as myfile:
            myfile.write("\n {:.2f} {:.5} {:.5f} {:.6f} {:.6f} {:.5f} {} {} {} {}".format(
                epoch, noise[0], loss_train, loss_valid, clas_loss, im_loss, loss_test, conf_string, conf_string_split, mat_string))

    def log_command(self, command):
        with open("./logs/log_{}.txt".format(self.name), "a") as myfile:
            myfile.write("\n {}".format(command))

    def stack_output(self, output, target, single_out):
        self.out_stack.extend(output)
        self.out_single_stack.extend(single_out)
        self.target_stack.extend(target)

    def plot_regressor(self, mode):
        if self.plot:
            out_array = np.asarray(self.out_stack)
            target_array = np.asarray(self.target_stack)
            out_labels = F.softmax(torch.tensor(out_array[:, -2:]), dim=1)
            out_class = (np.round(out_labels[:, 1]).numpy() + 1).astype(int)
            periods = np.stack((out_array[:, 0], target_array[:, 0]), axis=-1)
            dms = np.stack((out_array[:, 1], target_array[:, 1]), axis=-1)

            # colours = (out_labels[:,1].numpy()*255).astype(int)
            colours = np.asarray(((0, 0, 255), (255, 0, 0)))
            period_name = 'period_'  # + str(self.epoch)
            dm_name = 'dm_'  # + str(self.epoch)
            if mode == 'validation':
                self.vis.scatter(periods, out_class, win=period_name, opts=dict(ylabel='True Period / mean',
                                                                                xlabel='Measured Period / mean', markercolor=colours))
                self.vis.scatter(dms, out_class, win=dm_name, opts=dict(
                    ylabel='True DM / mean', xlabel='Measured DM / mean', markercolor=colours))
            elif mode == 'test':
                # print(out_labels)
                real_period = periods[out_class == 2, :]
                false_period = periods[out_class == 1, :]
                real_dm = dms[out_class == 2, :]
                false_dm = dms[out_class == 1, :]
                blue_color = np.expand_dims(np.asarray((17, 174, 207)), 0)
                red_color = np.expand_dims(np.asarray((251, 229, 8)), 0)
                self.vis.scatter(false_period, win=period_name, update='new', name='test', opts=dict(
                    markersymbol='star', markercolor=blue_color))
                self.vis.scatter(real_period, win=period_name, update='append', name='test', opts=dict(
                    markersymbol='star', markercolor=red_color))
                self.vis.scatter(false_dm, win=dm_name, update='new', name='test', opts=dict(
                    markersymbol='star', markercolor=blue_color))
                self.vis.scatter(real_dm, win=dm_name, update='append', name='test', opts=dict(
                    markersymbol='star', markercolor=red_color))

    def plot_autoencoder(self, input_data, output, target, labels):
        # output = output[:,:input_data.shape[1],:]
        if self.plot:
            if output.shape[2] < 6000:
                start = 0
                end = -1
            else:
                start = 2000
                end = -1
            try:
                cpu_labels = labels[:, 2].cpu()
                max_vals = np.max(target[:, 0, :20000], axis=1)
                pulsar_index = np.nonzero(cpu_labels)
                plotted_fil = np.random.choice(pulsar_index[0])
                noise_index = np.nonzero(cpu_labels == 0)
                plotted_fil_noise = np.random.choice(noise_index[0])
            except IndexError:
                plotted_fil = 0
                plotted_fil_noise = 1

            loss = F.mse_loss(torch.tensor(
                output[:,:1,:]), torch.tensor(target[:,:1,:]), reduction='none')
            loss_mean = torch.mean(loss, dim=1)
            loss_mean = torch.mean(loss_mean, dim=1)
            worst_loss = torch.argmax(loss_mean)
            best_loss = torch.argmin(loss_mean)
            self.vis.heatmap(
                input_data[plotted_fil, :, 1000:6000], win='ini_data', opts=dict(title='Input'))
            x_vals = np.arange(len(output[plotted_fil, 0, start:end]))
            self.vis.line(output[plotted_fil, 0, start:end],x_vals, win='out_data', opts=dict(
                title=f'Output <br> {loss_mean[plotted_fil]} {labels[plotted_fil,-1]}'))
            self.vis.line(target[plotted_fil, 0, start:end],x_vals, win='out_data', name='new', update='append')
            # x_vals_2 = np.arange(len(output[plotted_fil, 0, :]))
            # self.vis.line(output[plotted_fil, 1, :],x_vals_2, win='out_data_2')#, opts=dict(
            #     # title=f'Output <br> {loss_mean[plotted_fil]} {labels[plotted_fil,-1]}'))
            # self.vis.line(target[plotted_fil, 1, :],x_vals_2, win='out_data_2', name='new', update='append')
            # self.vis.line(output[plotted_fil, 2, :],x_vals_2, win='out_data_3')#, opts=dict(
            #     # title=f'Output <br> {loss_mean[plotted_fil]} {labels[plotted_fil,-1]}'))
            # self.vis.line(target[plotted_fil, 2, :],x_vals_2, win='out_data_3', name='new', update='append')
            self.vis.line(
                output[plotted_fil, 0, start:end] - target[plotted_fil, 0, start:end],x_vals, win='diff_data', opts=dict(title='Difference'))
            self.vis.line(output[worst_loss, 0, start:end],x_vals, win='worst_loss', opts=dict(
                title=f'Worst loss <br> {loss_mean[worst_loss]} {labels[worst_loss,-1]}'))
            self.vis.line(target[worst_loss, 0, start:end],x_vals, win='worst_loss', name='new', update='append')
            self.vis.line(output[best_loss, 0, start:end],x_vals, win='best_loss', opts=dict(
                title=f'Best loss <br> {loss_mean[best_loss]} {labels[best_loss,-1]}'))
            self.vis.line(target[best_loss, 0, start:end],x_vals, win='best_loss', name='new', update='append')



    def plot_estimate(self, mode, no_reg=True):
        if self.plot:
            out_array = np.asarray(self.out_stack)
            target_array = np.asarray(self.target_stack)
            # print(out_array, out_array.shape, target_array.shape)
            # print(out_array)
            out_labels = F.softmax(torch.tensor(out_array[:, :2]), dim=1)
            out_class = (np.round(out_labels[:, 1]).numpy() + 1).astype(int)
            # if no_reg:
            #     periods = np.stack((out_array[:, 0], target_array[:, 0]), axis=-1)
            # else:
            #     periods = np.stack((out_array[:, 3], target_array[:, 0]), axis=-1)
            periods = np.stack((out_array[:, 2], target_array[:, 0]), axis=-1)

            # colours = (out_labels[:,1].numpy()*255).astype(int)
            colours = np.asarray(((0, 0, 255), (255, 0, 0)))
            period_name = 'period_'  # + str(self.epoch)
            dm_name = 'dm_'  # + str(self.epoch)
            # try:
            if mode == 'validation':
                self.vis.scatter(periods, out_class, win=period_name, opts=dict(ylabel='True Period / mean',
                                                                                xlabel='Measured Period / mean', markercolor=colours))
            elif mode == 'test':
                # print(out_labels)
                real_period = periods[out_class == 2, :]
                false_period = periods[out_class == 1, :]
                blue_color = np.expand_dims(np.asarray((17, 174, 207)), 0)
                red_color = np.expand_dims(np.asarray((251, 229, 8)), 0)
                self.vis.scatter(false_period, win=period_name, update='new', name='test', opts=dict(
                    markersymbol='star', markercolor=blue_color))
                self.vis.scatter(real_period, win=period_name, update='append', name='test', opts=dict(
                    markersymbol='star', markercolor=red_color))
            # except AssertionError:
            #     print('Error while ploting scatter plot.')
            #     print(out_class)

def conv_mat_to_dict(mat):
    mat = mat.astype(int)
    class_1 = 'noise'
    class_2 = 'pulsar'
    mat_dict = {class_1: {class_1: int(mat[0,0]), class_2: int(mat[0,1])},class_2: {class_1: int(mat[1,0]), class_2: int(mat[1,1])}}
    return mat_dict