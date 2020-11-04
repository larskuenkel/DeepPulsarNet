import numpy as np
import sys
import torch
import copy


def test_parameters(length, kernel, list_channels, stride, pool, no_pad, part):
    # Alters the kernel size and lenth to give a suitable structure of the autoencoder

    # if len(list_channels1)>= len(list_channels2):
    #     list_channels = list_channels1
    # else:
    #     list_channels = list_channels2
    if not no_pad:
        if (stride * pool - stride) % 2 != 0:
            print('Please set stride and pool so that (stride * pool - stride) % 2 == 0.')
            sys.exit()
        while (kernel - stride * pool) % 2 != 0 or (kernel - stride) % 2 != 0:
            print(
                "Invalid combination of stride and kernel. Kernel size will be increased by 1.")
            kernel += 1
            if kernel > 100:
                print('Kernel larger than 100.')
                sys.exit()

        print('Kernel size: {}'.format(kernel))
        list_channels = np.asarray(list_channels)
        block_layers = len(list_channels[list_channels > 0])
        pool_layers = len(list_channels[list_channels < 0])
        whole_layers = block_layers * 2 + pool_layers
        min_length = length / ((stride * pool) ** whole_layers)
        if min_length == int(min_length):
            needed_length = length
        else:
            if min_length * (stride * pool) < 1:
                print('Consider using less layers for the given length.')
            min_length = int(np.ceil(min_length))
            needed_length = min_length * (stride * pool) ** whole_layers
            print('Some padding of the length is necessary.')
            print("New length: {}".format(needed_length))
        new_length = int(needed_length)
    else:
        print('Kernel size: {}'.format(kernel))
        list_channels = np.asarray(list_channels)
        block_layers = len(list_channels[list_channels > 0])
        pool_layers = len(list_channels[list_channels < 0])
        whole_layers = block_layers * 2 + pool_layers
        min_length = int(np.ceil(length / ((stride * pool) ** whole_layers)))

        if min_length < 1:
            print('Consider using less layers for the given length.')
            print('Minimum length for the encoded state of 2 will be used.')
            min_length = 2

        new_length = min_length

        for element in list_channels:
            new_length = (new_length * pool * stride - stride +
                          kernel)  # * pool * stride - stride + kernel
    # if part==0:
    print('Encoded Length: {}'.format(min_length))
    # if part==1:
    #     print('Encoded Length after second encoder: {}'.format(min_length))
    return kernel, new_length, int(min_length)


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.cuda.LongTensor(np.concatenate(
        [init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


def calc_tcn_depth(kernel, layers, dilation_factor, factor=0):
    def layer_range(kernel, dilation):
        return (kernel - 1) * dilation
    last_range = 0
    depth = 0
    for layer in range(layers):
        dilation = dilation_factor ** layer
        current_range = layer_range(kernel, dilation) * 2
        last_range = last_range + current_range
        depth += 1

    if factor == 0:
        print(f'Receptive field of TCN: {last_range}')
    else:
        print(f'Receptive field of classifier TCN: {last_range * factor}')
    return last_range


def untie_weights(net):
    # paras= []
    for child in net.tcn.children():
        for single_module in child.modules():
            if single_module.__module__ == 'torch.nn.modules.conv':
                dil = single_module.dilation[0]
                single_module.weight = copy.deepcopy(single_module.weight)
                single_module.bias = copy.deepcopy(single_module.bias)
                # if dil == 15 or dil ==17:
                #     print(dil)
                #     #print(single_module.weight)
                #     paras.append(single_module.weight)
    print('Weights are now untied.')
    # if paras[1] is paras[0]:
    #     print('Parameters still tied')
    # else:
    #     print('Parameters not tied anymore')
    # torch.save(net,'dummy.pt')
    # net = torch.load('dummy.pt')
    return net
