
#%%
import h5py
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import namedtuple


def col2im(im_col, height, width, channels):
    assert im_col.shape == (channels, height * width)
    out = np.zeros((height, width, channels))
    for c in range(channels):
        im_ch = im_col[c, :]
        assert im_ch.shape == (height * width, )
        out[:, :, c] = np.reshape(im_ch, (height, width))
    return out


def im2col(im, fl_height, fl_width, stride):
    in_height, in_width, in_channels = im.shape
    out_height = (in_height - fl_height) // stride + 1
    out_width = (in_width - fl_width) // stride + 1
    out = np.zeros((fl_height * fl_width * in_channels, out_height * out_width))
    for i in range(out_height):
        for j in range(out_width):
            patch = im[i * stride:i * stride + fl_height, j * stride:j * stride + fl_width, ...]
            assert patch.shape == (fl_height, fl_width, in_channels)
            out[:, i * out_width + j] = np.reshape(patch, -1)
    return out


def conv2d(dataset, filters, bias, stride, padding):
    batches, in_height, in_width, in_channels = dataset.shape
    fl_height, fl_width, in_channels, out_channels = filters.shape
    out_width = (in_width - fl_width + 2 * padding) // stride + 1
    out_height = (in_height - fl_height + 2 * padding) // stride + 1
    out = np.zeros((batches, out_height, out_width, out_channels))    
    weights = filters.reshape(-1, filters.shape[-1]).T
    assert weights.shape == (out_channels, fl_height * fl_width * in_channels)
    for i in range(batches):
        im = dataset[i]
        assert im.shape == (in_height, in_width, in_channels)
        im_pad = np.pad(im, ((padding, padding), (padding, padding), (0, 0)), 'constant')
        assert im_pad.shape == (in_height + 2 * padding, in_width + 2 * padding, in_channels)
        im_col = im2col(im_pad, fl_height, fl_width, stride)
        assert im_col.shape == (fl_height * fl_width * in_channels, out_height * out_width)
        z = np.dot(weights, im_col) + bias.reshape(out_channels, 1)
        assert z.shape == (out_channels, out_height * out_width)
        out[i, :, :, :] = col2im(z, out_height, out_width, out_channels)
    return out


def pool(dataset, window, stride, mode = 'max'):
    batches, in_height, in_width, in_channels = dataset.shape
    out_height = (in_height - window[0]) // stride + 1
    out_width = (in_width - window[1]) // stride + 1
    out = np.zeros((batches, out_height, out_width, in_channels))    
    for i in range(batches):
        for h in range(out_height):
            for w in range(out_width):
                for c in range(in_channels):
                    patch = dataset[i, h * stride: h * stride + window[0], w * stride:w * stride + window[1], c]
                    if mode == 'max':
                        out[i, h, w, c] = np.max(patch)
                    elif mode == 'avg':
                        out[i, h, w, c] = np.average(patch)
    return out                        


def load_dataset(file_name, prefix):
    model = h5py.File(file_name, 'r')
    X = np.array(model[prefix + '_x'][:], dtype=np.float)
    # X = X.reshape((X.shape[0], -1)).T
    X = X / 255
    Y = np.array(model[prefix + '_y'][:], dtype=np.int)
    #Y = Y.reshape((1, Y.shape[0]))
    return (X, Y)


def main():
    (train_X, train_Y) = load_dataset('datasets/images_train.h5', 'train_set')
    (test_X, test_Y) = load_dataset('datasets/images_test.h5', 'test_set')
    print('{} X{} Y{}'.format('train', train_X.shape, train_Y.shape))
    print('{} X{} Y{}'.format('test', test_X.shape, test_Y.shape))
    np.random.seed(1)
    A_prev = np.random.randn(10,4,4,3)
    W = np.random.randn(2,2,3,8)
    b = np.random.randn(1,1,1,8)
    Z = conv2d(A_prev, W, b, 2, 2)
    print(Z.shape)
    print("Z's mean =", np.mean(Z))
    print("Z[3,2,1] =", Z[3,2,1])

main()
