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
        assert im_ch.shape == (height * width,)
        out[:, :, c] = np.reshape(im_ch, (height, width))
    return out


def im2col(im, fl_height, fl_width, stride):
    in_height, in_width, in_channels = im.shape
    out_height = (in_height - fl_height) // stride + 1
    out_width = (in_width - fl_width) // stride + 1
    out = np.zeros((fl_height * fl_width * in_channels, out_height * out_width))
    for h in range(out_height):
        for w in range(out_width):
            h_s = h * stride
            w_s = w * stride
            patch = im[h_s : h_s + fl_height, w_s : w_s + fl_width, ...]
            assert patch.shape == (fl_height, fl_width, in_channels)
            out[:, h * out_width + w] = np.reshape(patch, -1)
    return out


def conv2d_forward(x, filters, bias, stride, pad):
    batches, in_height, in_width, in_channels = x.shape
    fl_height, fl_width, in_channels, out_channels = filters.shape
    out_width = (in_width - fl_width + 2 * pad) // stride + 1
    out_height = (in_height - fl_height + 2 * pad) // stride + 1
    out = np.zeros((batches, out_height, out_width, out_channels))
    weights = filters.reshape(-1, filters.shape[-1]).T
    assert weights.shape == (out_channels, fl_height * fl_width * in_channels)
    for i in range(batches):
        im = x[i]
        assert im.shape == (in_height, in_width, in_channels)
        im_pad = np.pad(im, ((pad, pad), (pad, pad), (0, 0)), 'constant')
        assert im_pad.shape == (in_height + 2 * pad, in_width + 2 * pad, in_channels)
        im_col = im2col(im_pad, fl_height, fl_width, stride)
        assert im_col.shape == (
            fl_height * fl_width * in_channels,
            out_height * out_width,
        )
        z = np.dot(weights, im_col) + bias.reshape(out_channels, 1)
        assert z.shape == (out_channels, out_height * out_width)
        out[i, :, :, :] = col2im(z, out_height, out_width, out_channels)
    cache = (x, filters, bias, stride, pad)
    return out, cache


def conv2d_backward(dout, cache):
    x, filters, bias, stride, pad = cache
    batches, out_height, out_width, out_channels = dout.shape
    fl_height, fl_width, in_channels, out_channels = filters.shape
    dx = np.zeros(x.shape)
    dw = np.zeros(filters.shape)
    db = np.zeros(bias.shape)
    x_pad = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
    dx_pad = np.zeros(x_pad.shape)
    for i in range(batches):
        for h in range(out_height):
            for w in range(out_width):
                for c in range(out_channels):
                    h_s = h * stride
                    w_s = w * stride
                    patch = x_pad[i, h_s : h_s + fl_height, w_s : w_s + fl_width, :]
                    assert patch.shape == (fl_height, fl_width, in_channels)
                    weights = filters[:, :, :, c]
                    assert weights.shape == (fl_height, fl_width, in_channels)
                    dx_pad[i, h_s : h_s + fl_height, w_s : w_s + fl_width, :] += (
                        weights * dout[i, h, w, c]
                    )
                    dw[:, :, :, c] += patch * dout[i, h, w, c]
                    db[:, :, :, c] += dout[i, h, w, c]
        dx[i, :, :, :] = dx_pad[i, pad:-pad, pad:-pad, :]
    assert dx.shape == x.shape
    return dx, dw, db


def pool_forward(x, window, stride, mode='max'):
    batches, in_height, in_width, in_channels = x.shape
    out_height = (in_height - window[0]) // stride + 1
    out_width = (in_width - window[1]) // stride + 1
    out = np.zeros((batches, out_height, out_width, in_channels))
    for i in range(batches):
        for h in range(out_height):
            for w in range(out_width):
                for c in range(in_channels):
                    h_s = h * stride
                    w_s = w * stride
                    patch = x[i, h_s : h_s + window[0], w_s : w_s + window[1], c]
                    assert patch.shape == (window[0], window[1])
                    if mode == 'max':
                        out[i, h, w, c] = np.max(patch)
                    elif mode == 'avg':
                        out[i, h, w, c] = np.average(patch)
    cache = (x, window, stride, mode)
    return out, cache


def pool_backward(dout, cache):
    x, window, stride, _ = cache
    batches, out_height, out_width, channels = dout.shape
    dx = np.zeros(x.shape)
    for i in range(batches):
        for h in range(out_height):
            for w in range(out_width):
                for c in range(channels):
                    h_s = h * stride
                    w_s = w * stride
                    patch = x[i, h_s : h_s + window[0], w_s : w_s + window[1], c]
                    assert patch.shape == (window[0], window[1])
                    mask = patch == np.max(patch)
                    dx[i, h_s : h_s + window[0], w_s : w_s + window[1], c] += (
                        mask * dout[i, h, w, c]
                    )
    assert dx.shape == x.shape
    return dx


def load_dataset(file_name, prefix):
    model = h5py.File(file_name, 'r')
    X = np.array(model[prefix + '_x'][:], dtype=np.float)
    # X = X.reshape((X.shape[0], -1)).T
    X = X / 255
    Y = np.array(model[prefix + '_y'][:], dtype=np.int)
    # Y = Y.reshape((1, Y.shape[0]))
    return (X, Y)


def main():
    (train_X, train_Y) = load_dataset('data/images_train.h5', 'train_set')
    (test_X, test_Y) = load_dataset('data/images_test.h5', 'test_set')
    print('{} X{} Y{}'.format('train', train_X.shape, train_Y.shape))
    print('{} X{} Y{}'.format('test', test_X.shape, test_Y.shape))

    np.random.seed(1)
    A_prev = np.random.randn(10, 4, 4, 3)
    W = np.random.randn(2, 2, 3, 8)
    b = np.random.randn(1, 1, 1, 8)
    Z, cache = conv2d_forward(A_prev, W, b, 2, 2)
    print(Z.shape)
    print("Z's mean =", np.mean(Z))
    print("Z[3,2,1] =", Z[3, 2, 1])

    np.random.seed(1)
    dA, dW, db = conv2d_backward(Z, cache)
    print("dA_mean =", np.mean(dA))
    print("dW_mean =", np.mean(dW))
    print("db_mean =", np.mean(db))

    np.random.seed(1)
    A_prev = np.random.randn(5, 5, 3, 2)
    A, cache = pool_forward(A_prev, [2, 2], 1)
    dA = np.random.randn(5, 4, 2, 2)
    dA_prev = pool_backward(dA, cache)
    print("mode = max")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev[1,1] = ', dA_prev[1, 1])
    print()


main()
