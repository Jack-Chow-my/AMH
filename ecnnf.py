# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jam
"""

import tensorflow as tf
import scipy.misc
import scipy.io

MODEL_DIR = '.data/cnn_f/imagenet-vgg-f.mat'
batch_size = 17
keep_prob = 0.8


def net_strucuture(input_image, bit, keep_prob):
    data = scipy.io.loadmat(MODEL_DIR)
    layers = ('conv1', 'relu1', 'norm1', 'pool1', 'conv2', 'relu2', 'norm2',
              'pool2', 'conv3', 'relu3', 'conv4', 'relu4', 'conv5', 'relu5',
              'pool5', 'fc6', 'relu6', 'fc7', 'relu7')
    weights = data['layers'][0]
    # mean = data['normalization'][0][0][0]
    net = {}
    ops = []
    current = tf.convert_to_tensor(input_image, dtype='float32')
    for i, name in enumerate(layers[:-1]):
        if name.startswith('conv'):
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            # kernels = np.transpose(kernels, (1, 0, 2, 3))

            bias = bias.reshape(-1)
            pad = weights[i][0][0][1]
            stride = weights[i][0][0][4]
            current = _conv_layer(current, kernels, bias, pad, stride, i, ops,
                                  net)
        elif name.startswith('relu'):
            current = tf.nn.relu(current)
        elif name.startswith('pool'):
            stride = weights[i][0][0][1]
            pad = weights[i][0][0][2]
            area = weights[i][0][0][5]
            current = _pool_layer(current, stride, pad, area)
        elif name.startswith('fc'):
            kernels, bias = weights[i][0][0][0][0]
            bias = bias.reshape(-1)
            current = _full_conv(current, kernels, bias, i, ops, net)
        elif name.startswith('norm'):
            current = tf.nn.local_response_normalization(
                current, depth_radius=2, bias=2.000, alpha=0.0001, beta=0.75)
        net[name] = current

    # ExtendNet
    # fc8
    w_fc8 = tf.get_variable('weights1_20',
                            shape=[4096, 1000],
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
    b_fc8 = tf.get_variable('biases1_20',
                            shape=[1000],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.1))
    current = tf.reshape(current, [-1, 4096])
    #    fc8 = tf.matmul(tf.squeeze(current), w_fc8) + b_fc8
    fc8 = tf.matmul(current, w_fc8) + b_fc8
    net['weigh1_20'] = w_fc8
    net['b1_20'] = b_fc8
    fc8_ = tf.nn.relu(fc8)
    net['fc1_8'] = fc8
    net['relu1_8'] = fc8_

    # dropout
    h_fc8_drop = tf.nn.dropout(fc8_, keep_prob)
    # fc9
    w_fc9 = tf.get_variable('weights1_21',
                            shape=[1000, bit],
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
    b_fc9 = tf.get_variable('biases1_21',
                            shape=[bit],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.1))
    fc9 = tf.matmul(h_fc8_drop, w_fc9) + b_fc9
    net['weigh1_21'] = w_fc9
    net['b1_21'] = b_fc9
    net['fc1_9'] = fc9

    # softmax
    w_cla = tf.Variable(tf.random_normal([bit, 20], stddev=1.0) * 0.01)
    b_cla = tf.Variable(tf.random_normal([20], stddev=1.0) * 0.01)
    classier = tf.matmul(fc9, w_cla) + b_cla
    net['weigh1_23'] = w_cla
    net['b1_23'] = b_cla
    net['classier1'] = classier
    return classier, net


def net_strucuture2(input_image, bit, keep_prob):
    data = scipy.io.loadmat(MODEL_DIR)
    layers = ('conv1', 'relu1', 'norm1', 'pool1', 'conv2', 'relu2', 'norm2',
              'pool2', 'conv3', 'relu3', 'conv4', 'relu4', 'conv5', 'relu5',
              'pool5', 'fc6', 'relu6', 'fc7', 'relu7')
    weights = data['layers'][0]
    # mean = data['normalization'][0][0][0]
    net = {}
    ops = []
    current = tf.convert_to_tensor(input_image, dtype='float32')
    for i, name in enumerate(layers[:-1]):
        if name.startswith('conv'):
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            # kernels = np.transpose(kernels, (1, 0, 2, 3))

            bias = bias.reshape(-1)
            pad = weights[i][0][0][1]
            stride = weights[i][0][0][4]
            current = _conv_layer(current, kernels, bias, pad, stride, i, ops,
                                  net)
        elif name.startswith('relu'):
            current = tf.nn.relu(current)
        elif name.startswith('pool'):
            stride = weights[i][0][0][1]
            pad = weights[i][0][0][2]
            area = weights[i][0][0][5]
            current = _pool_layer(current, stride, pad, area)
        elif name.startswith('fc'):
            kernels, bias = weights[i][0][0][0][0]
            bias = bias.reshape(-1)
            current = _full_conv(current, kernels, bias, i, ops, net)
        elif name.startswith('norm'):
            current = tf.nn.local_response_normalization(
                current, depth_radius=2, bias=2.000, alpha=0.0001, beta=0.75)
        net[name] = current

    # ExtendNet
    # fc8
    w_fc8 = tf.get_variable('weights2_20',
                            shape=[4096, 1000],
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
    b_fc8 = tf.get_variable('biases2_20',
                            shape=[1000],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.1))
    current = tf.reshape(current, [-1, 4096])
    #    fc8 = tf.matmul(tf.squeeze(current), w_fc8) + b_fc8
    fc8 = tf.matmul(current, w_fc8) + b_fc8
    net['weigh2_20'] = w_fc8
    net['b2_20'] = b_fc8
    fc8_ = tf.nn.relu(fc8)
    net['fc2_8'] = fc8
    net['relu2_8'] = fc8_

    # dropout
    h_fc8_drop = tf.nn.dropout(fc8_, keep_prob)
    # fc9
    w_fc9 = tf.get_variable('weights2_21',
                            shape=[1000, bit],
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
    b_fc9 = tf.get_variable('biases2_21',
                            shape=[bit],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.1))
    fc9 = tf.matmul(h_fc8_drop, w_fc9) + b_fc9
    net['weigh2_21'] = w_fc9
    net['b2_21'] = b_fc9
    net['fc2_9'] = fc9

    # softmax
    w_cla = tf.Variable(tf.random_normal([bit, 20], stddev=1.0) * 0.01)
    b_cla = tf.Variable(tf.random_normal([20], stddev=1.0) * 0.01)
    classier = tf.matmul(fc9, w_cla) + b_cla
    net['weigh2_23'] = w_cla
    net['b2_23'] = b_cla
    net['classier2'] = classier
    return classier, net


def net_strucuture3(input_image, bit, keep_prob):
    data = scipy.io.loadmat(MODEL_DIR)
    layers = ('conv1', 'relu1', 'norm1', 'pool1', 'conv2', 'relu2', 'norm2',
              'pool2', 'conv3', 'relu3', 'conv4', 'relu4', 'conv5', 'relu5',
              'pool5', 'fc6', 'relu6', 'fc7', 'relu7')
    weights = data['layers'][0]
    # mean = data['normalization'][0][0][0]
    net = {}
    ops = []
    current = tf.convert_to_tensor(input_image, dtype='float32')
    for i, name in enumerate(layers[:-1]):
        if name.startswith('conv'):
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            # kernels = np.transpose(kernels, (1, 0, 2, 3))

            bias = bias.reshape(-1)
            pad = weights[i][0][0][1]
            stride = weights[i][0][0][4]
            current = _conv_layer(current, kernels, bias, pad, stride, i, ops,
                                  net)
        elif name.startswith('relu'):
            current = tf.nn.relu(current)
        elif name.startswith('pool'):
            stride = weights[i][0][0][1]
            pad = weights[i][0][0][2]
            area = weights[i][0][0][5]
            current = _pool_layer(current, stride, pad, area)
        elif name.startswith('fc'):
            kernels, bias = weights[i][0][0][0][0]
            bias = bias.reshape(-1)
            current = _full_conv(current, kernels, bias, i, ops, net)
        elif name.startswith('norm'):
            current = tf.nn.local_response_normalization(
                current, depth_radius=2, bias=2.000, alpha=0.0001, beta=0.75)
        net[name] = current

    # ExtendNet
    # fc8
    w_fc8 = tf.get_variable('weights3_20',
                            shape=[4096, 1000],
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
    b_fc8 = tf.get_variable('biases3_20',
                            shape=[1000],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.1))
    current = tf.reshape(current, [-1, 4096])
    #    fc8 = tf.matmul(tf.squeeze(current), w_fc8) + b_fc8
    fc8 = tf.matmul(current, w_fc8) + b_fc8
    net['weigh3_20'] = w_fc8
    net['b3_20'] = b_fc8
    fc8_ = tf.nn.relu(fc8)
    net['fc3_8'] = fc8
    net['relu3_8'] = fc8_

    # dropout
    h_fc8_drop = tf.nn.dropout(fc8_, keep_prob)
    # fc9
    w_fc9 = tf.get_variable('weights3_21',
                            shape=[1000, bit],
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
    b_fc9 = tf.get_variable('biases3_21',
                            shape=[bit],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.1))
    fc9 = tf.matmul(h_fc8_drop, w_fc9) + b_fc9
    net['weigh3_21'] = w_fc9
    net['b3_21'] = b_fc9
    net['fc3_9'] = fc9

    # softmax
    w_cla = tf.Variable(tf.random_normal([bit, 20], stddev=1.0) * 0.01)
    b_cla = tf.Variable(tf.random_normal([20], stddev=1.0) * 0.01)
    classier = tf.matmul(fc9, w_cla) + b_cla
    net['weigh3_23'] = w_cla
    net['b3_23'] = b_cla
    net['classier3'] = classier
    return classier, net


def _conv_layer(input, weights, bias, pad, stride, i, ops, net):
    pad = pad[0]
    stride = stride[0]
    input = tf.pad(input, [[0, 0], [pad[0], pad[1]], [pad[2], pad[3]], [0, 0]],
                   "CONSTANT")
    w = tf.Variable(weights, name='w' + str(i), dtype='float32')
    b = tf.Variable(bias, name='bias' + str(i), dtype='float32')
    ops.append(w)
    ops.append(b)
    net['weights' + str(i)] = w
    net['b' + str(i)] = b
    conv = tf.nn.conv2d(
        input,
        w,
        strides=[1, stride[0], stride[1], 1],
        padding='VALID',
        name='conv' + str(i))
    return tf.nn.bias_add(conv, b, name='add' + str(i))


def _full_conv(input, weights, bias, i, ops, net):
    w = tf.Variable(weights, name='w' + str(i), dtype='float32')
    b = tf.Variable(bias, name='bias' + str(i), dtype='float32')
    ops.append(w)
    ops.append(b)
    net['weights' + str(i)] = w
    net['b' + str(i)] = b
    conv = tf.nn.conv2d(
        input, w, strides=[1, 1, 1, 1], padding='VALID', name='fc' + str(i))
    return tf.nn.bias_add(conv, b, name='add' + str(i))


def _pool_layer(input, stride, pad, area):
    pad = pad[0]
    area = area[0]
    stride = stride[0]
    input = tf.pad(input, [[0, 0], [pad[0], pad[1]], [pad[2], pad[3]], [0, 0]],
                   "CONSTANT")
    return tf.nn.max_pool(
        input,
        ksize=[1, area[0], area[1], 1],
        strides=[1, stride[0], stride[1], 1],
        padding='VALID')

