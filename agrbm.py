#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: jam
"""

import numpy as np
import tensorflow as tf

v_number = 256
bit = 32


class agrbm(object):
    """
    Gaussian-binary Restricted Boltzmann Machines
    Note we assume that the standard deviation is a constant (not training parameter)
    You better normalize you data with range of [0, 1.0].
    """

    def __init__(self,
                 inpt1=None,
                 inpt2=None,
                 inpt3=None,
                 n_visiable1=v_number,
                 n_visiable2=v_number,
                 n_visiable3=v_number,
                 n_hidden=bit,
                 sigma=1.0,
                 W1=None,
                 W2=None,
                 W3=None,
                 hbias=None,
                 vbias1=None,
                 vbias2=None,
                 vbias3=None,
                 sample_visible=True):
        """
        :param inpt: Tensor, the input tensor [None, n_visiable]
        :param n_visiable: int, number of visiable units
        :param n_hidden: int, number of hidden units
        :param sigma: float, the standard deviation (note we use the same σ for all visible units)
        :param W, hbias, vbias: Tensor, the parameters of RBM (tf.Variable)
        :param sample_visble: bool, if True, do gaussian sampling.
        """

        self.n_visiable1 = n_visiable1
        self.n_visiable2 = n_visiable2
        self.n_visiable3 = n_visiable3
        self.n_hidden = n_hidden
        # Optionally initialize input
        if inpt1 is None:
            inpt1 = tf.placeholder(
                dtype=tf.float32, shape=[None, self.n_visiable1])
        self.input1 = inpt1
        if inpt2 is None:
            inpt2 = tf.placeholder(
                dtype=tf.float32, shape=[None, self.n_visiable2])
        self.input2 = inpt2
        if inpt3 is None:
            inpt3 = tf.placeholder(
                dtype=tf.float32, shape=[None, self.n_visiable3])
        self.input3 = inpt3
        # Initialize the parameters if not given
        if W1 is None:
            bounds = -4.0 * np.sqrt(6.0 / (self.n_visiable1 + self.n_hidden))
            W1 = tf.Variable(
                tf.random_uniform(
                    [self.n_visiable1, self.n_hidden],
                    minval=-bounds,
                    maxval=bounds),
                dtype=tf.float32)
        if W2 is None:
            bounds = -4.0 * np.sqrt(6.0 / (self.n_visiable2 + self.n_hidden))
            W2 = tf.Variable(
                tf.random_uniform(
                    [self.n_visiable2, self.n_hidden],
                    minval=-bounds,
                    maxval=bounds),
                dtype=tf.float32)
        if W3 is None:
            bounds = -4.0 * np.sqrt(6.0 / (self.n_visiable3 + self.n_hidden))
            W3 = tf.Variable(
                tf.random_uniform(
                    [self.n_visiable3, self.n_hidden],
                    minval=-bounds,
                    maxval=bounds),
                dtype=tf.float32)
        if hbias is None:
            hbias = self.get_variable_zero([
                self.n_hidden,
            ])
        if vbias1 is None:
            vbias1 = self.get_variable_zero([
                self.n_visiable1,
            ])
        if vbias2 is None:
            vbias2 = self.get_variable_zero([
                self.n_visiable2,
            ])
        if vbias3 is None:
            vbias3 = self.get_variable_zero([
                self.n_visiable3,
            ])
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        self.hbias = hbias
        self.vbias1 = vbias1
        self.vbias2 = vbias2
        self.vbias3 = vbias3
        self.params = [
            self.W1, self.W2, self.W3, self.hbias, self.vbias1, self.vbias2,
            self.vbias3
        ]
        self.sigma = sigma
        self.sample_visible = sample_visible

    #    def get_variable_random(self, shape=[], dtype=tf.float32):
    #        return tf.Variable(tf.random_normal(shape, stddev=2, dtype=dtype))

    def get_variable_zero(self, shape=[], dtype=tf.float32):
        return tf.Variable(tf.zeros(shape), dtype=tf.float32)

    def multi_propup(self, f1, f2, f3):
        f1w1 = tf.matmul(f1, self.W1) / self.sigma ** 2
        f2w2 = tf.matmul(f2, self.W2) / self.sigma ** 2
        f3w3 = tf.matmul(f3, self.W3) / self.sigma ** 2
        return tf.nn.sigmoid(f1w1 + f2w2 + f3w3 + self.hbias)

    def multi_propdown(self, h):
        a1 = tf.matmul(h, tf.transpose(self.W1)) + self.vbias1
        a2 = tf.matmul(h, tf.transpose(self.W2)) + self.vbias2
        a3 = tf.matmul(h, tf.transpose(self.W3)) + self.vbias3
        return a1, a2, a3

    def sample_gaussian(self, x):
        return x + tf.random_normal(tf.shape(x), mean=0.0, stddev=self.sigma, dtype=tf.float32)

    def sample_bernoulli(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    def sample_h_given_v(self, v1_0_sample, v2_0_sample, v3_0_sample):
        h_0_mean = self.multi_propup(v1_0_sample, v2_0_sample, v3_0_sample)
        h_0_sample = self.sample_bernoulli(h_0_mean)
        return (h_0_mean, h_0_sample)

    def get_h_given_v(self, v1_0_sample, v2_0_sample, v3_0_sample):
        h_0_mean = self.multi_propup(v1_0_sample, v2_0_sample, v3_0_sample)
        return h_0_mean

    def sample_v_given_h(self, h_0_sample):
        v1_1_mean, v2_1_mean, v3_1_mean = self.multi_propdown(h_0_sample)
        v1_1_sample = self.sample_gaussian(v1_1_mean)
        v2_1_sample = self.sample_gaussian(v2_1_mean)
        v3_1_sample = self.sample_gaussian(v3_1_mean)
        return (v1_1_mean, v2_1_mean, v3_1_mean, v1_1_sample, v2_1_sample,
                v3_1_sample)

    def gibbs_vhv(self, v1_0_sample, v2_0_sample, v3_0_sample):
        h_0_mean, h_0_sample = self.sample_h_given_v(v1_0_sample, v2_0_sample,
                                                     v3_0_sample)
        v1_1_mean, v2_1_mean, v3_1_mean, v1_1_sample, v2_1_sample, v3_1_sample = self.sample_v_given_h(
            h_0_sample)
        return (h_0_mean, h_0_sample, v1_1_mean, v2_1_mean, v3_1_mean,
                v1_1_sample, v2_1_sample, v3_1_sample)

    def gibbs_hvh(self, h_0_sample):
        v1_1_mean, v2_1_mean, v3_1_mean, v1_1_sample, v2_1_sample, v3_1_sample = self.sample_v_given_h(
            h_0_sample)
        h_1_mean, h_1_sample = self.sample_h_given_v(v1_1_sample, v2_1_sample,
                                                     v3_1_sample)
        return (v1_1_mean, v2_1_mean, v3_1_mean, v1_1_sample, v2_1_sample,
                v3_1_sample, h_1_mean, h_1_sample)

    def get_train_ops(self, learning_rate=0.01, k=1, persistent=None):

        h_0_mean, h_0_sample = self.sample_h_given_v(self.input1, self.input2, self.input3)
        v1_1_mean, v2_1_mean, v3_1_mean, \
        v1_1_sample, v2_1_sample, v3_1_sample = self.sample_v_given_h(h_0_sample)
        h_1_mean, h_1_sample = self.sample_h_given_v(v1_1_sample, v2_1_sample, v3_1_sample)

        w1_positive_grad = tf.matmul(tf.transpose(self.input1), h_0_sample) / self.sigma ** 2
        w1_negative_grad = tf.matmul(tf.transpose(v1_1_sample), h_1_sample) / self.sigma ** 2

        w2_positive_grad = tf.matmul(tf.transpose(self.input2), h_0_sample) / self.sigma ** 2
        w2_negative_grad = tf.matmul(tf.transpose(v2_1_sample), h_1_sample) / self.sigma ** 2

        w3_positive_grad = tf.matmul(tf.transpose(self.input3), h_0_sample) / self.sigma ** 2
        w3_negative_grad = tf.matmul(tf.transpose(v3_1_sample), h_1_sample) / self.sigma ** 2

        update_w1 = self.W1 + learning_rate * (
                w1_positive_grad - w1_negative_grad) / tf.to_float(tf.shape(self.input1)[0])
        update_w2 = self.W2 + learning_rate * (
                w2_positive_grad - w2_negative_grad) / tf.to_float(tf.shape(self.input2)[0])
        update_w3 = self.W3 + learning_rate * (
                w3_positive_grad - w3_negative_grad) / tf.to_float(tf.shape(self.input3)[0])

        update_vb1 = self.vbias1 + learning_rate * tf.reduce_mean(self.input1 - v1_1_sample, 0)
        update_vb2 = self.vbias2 + learning_rate * tf.reduce_mean(self.input2 - v2_1_sample, 0)
        update_vb3 = self.vbias3 + learning_rate * tf.reduce_mean(self.input3 - v3_1_sample, 0)

        update_hb = self.hbias + learning_rate * tf.reduce_mean(h_0_sample - h_1_sample, 0)
        # Compute the gradients
        gparams = [update_w1, update_w2, update_w3, update_hb, update_vb1, update_vb2, update_vb3]
        new_params = []
        for gparam, param in zip(gparams, self.params):
            new_params.append(tf.assign(param, gparam))
        return new_params

    def get_reconstruction_cost(self):
        """Compute the cross-entropy of the original input and the reconstruction"""
        activation_h = self.multi_propup(self.input1, self.input2, self.input3)
        activation_f1, activation_f2, activation_f3 = self.multi_propdown(
            activation_h)

        mse1 = tf.reduce_mean(
            tf.reduce_sum(tf.square(self.input1 - activation_f1), axis=1))
        mse2 = tf.reduce_mean(
            tf.reduce_sum(tf.square(self.input2 - activation_f2), axis=1))
        mse3 = tf.reduce_mean(
            tf.reduce_sum(tf.square(self.input3 - activation_f3), axis=1))
        mse = mse1 + mse2 + mse3
        return mse

    def reconstruction(self, f1, f2, f3):
        act_h = self.multi_propup(f1, f2, f3)
        return self.multi_propdown(act_h)


