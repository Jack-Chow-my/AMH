#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:31:29 2018

@author: jam
"""

import timeit
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from agrbm import agrbm


def fun(metrix):
    shape = metrix.shape
    for i in np.arange(shape[0]):
        for j in np.arange(shape[1]):
            if metrix[i][j] < 0.5:
                metrix[i][j] = 0
            else:
                metrix[i][j] = 1

    return metrix


if __name__ == "__main__":

    var = {}
    bit = 32
    v_number = 256
    training_epochs = 1000
    lr1 = np.linspace(np.power(10, -2.), np.power(10, -4.), training_epochs)
    var['lr'] = lr1

    # %%
    # load trian/test data

    m1_train_nor = np.load('./modal/m1_nor.npy')
    m2_train_nor = np.load('./modal/m2_nor.npy')
    m3_train_nor = np.load('./modal/m3_nor.npy')
    m1_test_nor = np.load('./modal/m1_test_nor.npy')
    m2_test_nor = np.load('./modal/m2_test_nor.npy')
    m3_test_nor = np.load('./modal/m3_test_nor.npy')

    zero_train = np.zeros(m1_train_nor.shape)
    zero_test = np.zeros(m1_test_nor.shape)

    # placeholder
    f1 = tf.placeholder(tf.float32, [None, v_number], name='f1_input')
    f2 = tf.placeholder(tf.float32, [None, v_number], name='f2_input')
    f3 = tf.placeholder(tf.float32, [None, v_number], name='f3_input')
    lr_MGRBM = tf.placeholder('float32', (), name='lr')

    n_visiable1, n_visiable2, n_visiable3, n_hidden = 256, 256, 256, bit
    grbm = agrbm(
        f1,
        f2,
        f3,
        n_visiable1=n_visiable1,
        n_visiable2=n_visiable2,
        n_visiable3=n_visiable3,
        n_hidden=n_hidden)
    cost = grbm.get_reconstruction_cost()
    persistent_chain = None
    train_ops = grbm.get_train_ops(
        learning_rate=lr_MGRBM, k=1, persistent=persistent_chain)
    init = tf.global_variables_initializer()
    STEP = []
    COST = []
    print("Start training...")

    # %% train
    with tf.Session() as sess:
        start_time = timeit.default_timer()
        sess.run(init)

        for epoch in range(training_epochs):
            #            avg_cost = 0.0
            lr = var['lr'][epoch]
            a = sess.run(train_ops,
                         feed_dict={
                             f1: m1_train_nor,
                             f2: m2_train_nor,
                             f3: m3_train_nor,
                             lr_MGRBM: lr
                         })
            avg_cost = sess.run(cost,
                                feed_dict={
                                    f1: m1_train_nor,
                                    f2: m2_train_nor,
                                    f3: m3_train_nor
                                })

            if epoch % 100 == 0:
                print("epoch", '%04d' % (epoch + 1), "cost", "{:.4f}".format(avg_cost))
                STEP.append(epoch)
                COST.append(avg_cost)

        end_time = timeit.default_timer()
        training_time = end_time - start_time
        print("Finished!")
        print("  The training ran for {0} minutes.".format(
            training_time / 60, ))

        fig = plt.figure(1)
        plt.plot(STEP, COST, color='blue', linestyle='solid', label='cost')
        plt.xlabel('steps', fontsize=12)
        plt.ylabel('loss', fontsize=12)
        plt.title('MGRBM_TRAIN_COST', fontsize=14, fontweight='bold')
        plt.legend(loc="upper right")
        plt.show()

        h_mean_train = sess.run(
            grbm.multi_propup(f1, f2, f3),
            feed_dict={
                f1: m1_train_nor,
                f2: m2_train_nor,
                f3: zero_train
            })
        h_mean_test_m1 = sess.run(
            grbm.multi_propup(f1, f2, f3),
            feed_dict={
                f1: m1_test_nor,
                f2: zero_test,
                f3: zero_test
            })

        h_mean_test_m2 = sess.run(
            grbm.multi_propup(f1, f2, f3),
            feed_dict={
                f1: zero_test,
                f2: m2_test_nor,
                f3: zero_test
            })
        h_mean_test_m3 = sess.run(
            grbm.multi_propup(f1, f2, f3),
            feed_dict={
                f1: zero_test,
                f2: zero_test,
                f3: m3_test_nor
            })
        print('h_mean_train shape:{0}'.format(h_mean_train.shape))
        print('h_mean_test_m1 shape:{0}'.format(h_mean_test_m1.shape))
        print('h_mean_test_m2 shape:{0}'.format(h_mean_test_m2.shape))
        print('h_mean_test_m3 shape:{0}'.format(h_mean_test_m3.shape))
        hc_train = fun(h_mean_train)
        hc_test_m1 = fun(h_mean_test_m1)
        hc_test_m2 = fun(h_mean_test_m2)
        hc_test_m3 = fun(h_mean_test_m3)

        # save train/test  dataset        
        np.save('./modal/hc_train_{0}.npy'.format(bit), hc_train)
        np.save('./modal/hc_test_m1_{0}.npy'.format(bit), hc_test_m1)
        np.save('./modal/hc_test_m2_{0}.npy'.format(bit), hc_test_m2)
        np.save('./modal/hc_test_m3_{0}.npy'.format(bit), hc_test_m3)
