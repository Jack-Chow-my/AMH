#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 23:36:05 2018

@author: jam
"""

import tensorflow as tf
import numpy as np
import os
import re


train_dir1 = './data/dataset/train/color/'
train_dir2 = './data/dataset/train/depth/'
train_dir3 = './data/dataset/train/user/'
test_dir1 = './data/dataset/test/color/'
test_dir2 = './data/dataset/test/depth/'
test_dir3 = './data/dataset/test/user/'


def get_files(file_dir):
    bastas = []
    label_bastas = []
    buonissimos = []
    label_buonissimos = []
    cheduepalles = []
    label_cheduepalles = []
    combinatos = []
    label_combinatos = []
    fames = []
    label_fames = []
    oks = []
    label_oks = []
    sonostufos = []
    label_sonostufos = []
    tantotempos = []
    label_tantotempos = []
    prenderes = []
    label_prenderes = []
    vieniquis = []
    label_vieniquis = []

    
    for file in os.listdir(file_dir):
        name = re.findall("[a-z]+", file)
        if name[0] == 'basta':
            bastas.append(file_dir + file)
            label_bastas.append(0)
        if name[0] == 'buonissimo':
            buonissimos.append(file_dir + file)
            label_buonissimos.append(1)
        if name[0] == 'cheduepalle':
            cheduepalles.append(file_dir + file)
            label_cheduepalles.append(2)
        if name[0] == 'combinato':
            combinatos.append(file_dir + file)
            label_combinatos.append(3)
        if name[0] == 'fame':
            fames.append(file_dir + file)
            label_fames.append(4)
        if name[0] == 'ok':
            oks.append(file_dir + file)
            label_oks.append(5)
        if name[0] == 'sonostufo':
            sonostufos.append(file_dir + file)
            label_sonostufos.append(6)
        if name[0] == 'tantotempo':
            tantotempos.append(file_dir + file)
            label_tantotempos.append(7)
        if name[0] == 'prendere':
            prenderes.append(file_dir + file)
            label_prenderes.append(8)
        if name[0] == 'vieniqui':
            vieniquis.append(file_dir + file)
            label_vieniquis.append(9)

    image_list = np.hstack((bastas,buonissimos,cheduepalles,
                            combinatos, fames,oks,
                            sonostufos,tantotempos,prenderes,
                            vieniquis))
    label_list = np.hstack((label_bastas,label_buonissimos,label_cheduepalles,
                            label_combinatos,label_fames,label_oks,
                            label_sonostufos,label_tantotempos,label_prenderes,
                            label_vieniquis))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.seed(121)
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    
    return image_list, label_list

#%%


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''

    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    ######################################
    # data argumentation should go to here
    ######################################

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    # if you want to test the generated batches of images, you might want to comment the following line.
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)

#    you can also use shuffle_batch
#    image_batch, label_batch = tf.train.shuffle_batch([image,label],
#                                                          batch_size=batch_size,
#                                                          num_threads=64,
#                                                          capacity=capacity,
#                                                          min_after_dequeue=capacity-1)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch

def get_3M_batch(image1, image2, image3, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''

    image1 = tf.cast(image1, tf.string)
    image2 = tf.cast(image2, tf.string)
    image3 = tf.cast(image3, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image1, image2, image3, label])

    label = input_queue[3]
    image_contents1 = tf.read_file(input_queue[0])
    image_contents2 = tf.read_file(input_queue[1])
    image_contents3 = tf.read_file(input_queue[2])
    image1 = tf.image.decode_jpeg(image_contents1, channels=3)
    image2 = tf.image.decode_jpeg(image_contents2, channels=3)
    image3 = tf.image.decode_jpeg(image_contents3, channels=3)

    ######################################
    # data argumentation should go to here
    ######################################

    image1 = tf.image.resize_image_with_crop_or_pad(image1, image_W, image_H)
    image2 = tf.image.resize_image_with_crop_or_pad(image2, image_W, image_H)
    image3 = tf.image.resize_image_with_crop_or_pad(image3, image_W, image_H)
    # if you want to test the generated batches of images, you might want to comment the following line.
    image1 = tf.image.per_image_standardization(image1)
    image2 = tf.image.per_image_standardization(image2)
    image3 = tf.image.per_image_standardization(image3)
    image_batch1, image_batch2, image_batch3, label_batch = tf.train.batch([image1, image2, image3, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)

#    you can also use shuffle_batch
#    image_batch, label_batch = tf.train.shuffle_batch([image,label],
#                                                          batch_size=batch_size,
#                                                          num_threads=64,
#                                                          capacity=capacity,
#                                                          min_after_dequeue=capacity-1)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch1 = tf.cast(image_batch1, tf.float32)
    image_batch2 = tf.cast(image_batch2, tf.float32)
    image_batch3 = tf.cast(image_batch3, tf.float32)

    return image_batch1, image_batch2, image_batch3, label_batch


