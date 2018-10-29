#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 04:39:12 2018

@author: jam
"""

# %%
import tensorflow as tf
import numpy as np
import os
import input_data
import model_loss
from ECNNF import net_strucuture, net_strucuture2, net_strucuture3
from PIL import Image
import matplotlib.pyplot as plt

TRAIN_DIR1 = './data/dataset/train/color/'
TRAIN_DIR2 = './data/dataset/train/depth/'
TRAIN_DIR3 = './data/dataset/train/user/'
TEST_DIR1 = './data/dataset/test/color/'
TEST_DIR2 = './data/dataset/test/depth/'
TEST_DIR3 = './data/dataset/test/user/'
logs_train_dir = '.data/modal/logs/'

var = {}
BIT = 256
BATCH_SIZE = 2
CAPACITY = 1000
IMG_W = 224
IMG_H = 224
MAX_STEP = 18000
lr = np.linspace(np.power(10, -2.), np.power(10, -5.), MAX_STEP)
var['lr'] = lr


# %%

def get_image(img_dir):
    image = Image.open(img_dir)
    image = np.array(image)
    return image


# %% train three CNNF at the same time
select_gpu = '0'
per_process_gpu_memory_fraction = 0.7
gpuconfig = tf.ConfigProto(gpu_options=tf.GPUOptions(
    per_process_gpu_memory_fraction=per_process_gpu_memory_fraction))
os.environ["CUDA_VISIBLE_DEVICES"] = select_gpu
with tf.Graph().as_default(), tf.Session(config=gpuconfig) as sess:
    # three inputs
    lr_ = tf.placeholder('float32', (), name='lr')
    y_ = tf.placeholder(tf.int32, shape=[None])
    keep_prob = tf.placeholder("float")

    input_1 = tf.placeholder(tf.float32, (None,) + (224, 224, 3), name='x_input1')
    input_2 = tf.placeholder(tf.float32, (None,) + (224, 224, 3), name='x_input2')
    input_3 = tf.placeholder(tf.float32, (None,) + (224, 224, 3), name='x_input3')

    # modal operation
    output_1, fc9_1 = net_strucuture(input_1, BIT, keep_prob)
    output_2, fc9_2 = net_strucuture2(input_2, BIT, keep_prob)
    output_3, fc9_3 = net_strucuture3(input_3, BIT, keep_prob)

    train_loss1 = model_loss.losses(output_1, y_)
    train_loss2 = model_loss.losses(output_2, y_)
    train_loss3 = model_loss.losses(output_3, y_)

    train_loss = (train_loss1 + train_loss2 + train_loss3) / 3.0
    train_op = model_loss.trainning(train_loss, lr_)

    train__acc1 = model_loss.evaluation(output_1, y_)
    train__acc2 = model_loss.evaluation(output_2, y_)
    train__acc3 = model_loss.evaluation(output_3, y_)

    print('******************** three modal ********************')

    train_list1, train_label_list1 = input_data.get_files(TRAIN_DIR1)
    train_list2, train_label_list2 = input_data.get_files(TRAIN_DIR2)
    train_list3, train_label_list3 = input_data.get_files(TRAIN_DIR3)
    train_sim = len(train_list1)

    train_batch1, train_batch2, train_batch3, train_label_batch = input_data.get_3M_batch(
        train_list1,
        train_list2,
        train_list3,
        train_label_list1,
        IMG_W, IMG_H,
        BATCH_SIZE,
        CAPACITY)
    # get test_batch
    test_list1, test_label_list1 = input_data.get_files(TEST_DIR1)
    test_list2, test_label_list2 = input_data.get_files(TEST_DIR2)
    test_list3, test_label_list3 = input_data.get_files(TEST_DIR3)
    test_sim = len(test_list1)
    test_batch1, test_batch2, test_batch3, test_label_batch = input_data.get_3M_batch(
        test_list1,
        test_list2,
        test_list3,
        test_label_list1,
        IMG_W, IMG_H,
        64,
        CAPACITY)
    sess.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()
    tf.add_to_collection('input1', input_1)
    tf.add_to_collection('input2', input_2)
    tf.add_to_collection('input3', input_3)
    tf.add_to_collection('fc9_1', fc9_1)
    tf.add_to_collection('fc9_2', fc9_2)
    tf.add_to_collection('fc9_3', fc9_3)
    tf.add_to_collection('kp', keep_prob)
    print('^^^^^^^^^^^^^^^^^^^^ all_color_data ^^^^^^^^^^^^^^^^^^^^')
    print("there are %d training simples" % (train_sim))
    print("there are %d test simples" % (test_sim))
    print('^^^^^^^^^^^^^^^^^^^^ start_training ^^^^^^^^^^^^^^^^^^^^')
    # for matplotlib
    STEP = []
    LOSS_TRAIN = []
    LOSS_VAL = []
    ACCURACY_TRAIN1 = []
    ACCURACY_TRAIN2 = []
    ACCURACY_TRAIN3 = []
    ACCURACY_VAL1 = []
    ACCURACY_VAL2 = []
    ACCURACY_VAL3 = []

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        for step in np.arange(MAX_STEP):
            lr1 = var['lr'][step]
            if coord.should_stop():
                break
            tra_images1, tra_images2, tra_images3, tra_labels = sess.run([
                train_batch1,
                train_batch2,
                train_batch3,
                train_label_batch
            ])
            _, tra_loss, tra_acc1, tra_acc2, tra_acc3 = sess.run([
                train_op,
                train_loss,
                train__acc1,
                train__acc2,
                train__acc3
            ],
                feed_dict={
                    input_1: tra_images1,
                    input_2: tra_images2,
                    input_3: tra_images3,
                    y_: tra_labels,
                    keep_prob: 0.5,
                    lr_: lr1
                })
            if step % 100 == 0:
                print(
                    'Step %d,train loss = %.2f,train accuracy1 = %.2f%%,train accuracy2 = %.2f%%,train accuracy3 = %.2f%%' % (
                        step, \
                        tra_loss, \
                        tra_acc1 * 100.0, \
                        tra_acc2 * 100.0, \
                        tra_acc3 * 100.0))
                val_images1, val_images2, val_images3, val_labels = sess.run([
                    test_batch1,
                    test_batch2,
                    test_batch3,
                    test_label_batch
                ])
                val_loss, val_acc1, val_acc2, val_acc3 = sess.run([
                    train_loss,
                    train__acc1,
                    train__acc2,
                    train__acc3
                ],
                    feed_dict={
                        input_1: val_images1,
                        input_2: val_images2,
                        input_3: val_images3,
                        y_: val_labels,
                        keep_prob: 1.0
                    })
                print(
                    '**  Step %d,test loss = %.2f,test accuracy1 = %.2f%%,test accuracy2 = %.2f%%,test accuracy3 = %.2f%% **' % (
                        step, val_loss, val_acc1 * 100.0, val_acc2 * 100.0, val_acc3 * 100.0))
                STEP.append(step)
                LOSS_TRAIN.append(tra_loss)
                LOSS_VAL.append(val_loss)
                ACCURACY_TRAIN1.append(tra_acc1)
                ACCURACY_TRAIN2.append(tra_acc2)
                ACCURACY_TRAIN3.append(tra_acc3)
                ACCURACY_VAL1.append(val_acc1)
                ACCURACY_VAL2.append(val_acc2)
                ACCURACY_VAL3.append(val_acc3)
            #                summary_str = sess.run(summary_op)
            #                train_writer.add_summary(summary_str, step)
            if step % 500 == 0:
                print(lr1)
            if step % 3000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'mecnnf')
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    print('^^^^^^^^^^^^^^^^^^^^ test ^^^^^^^^^^^^^^^^^^^^')
    test_image1, test_image2, test_image3, test_label = sess.run([
        test_batch1,
        test_batch2,
        test_batch3,
        test_label_batch
    ])
    test_loss, test_acc1, test_acc2, test_acc3, = sess.run([
        train_loss,
        train__acc1,
        train__acc2,
        train__acc3
    ],
        feed_dict={
            input_1: test_image1,
            input_2: test_image2,
            input_3: test_image3,
            y_: test_label,
            keep_prob: 1.0})
    print('^^ test loss = %.2f,\
              test accuracy1 = %.2f%%,\
              test accuracy2 = %.2f%%,\
              test accuracy3 = %.2f%% ^^' % (test_loss, \
                                             test_acc1 * 100.0, \
                                             test_acc2 * 100.0, \
                                             test_acc3 * 100.0))

    # %%  get image feature
    # m1_train
    print('************get image feature*************')
    v_color = np.ones([BIT])
    fl1 = open('./modal/color_images_address.txt', 'w')
    for i in np.arange(5600):
        img_dir = train_list1[i]
        fl1.write(img_dir)
        fl1.write("\n")
        image_array = get_image(img_dir)
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 224, 224, 3])
        image = sess.run(image)
        net_modal = sess.run(fc9_1, feed_dict={input_1: image, keep_prob: 1.0})
        v_color = np.row_stack((v_color, net_modal['fc1_9']))
        if i % 100 == 0:
            print(v_color.shape)
    np.save("./modal/m1.npy", v_color)
    fl1.close()
    # m2_train
    v_depth = np.ones([BIT])
    fl1 = open('./modal/depth_images_address.txt', 'w')
    for i in np.arange(5600):
        img_dir = train_list2[i]
        fl1.write(img_dir)
        fl1.write("\n")
        image_array = get_image(img_dir)
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 224, 224, 3])
        image = sess.run(image)
        net_modal = sess.run(fc9_2, feed_dict={input_2: image, keep_prob: 1.0})
        v_depth = np.row_stack((v_depth, net_modal['fc2_9']))
        if i % 100 == 0:
            print(v_depth.shape)
    np.save("./modal/m2.npy", v_depth)
    fl1.close()
    # m3_train
    v_user = np.ones([BIT])
    fl1 = open('./modal/user_images_address.txt', 'w')
    for i in np.arange(5600):
        img_dir = train_list3[i]
        fl1.write(img_dir)
        fl1.write("\n")
        image_array = get_image(img_dir)
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 224, 224, 3])
        image = sess.run(image)
        net_modal = sess.run(fc9_3, feed_dict={input_3: image, keep_prob: 1.0})
        v_user = np.row_stack((v_user, net_modal['fc3_9']))
        if i % 100 == 0:
            print(v_user.shape)
    np.save("./modal/m3.npy", v_user)
    fl1.close()
    # m1_test
    v_color_test = np.ones([BIT])
    fl1 = open('./modal/color_test_images_address.txt', 'w')
    for i in np.arange(1400):
        img_dir = test_list1[i]
        fl1.write(img_dir)
        fl1.write("\n")
        image_array = get_image(img_dir)
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 224, 224, 3])
        image = sess.run(image)
        net_modal = sess.run(fc9_1, feed_dict={input_1: image, keep_prob: 1.0})
        v_color_test = np.row_stack((v_color_test, net_modal['fc1_9']))
        if i % 100 == 0:
            print(v_color_test.shape)
    np.save("./modal/m1_test.npy", v_color_test)
    fl1.close()
    # m2_test
    v_depth_test = np.ones([BIT])
    fl1 = open('./modal/depth_test_images_address.txt', 'w')
    for i in np.arange(1400):
        img_dir = test_list2[i]
        fl1.write(img_dir)
        fl1.write("\n")
        image_array = get_image(img_dir)
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 224, 224, 3])
        image = sess.run(image)
        net_modal = sess.run(fc9_2, feed_dict={input_2: image, keep_prob: 1.0})
        v_depth_test = np.row_stack((v_depth_test, net_modal['fc2_9']))
        if i % 100 == 0:
            print(v_depth_test.shape)
    np.save("./modal/m2_test.npy", v_depth_test)
    fl1.close()

    v_user_test = np.ones([BIT])
    fl1 = open('./modal/user_test_images_address.txt', 'w')
    for i in np.arange(1400):
        img_dir = test_list3[i]
        fl1.write(img_dir)
        fl1.write("\n")
        image_array = get_image(img_dir)
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 224, 224, 3])
        image = sess.run(image)
        net_modal = sess.run(fc9_3, feed_dict={input_3: image, keep_prob: 1.0})
        v_user_test = np.row_stack((v_user_test, net_modal['fc3_9']))
        if i % 100 == 0:
            print(v_user_test.shape)
    np.save("./modal/m3_test.npy", v_user_test)
    fl1.close()

# %%  plot
fig1 = plt.figure(1)
plt.plot(STEP, LOSS_TRAIN, color='green', linestyle='solid', label='train_loss')
plt.plot(STEP, LOSS_VAL, color='red', linestyle='dashed', label='val_loss')
plt.xlabel('steps', fontsize=12)
plt.ylabel('loss', fontsize=12)
plt.title('LOSS', fontsize=14, fontweight='bold')
plt.legend(loc="lower left")

fig2 = plt.figure(2)
plt.plot(STEP, ACCURACY_TRAIN1, color='blue', linestyle='solid', label='train_acc1')
plt.plot(STEP, ACCURACY_VAL1, color='red', linestyle='dashed', label='val_acc1')
plt.xlabel('steps', fontsize=12)
plt.ylabel('m1_accuracy', fontsize=12)
plt.title('m1_ACCURACY', fontsize=14, fontweight='bold')
plt.legend(loc="upper left")

fig3 = plt.figure(3)
plt.plot(STEP, ACCURACY_TRAIN2, color='blue', linestyle='solid', label='train_acc2')
plt.plot(STEP, ACCURACY_VAL2, color='red', linestyle='dashed', label='val_acc2')
plt.xlabel('steps', fontsize=12)
plt.ylabel('accuracy', fontsize=12)
plt.title('m2_ACCURACY', fontsize=14, fontweight='bold')
plt.legend(loc="upper left")

fig4 = plt.figure(4)
plt.plot(STEP, ACCURACY_TRAIN3, color='blue', linestyle='solid', label='train_acc3')
plt.plot(STEP, ACCURACY_VAL3, color='red', linestyle='dashed', label='val_acc3')
plt.xlabel('steps', fontsize=12)
plt.ylabel('accuracy', fontsize=12)
plt.title('m3_ACCURACY', fontsize=14, fontweight='bold')
plt.legend(loc="upper left")
plt.show()


