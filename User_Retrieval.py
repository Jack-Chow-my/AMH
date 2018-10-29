#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 17:59:47 2018

@author: jam
"""

# %%
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt
import shutil

bit = 32
return_num = 10

TEST_DIR = './data/dataset/test/depth/'
TRAIN_DIR_1 = './data/dataset/train/color/'
TRAIN_DIR_2 = './data/dataset/train/depth/'
TRAIN_DIR_3 = './data/dataset/train/user/'


# %%

def txt_list1(txtpath):
    fp = open(txtpath)
    arr = []
    for lines in fp.readlines():
        temp1 = lines.split('\n')
        temp2 = "".join(temp1)
        temp3 = temp2.split('/')
        arr.append(temp3[-1])
    fp.close()

    return arr


def txt_list2(txtpath):
    fp = open(txtpath)
    arr = []
    for lines in fp.readlines():
        temp1 = lines.split('\n')
        temp2 = "".join(temp1)
        arr.append(temp2)
    fp.close()

    return arr


# hamming distance of two metrix
def calc_hamming_dist(code_1=None, code_2=None):
    e_num1 = code_1.shape[0]
    e_num2 = code_2.shape[0]
    dist = np.zeros((e_num1, e_num2), dtype=np.uint16)
    for e_ind in np.arange(e_num1).reshape(-1):
        one_pair_feat = np.logical_xor(code_1[e_ind, :], code_2)
        one_dist = np.sum(one_pair_feat, axis=1)
        dist[e_ind, :] = np.uint16(one_dist)

    return dist


# joint two list
def combine(outputList, sortList):
    CombineList = list();
    for index in xrange(len(outputList)):
        CombineList.append((outputList[index], sortList[index]));
    return CombineList


# show image according to image address
def show_img(img_dir):
    img = Image.open(img_dir)
    plt.figure()
    plt.imshow(img)
    plt.show()


# %%

# load dataset 
hashcode_dataset = np.load('./modal/hc_train_{0}.npy'.format(bit))
hashcode_test_m1 = np.load('./modal/hc_test_m1_{0}.npy'.format(bit))
hashcode_test_m2 = np.load('./modal/hc_test_m2_{0}.npy'.format(bit))
hashcode_test_m3 = np.load('./modal/hc_test_m3_{0}.npy'.format(bit))

test_txtpath = r"./modal/color_test_images_address.txt"
train_txtpath = r"./modal/color_images_address.txt"
test_images_labels = txt_list1(test_txtpath)
train_images_labels = txt_list1(train_txtpath)

test_images_address = txt_list2(test_txtpath)
train_images_address = txt_list2(train_txtpath)

# get query image
np.random.seed(22)
ind = np.random.randint(0, 698)

img_dir = TEST_DIR + test_images_labels[ind]
dist_metrix = calc_hamming_dist(hashcode_dataset, hashcode_test_m1[ind].reshape(1, bit))
a = dist_metrix.reshape(2771, )
dist_list = a.tolist()

name_query = re.findall("[a-z]+", test_images_labels[ind])
label_query = name_query[0]

print('Image to be retrieved:')
show_img(img_dir)
shutil.copy(img_dir, "./retrieval_result/retrieved_{0}.jpg".format(label_query))
print('Label:')
print(label_query)

# get return sort result
result = combine(train_images_labels, dist_list)
result.sort(key=lambda x: x[1], reverse=False)
return_result = result[:return_num]

print('\nTOP {0} Search Results:'.format(return_num))

for i in np.arange(return_num):
    a = int(i) + 1
    print(a)
    img_dir1 = TRAIN_DIR_1 + return_result[i][0]
    img_dir2 = TRAIN_DIR_2 + return_result[i][0]
    img_dir3 = TRAIN_DIR_3 + return_result[i][0]
    result_label = re.findall("[a-z]+", return_result[i][0])
    show_img(img_dir1)
    shutil.copy(img_dir1, "./retrieval_result/color_{0}_{1}.jpg".format(i, result_label[0]))
    show_img(img_dir2)
    shutil.copy(img_dir2, "./retrieval_result/depth_{0}_{1}.jpg".format(i, result_label[0]))
    show_img(img_dir3)
    shutil.copy(img_dir3, "./retrieval_result/user_{0}_{1}.jpg".format(i, result_label[0]))
    print(result_label[0])
    print('\n')
