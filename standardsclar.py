#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:19:28 2018

@author: jam
"""
import numpy as np
from sklearn import preprocessing


# StandardScaler normalazation dataset
def StandardScaler_normalazation(metrix, mertix_test):
    scaler_metrix = preprocessing.StandardScaler().fit(metrix)
    metrix_normalazation = scaler_metrix.transform(metrix)
    metrix_test_normalazation = scaler_metrix.transform(mertix_test)

    return metrix_normalazation, metrix_test_normalazation


# %% StandardScaler normalazation m1/m2/m3
m1 = np.load('./modal/m1.npy')
m1_or = np.delete(m1, 0, axis=0)
m1_test = np.load('./modal/m1_test.npy')
m1_test_or = np.delete(m1_test, 0, axis=0)

scaler_m1 = preprocessing.StandardScaler().fit(m1_or)

m1_nor = scaler_m1.transform(m1_or)
m1_test_nor = scaler_m1.transform(m1_test_or)
np.save("./modal/m1_nor.npy", m1_nor)
np.save("./modal/m1_test_nor.npy", m1_test_nor)

# StandardScaler normalazation m2
m2 = np.load('./modal/m2.npy')
m2_or = np.delete(m2, 0, axis=0)
m2_test = np.load('./modal/m2_test.npy')
m2_test_or = np.delete(m2_test, 0, axis=0)

scaler_m2 = preprocessing.StandardScaler().fit(m2_or)

m2_nor = scaler_m2.transform(m2_or)
m2_test_nor = scaler_m2.transform(m2_test_or)
np.save("./modal/m2_nor.npy", m2_nor)
np.save("./modal/m2_test_nor.npy", m2_test_nor)

# StandardScaler normalazation m3
m3 = np.load('./modal/m3.npy')
m3_or = np.delete(m3, 0, axis=0)
m3_test = np.load('./modal/m3_test.npy')
m3_test_or = np.delete(m3_test, 0, axis=0)

scaler_m3 = preprocessing.StandardScaler().fit(m3_or)

m3_nor = scaler_m3.transform(m3_or)
m3_test_nor = scaler_m3.transform(m3_test_or)
np.save("./modal/m3_nor.npy", m3_nor)
np.save("./modal/m3_test_nor.npy", m3_test_nor)


print(m1_nor.shape)
print(m2_nor.shape)
print(m3_nor.shape)
print(m1_test_nor.shape)
print(m2_test_nor.shape)
print(m3_test_nor.shape)
