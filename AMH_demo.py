#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 23:36:05 2018

@author: jam
"""

import os
os.system("python ./train_mecnnf.py")
os.system("python ./standardsclar.py")
os.system("python ./train_agrbm.py")

