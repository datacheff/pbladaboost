# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import h5py
import sys
import numpy as np
import pandas as pd
import re
#subset_filepath=sys.argv[1]

subset_filepath = "C:/Users/kyeb2/PycharmProjects/project1/CASIA-HWDB1.1-cnn/src/HWDB1.1subset.hdf5"
with h5py.File(subset_filepath, 'r') as f:
    trnx = []

    for image in range(40000):
        for pixel in (f['trn/x'][image][:]):
            for row in range(64):
                data = list()
                for column in range(64):
                    data.append(int(pixel[row, column]))
                        # print(data)
            trnx.append(data)

    tstx=[]
    for image in range(11946):
        for pixel in (f['tst/x'][image][:]):
            for row in range(64):
                data = list()
                for column in range(64):
                    data.append(int(pixel[row, column]))
                        # print(data)
            tstx.append(data)

    trnlabel=[]
    for a in range(200):
        for b in range(200):
            trnlabel.append(b)
'''
tstlabel=[]
with open('C:/Users/kyeb2/PycharmProjects/project1/CASIA-HWDB1.1-cnn/src/tstlist.txt', 'w') as f:
    for label in f:
        tstlabel.append(f[label])

'''

def loadSimpData():
    datMat = trnx
    classLabels = trnlabel
    return datMat,classLabels



