
import h5py
import sys
import numpy as np
subset_filepath = "C:/Users/kyeb2/PycharmProjects/project1/CASIA-HWDB1.1-cnn/src/HWDB1.1subset.hdf5"

'''
with h5py.File(subset_filepath, 'r') as f:
    print(f['trn/x'].value.shape)

    #'trn/x': (40000, 1, 64, 64)
    
    #'tst/x': (11947, 1, 64, 64)
    #tst/y : (11947, 200)

    #'vld/x: (8020, 1, 64, 64)
    #vld/y: (8020, 200)
'''

tstlist=[]
with h5py.File(subset_filepath, 'r') as f:
    for i in range(11947):
        data=f['tst/y'].value
        #ind=data.index('1')
        ind=np.where(data[i]==1)[0][0]
        #print(ind,end=',')
        tstlist.append(ind)
        #tstlist.('tstlist.txt')
with open('tstlist.txt', 'w') as f:
    for item in tstlist:
        f.write("%s," % item)
'''
for a in range(200):
    for i in range(200):
        print(i,end=",")
'''