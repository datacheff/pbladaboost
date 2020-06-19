# coding=utf-8
import os
import multiprocessing
from haar import Haar
from setting import WINDOW_HEIGHT, WINDOW_WIDTH, TRAIN_FACE, TRAIN_NON_FACE, FEATURE_CACHE_FILE, FACE, NON_FACE
from image import Img
import numpy as np
import threading
import pickle
from model import getModel
from adaboost import Adaboost
from model import calAndSaveModel
from model import loadModel
from features import calAndSaveFeatures


#calAndSaveFeatures()
calAndSaveModel()

#a=getModel()
#print(a.n_estimators)

# for i in range(1,30):
#      print("python faceDetection.py \"./testfinal4/"+ str(i)+".pgm\" --show=True --save=True --saveInfo=False")

#clf = Adaboost(n_estimators=2, debug=True)
#print(clf)
#print(a)
#
# from PIL import Image
# im1 = Image.open('./qwqwee.jpg')
# a= im1.convert('L').save('./qwqwee.pgm')

#from PIL import Image
#im1 = Image.open('./pngdata/aabbdd.').resize((19,19), Image.ANTIALIAS)
# #
# from PIL import Image
# file_list = os.listdir("./test4/")
# i=1
# for filename in file_list:
#      im1 = Image.open("./test4/"+filename).convert('L').save("./testfinal4/"+str(i)+'.pgm')
#      i=i+1
#


# #jpg to pgm
# from PIL import Image
# # #im1 = Image.open('./pngdata/mu.jpg').convert('L').save('mu'+'.pgm')
# file_list = os.listdir("D:/non/")
# for filename in file_list:
#     im1 = Image.open("D:/non/"+filename).resize((10,20), Image.ANTIALIAS).convert('L').save("C:/Users/kyeb2/PycharmProjects/face-detection/train/face/"+filename)
#

#이미
'''
from PIL import Image

im = Image.open('C:/Users/kyeb2/PycharmProjects/project1/face-detection/train/non_face/s.bmp')

im1 = Image.open('./pngdata/火2.png')
im2 = Image.open('./pngdata/火3.png')
im3 = Image.open('./pngdata/火4.png')
im4 = Image.open('./pngdata/火5.png')


a= im1.resize((19,19), Image.ANTIALIAS).convert('L')
a.save('i1'+'.bmp')

b= im2.resize((19,19), Image.ANTIALIAS).convert('L')
b.save('i2'+'.bmp')

c= im3.resize((19,19), Image.ANTIALIAS).convert('L')
c.save('i2'+'.bmp')

d= im4.resize((19,19), Image.ANTIALIAS).convert('L')
d.save('i3'+'.bmp')
'''