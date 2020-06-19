import os, numpy, PIL
from PIL import Image, ImageEnhance
import numpy as np
from PIL import Image
import sys



size=[[20,35],
[20,35],
[20,35],
[20,40],
[20,40],
[20,40],
[20,40],
[20,40],
[20,40],
[25,22],
[25,22],
[25,22]]


def HaarArr ():
    from dataList import haar_sizes
    haar_sizes = haar_sizes()

    imgArr = []

    for i in range(1,34):
        if i == 4 :
            # bugun data
            allfiles = os.listdir('./bugun/' + str(i) + '/')
            imList = ['./bugun/' + str(i) + '/' + filename for filename in allfiles]
            images = np.array([np.array(Image.open(filename)) for filename in imList])
            arr = np.array(np.mean(images, axis=(0)), dtype=np.uint8)
            out = Image.fromarray(arr)
            avg = PIL.ImageEnhance.Contrast(out).enhance(1000)
            im = np.array(avg.convert('L').resize((size[0][0],size[0][1])))

            sys.stdout = open('C:/Users/kyeb2/PycharmProjects/face-detectio/haar1-3.txt', 'w')
            th1 = 30
            im_bin_128_1 = (im > th1) * 255
            #print(im_bin_128) # 0 까만, 255 흰색
            Image.fromarray(np.uint8(im_bin_128_1)).save('./avgImgs1/' + 'avg' + str(i) + '.bmp')
            imgArr.append(im_bin_128_1)


            # th2 = 70
            # im_bin_128_2 = (im > th2) * 255
            # #print(im_bin_128) # 0 까만, 255 흰색
            # Image.fromarray(np.uint8(im_bin_128_2)).save('./avgImgs2/' + 'avg' + str(i) + '.bmp')
            # imgArr.append(im_bin_128_2)
            #
            # th3 = 100
            # im_bin_128_3 = (im > th3) * 255
            # #print(im_bin_128) # 0 까만, 255 흰색
            # Image.fromarray(np.uint8(im_bin_128_3)).save('./avgImgs3/' + 'avg' + str(i) + '.bmp')
            # imgArr.append(im_bin_128_3)

        if i == 11 :

            # bugun data
            allfiles = os.listdir('./bugun/' + str(i) + '/')
            imList = ['./bugun/' + str(i) + '/' + filename for filename in allfiles]
            images = np.array([np.array(Image.open(filename)) for filename in imList])
            arr = np.array(np.mean(images, axis=(0)), dtype=np.uint8)
            out = Image.fromarray(arr)
            avg = PIL.ImageEnhance.Contrast(out).enhance(1000)
            im = np.array(avg.convert('L').resize((size[3][0],size[3][1])))

            th1 = 30
            im_bin_128_1 = (im > th1) * 255
            #print(im_bin_128) # 0 까만, 255 흰색
            Image.fromarray(np.uint8(im_bin_128_1)).save('./avgImgs1/' + 'avg' + str(i) + '.bmp')
            imgArr.append(im_bin_128_1)

            th2 = 70
            im_bin_128_2 = (im > th2) * 255
            #print(im_bin_128) # 0 까만, 255 흰색
            Image.fromarray(np.uint8(im_bin_128_2)).save('./avgImgs2/' + 'avg' + str(i) + '.bmp')
            imgArr.append(im_bin_128_2)

            th3 = 100
            im_bin_128_3 = (im > th3) * 255
            #print(im_bin_128) # 0 까만, 255 흰색
            Image.fromarray(np.uint8(im_bin_128_3)).save('./avgImgs3/' + 'avg' + str(i) + '.bmp')
            imgArr.append(im_bin_128_3)

        if i == 14 :
            # bugun data
            allfiles = os.listdir('./bugun/' + str(i) + '/')
            imList = ['./bugun/' + str(i) + '/' + filename for filename in allfiles]
            images = np.array([np.array(Image.open(filename)) for filename in imList])
            arr = np.array(np.mean(images, axis=(0)), dtype=np.uint8)
            out = Image.fromarray(arr)
            avg = PIL.ImageEnhance.Contrast(out).enhance(1000)
            im = np.array(avg.convert('L').resize((size[6][0],size[6][1])))

            th1 = 30
            im_bin_128_1 = (im > th1) * 255
            #print(im_bin_128) # 0 까만, 255 흰색
            Image.fromarray(np.uint8(im_bin_128_1)).save('./avgImgs1/' + 'avg' + str(i) + '.bmp')
            imgArr.append(im_bin_128_1)

            th2 = 70
            im_bin_128_2 = (im > th2) * 255
            #print(im_bin_128) # 0 까만, 255 흰색
            Image.fromarray(np.uint8(im_bin_128_2)).save('./avgImgs2/' + 'avg' + str(i) + '.bmp')
            imgArr.append(im_bin_128_2)

            th3 = 100
            im_bin_128_3 = (im > th3) * 255
            #print(im_bin_128) # 0 까만, 255 흰색
            Image.fromarray(np.uint8(im_bin_128_3)).save('./avgImgs3/' + 'avg' + str(i) + '.bmp')
            imgArr.append(im_bin_128_3)
        if i == 24 :
            # bugun data
            allfiles = os.listdir('./bugun/' + str(i) + '/')
            imList = ['./bugun/' + str(i) + '/' + filename for filename in allfiles]
            images = np.array([np.array(Image.open(filename)) for filename in imList])
            arr = np.array(np.mean(images, axis=(0)), dtype=np.uint8)
            out = Image.fromarray(arr)
            avg = PIL.ImageEnhance.Contrast(out).enhance(1000)
            im = np.array(avg.convert('L').resize((size[9][0],size[9][1])))

            th1 = 30
            im_bin_128_1 = (im > th1) * 255
            #print(im_bin_128) # 0 까만, 255 흰색
            Image.fromarray(np.uint8(im_bin_128_1)).save('./avgImgs1/' + 'avg' + str(i) + '.bmp')
            imgArr.append(im_bin_128_1)

            th2 = 70
            im_bin_128_2 = (im > th2) * 255
            #print(im_bin_128) # 0 까만, 255 흰색
            Image.fromarray(np.uint8(im_bin_128_2)).save('./avgImgs2/' + 'avg' + str(i) + '.bmp')
            imgArr.append(im_bin_128_2)

            th3 = 100
            im_bin_128_3 = (im > th3) * 255
            #print(im_bin_128) # 0 까만, 255 흰색
            Image.fromarray(np.uint8(im_bin_128_3)).save('./avgImgs3/' + 'avg' + str(i) + '.bmp')
            imgArr.append(im_bin_128_3)
    return imgArr

# 실행
HaarArr = HaarArr ()
# print(HaarArr)
# print(len(HaarArr))
# print(len(size))

# 각 점의 좌표 찍어내기 (Haar)
for i in range(len(HaarArr)) :
    pos = []
    neg = []
    for h1 in range(size[i][1]):
        for w1 in range(size[i][0]):
            if str(HaarArr[i][h1,w1])=='255':
                print('neg' + str(w1) + str(h1) + '=' + 'self.getPixelValInIntegralMat(x+' + str(w1) + '*w' + ',y+' + str(h1) + '*h' + ', w, h, IntegralMat)')
                neg.append('neg' + str(w1) + str(h1))
            else :
                print('pos' + str(w1) + str(h1) + '=' + 'self.getPixelValInIntegralMat(x+' + str(w1) + '*w' + ',y+' + str(h1) + '*h' + ', w, h, IntegralMat)')
                pos.append('pos' + str(w1) + str(h1))
    print('featureVal[feature_index] = ((\\')
    for n in range(len(neg)):
        if n == len(neg)-1 :
            print(str(neg[n]),flush=True)
        else :
            print(str(neg[n]) + '+', end='', flush=True)
    print(")-(")
    for p in range(len(pos)):
        if i == len(pos)-1 :
            print(str(pos[p]), flush=True)
        else :
            print(str(pos[p]) + "+", end='', flush=True)
    area = size[i][0] * size[i][1]
    print(")/("+str(area)+"* w * h))")
    print()
    print("============================================="+str(i+1)+"번째 이미지 끝")
print("")

print("!! HAAR 종료 이제 Detector 시작 !!")

# 각 점의 좌표 찍어내기 (Detector)
for i in range(len(HaarArr)) :
    pos = []
    neg = []
    for h1 in range(size[i][1]):
        for w1 in range(size[i][0]):
            if str(HaarArr[i][h1,w1])=='255':
                print('neg' + str(w1) + str(h1) + '=' + 'self.haar.getPixelValInIntegralMat(x+' + str(w1) + '*w' + ',y+' + str(h1) + '*h' + ', w, h, subWindowImgIntegral)')
                neg.append('neg' + str(w1) + str(h1))
            else :
                print('pos' + str(w1) + str(h1) + '=' + 'self.haar.getPixelValInIntegralMat(x+' + str(w1) + '*w' + ',y+' + str(h1) + '*h' + ', w, h, subWindowImgIntegral)')
                pos.append('pos' + str(w1) + str(h1))
    print('   scaledWindowsMat[window][dimension] = ((\\')
    for n in range(len(neg)):
        if n == len(neg)-1 :
            print(str(neg[n]),flush=True)
        else :
            print(str(neg[n]) + '+', end='', flush=True)
    print(")-(")
    for p in range(len(pos)):
        if i == len(pos)-1 :
            print(str(pos[p]), flush=True)
        else :
            print(str(pos[p]) + "+", end='', flush=True)
    area = size[i][0] * size[i][1]
    print(")/("+str(area)+"* w * h))")
    print()
    print("============================================="+str(i+1)+"번째 이미지 끝")




