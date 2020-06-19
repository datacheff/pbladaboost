# coding=utf-8
from PIL import Image
from matplotlib import image
from setting import WINDOW_WIDTH, WINDOW_HEIGHT, FACE, NON_FACE, TEST_RESULT_PIC, TEST_RESULT_INFO
from image import Img
import  numpy as np
from adaboost import Adaboost
from weakClassifier import WeakClassifier
from haar import Haar

from setting import WINDOW_HEIGHT, WINDOW_WIDTH

class Detector(object):

    def __init__(self, model):

        self.DETECT_START = 1.
        self.DETECT_END   = 8.
        self.DETECT_STEP  = 0.1
        self.DETECT_STEP_FACTOR = 1

        self.haar = Haar(WINDOW_WIDTH, WINDOW_HEIGHT)

        self.model = model
        self.selectedFeatures = [None for i in range(model.n_estimators)]
        self._selectFeatures()

    def detectFace(self, fileName, _show=True, _save=False, _saveInfo=False):
        """
        :param fileName:
        :param _show:
        :param _save:
        :return:
        """
        img = Img(fileName, calIntegral=False)

        #scaledWindows: [[window_x, window_y, window_w, window_h, window_scale],...]
        scaledWindows = []

        for scale in np.arange(self.DETECT_START, self.DETECT_END, self.DETECT_STEP):
            self._detectInDiffScale(scale, img, scaledWindows)

        scaledWindows = np.array(scaledWindows)

        # detect whether the scaledWindows are face
        predWindow = self._detectScaledWindow(scaledWindows, img)


        mostProbWindow = self._getMostProbWindow(predWindow)

        print(mostProbWindow)
        if _show:
            self.show(img.mat, mostProbWindow)
        if _save:
            self.save(img.mat, mostProbWindow, fileName)
        if _saveInfo:
            self.saveProbWindowInfo(mostProbWindow, fileName)

    def show(self, imageMat, faceWindows):
        """show the result of detection
        :param imageMat:
        :param faceWindows:
        :return:
        """
        if faceWindows[0].shape[0] == 0:
            Image.fromarray(imageMat).show()
            return
        for i in range(len(faceWindows)):
            window_x, window_y, window_w, window_h, scale, prob = faceWindows[i]
            self._drawLine(imageMat, int(window_x), int(window_y), int(window_w), int(window_h))

        Image.fromarray(imageMat).show()

    def save(self, imageMat, faceWindows, originFileName):
        if faceWindows[0].shape[0] == 0:
            return
        for i in range(len(faceWindows)):
            window_x, window_y, window_w, window_h, scale, prob = faceWindows[i]
            self._drawLine(imageMat, int(window_x), int(window_y), int(window_w), int(window_h))
        Image.fromarray(imageMat).save((TEST_RESULT_PIC + "detected" +
                                        originFileName.split('/')[-1]).replace("pgm", "bmp") )

    def saveProbWindowInfo(self, window, originFileName):
        with open((TEST_RESULT_INFO  +
                                        originFileName.split('/')[-1]).replace("pgm", "pts"), "w") as f:
            if len(window[0] > 0 ):
                f.write(str(window[0][0]) + " " + str(window[0][1]) + " " + str(window[0][2]) + " " + str(window[0][3]))


    def _selectFeatures(self):
        """ select the features according to Adaboost classifier
        :return: [[haar_type, y, w, h, dimension],...]
        """
        for i in range(self.model.n_estimators):
            self.selectedFeatures[i] = self.haar.features[self.model.weakClassifiers[i].dimension] + [self.model.weakClassifiers[i].dimension]

            print(self.selectedFeatures[i]) #h_type, x, y, w, h, dimension

    def _getMostProbWindow(self, predWindow):
        """ return the most likely one
        :param predWindow:
        :return:
        """
        mostProb = -np.inf
        mostProbWindow = np.array([])
        for i in predWindow:
            if i[-1] > mostProb:
                mostProbWindow = i
                mostProb = i[-1]
        print(mostProbWindow)
        return [mostProbWindow]

    def _drawLine(self, imageMat, x, y, w, h):
        """draw the boundary of the face in the image
        """
        imageMat[y,     x:x+w] = 0
        imageMat[y+h,   x:x+w] = 0
        imageMat[y:y+h, x    ] = 0
        imageMat[y:y+h, x+w  ] = 0

    def _detectInDiffScale(self, scale, img, scaledWindows):
        """
        :param scale:
        :param img:
        :param scaledWindows:
        :return:
        """
        SCALED_WINDOW_WIDTH  = int(WINDOW_WIDTH  * scale)
        SCALED_WINDOW_HEIGHT = int(WINDOW_HEIGHT * scale)

        scaled_window_x_limit = img.WIDTH  - SCALED_WINDOW_WIDTH
        scaled_window_y_limit = img.HEIGHT - SCALED_WINDOW_HEIGHT

        step = int(SCALED_WINDOW_WIDTH/self.DETECT_STEP_FACTOR)

        for x in range(0, scaled_window_x_limit, step):
            for y in range(0, scaled_window_y_limit, step):
                scaledWindows.append((x, y, SCALED_WINDOW_WIDTH, SCALED_WINDOW_HEIGHT, scale))

    def _detectScaledWindow(self, scaledWindows, img):
        """detect each of scaledWindow
        :param scaledWindows:
        :param img:
        :return:
        """
        scaledWindowsMat = np.zeros((scaledWindows.shape[0], len(self.haar.features)), dtype='float32')

        for window in range(scaledWindows.shape[0]):
            window_x, window_y, window_w, window_h, scale = scaledWindows[window]

            window_x, window_y, window_w, window_h = int(window_x), int(window_y), int(window_w), int(window_h)

            subWindowImg          = Img(mat=img.mat[window_y : window_y+window_h, window_x : window_x+window_w])
            subWindowImgIntegral  = subWindowImg.integralMat

            # #normalization
            # sumVal        = sum(sum(subWindowImg.mat[y:y+h, x:x+w]))
            # sqSumVal      = sum(sum(subWindowImg.mat[y:y+h, x:x+w] ** 2))
            # meanVal       = sumVal   / (w * h)
            # sqMeanVal     = sqSumVal / (w * h)
            # normFactorVal = np.sqrt(sqMeanVal - meanVal ** 2)
            #
            # if normFactorVal == 0:
            #     normFactorVal = 1

            for f in range(len(self.selectedFeatures)):
                h_type, x, y, w, h, dimension = self.selectedFeatures[f]
                x, y, w, h = int(x * scale), int(y * scale), int(w * scale), int(h * scale)

                if h_type == "HAAR_TYPE_6":
                    neg00 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 0 * h, w, h, subWindowImgIntegral)
                    neg10 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 0 * h, w, h, subWindowImgIntegral)
                    neg20 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 0 * h, w, h, subWindowImgIntegral)
                    neg30 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 0 * h, w, h, subWindowImgIntegral)
                    neg40 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 0 * h, w, h, subWindowImgIntegral)
                    neg50 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 0 * h, w, h, subWindowImgIntegral)
                    neg60 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 0 * h, w, h, subWindowImgIntegral)
                    neg70 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 0 * h, w, h, subWindowImgIntegral)
                    neg80 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 0 * h, w, h, subWindowImgIntegral)
                    neg90 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 0 * h, w, h, subWindowImgIntegral)
                    neg100 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 0 * h, w, h, subWindowImgIntegral)
                    neg110 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 0 * h, w, h, subWindowImgIntegral)
                    neg120 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 0 * h, w, h, subWindowImgIntegral)
                    neg130 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 0 * h, w, h, subWindowImgIntegral)
                    neg140 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 0 * h, w, h, subWindowImgIntegral)
                    neg150 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 0 * h, w, h, subWindowImgIntegral)
                    neg160 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 0 * h, w, h, subWindowImgIntegral)
                    neg170 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 0 * h, w, h, subWindowImgIntegral)
                    neg180 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 0 * h, w, h, subWindowImgIntegral)
                    neg190 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 0 * h, w, h, subWindowImgIntegral)
                    neg01 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 1 * h, w, h, subWindowImgIntegral)
                    neg11 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 1 * h, w, h, subWindowImgIntegral)
                    neg21 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 1 * h, w, h, subWindowImgIntegral)
                    neg31 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 1 * h, w, h, subWindowImgIntegral)
                    neg41 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 1 * h, w, h, subWindowImgIntegral)
                    neg51 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 1 * h, w, h, subWindowImgIntegral)
                    neg61 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 1 * h, w, h, subWindowImgIntegral)
                    neg71 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 1 * h, w, h, subWindowImgIntegral)
                    neg81 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 1 * h, w, h, subWindowImgIntegral)
                    neg91 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 1 * h, w, h, subWindowImgIntegral)
                    neg101 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 1 * h, w, h, subWindowImgIntegral)
                    neg111 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 1 * h, w, h, subWindowImgIntegral)
                    neg121 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 1 * h, w, h, subWindowImgIntegral)
                    neg131 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 1 * h, w, h, subWindowImgIntegral)
                    neg141 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 1 * h, w, h, subWindowImgIntegral)
                    neg151 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 1 * h, w, h, subWindowImgIntegral)
                    neg161 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 1 * h, w, h, subWindowImgIntegral)
                    neg171 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 1 * h, w, h, subWindowImgIntegral)
                    neg181 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 1 * h, w, h, subWindowImgIntegral)
                    neg191 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 1 * h, w, h, subWindowImgIntegral)
                    neg02 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 2 * h, w, h, subWindowImgIntegral)
                    neg12 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 2 * h, w, h, subWindowImgIntegral)
                    neg22 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 2 * h, w, h, subWindowImgIntegral)
                    neg32 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 2 * h, w, h, subWindowImgIntegral)
                    neg42 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 2 * h, w, h, subWindowImgIntegral)
                    neg52 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 2 * h, w, h, subWindowImgIntegral)
                    neg62 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 2 * h, w, h, subWindowImgIntegral)
                    neg72 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 2 * h, w, h, subWindowImgIntegral)
                    pos82 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 2 * h, w, h, subWindowImgIntegral)
                    pos92 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 2 * h, w, h, subWindowImgIntegral)
                    neg102 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 2 * h, w, h, subWindowImgIntegral)
                    neg112 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 2 * h, w, h, subWindowImgIntegral)
                    neg122 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 2 * h, w, h, subWindowImgIntegral)
                    neg132 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 2 * h, w, h, subWindowImgIntegral)
                    neg142 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 2 * h, w, h, subWindowImgIntegral)
                    neg152 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 2 * h, w, h, subWindowImgIntegral)
                    neg162 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 2 * h, w, h, subWindowImgIntegral)
                    neg172 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 2 * h, w, h, subWindowImgIntegral)
                    neg182 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 2 * h, w, h, subWindowImgIntegral)
                    neg192 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 2 * h, w, h, subWindowImgIntegral)
                    neg03 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 3 * h, w, h, subWindowImgIntegral)
                    neg13 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 3 * h, w, h, subWindowImgIntegral)
                    neg23 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 3 * h, w, h, subWindowImgIntegral)
                    neg33 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 3 * h, w, h, subWindowImgIntegral)
                    neg43 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 3 * h, w, h, subWindowImgIntegral)
                    neg53 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 3 * h, w, h, subWindowImgIntegral)
                    neg63 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 3 * h, w, h, subWindowImgIntegral)
                    pos73 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 3 * h, w, h, subWindowImgIntegral)
                    pos83 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 3 * h, w, h, subWindowImgIntegral)
                    pos93 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 3 * h, w, h, subWindowImgIntegral)
                    pos103 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 3 * h, w, h, subWindowImgIntegral)
                    neg113 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 3 * h, w, h, subWindowImgIntegral)
                    neg123 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 3 * h, w, h, subWindowImgIntegral)
                    neg133 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 3 * h, w, h, subWindowImgIntegral)
                    neg143 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 3 * h, w, h, subWindowImgIntegral)
                    neg153 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 3 * h, w, h, subWindowImgIntegral)
                    neg163 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 3 * h, w, h, subWindowImgIntegral)
                    neg173 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 3 * h, w, h, subWindowImgIntegral)
                    neg183 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 3 * h, w, h, subWindowImgIntegral)
                    neg193 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 3 * h, w, h, subWindowImgIntegral)
                    neg04 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 4 * h, w, h, subWindowImgIntegral)
                    neg14 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 4 * h, w, h, subWindowImgIntegral)
                    neg24 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 4 * h, w, h, subWindowImgIntegral)
                    neg34 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 4 * h, w, h, subWindowImgIntegral)
                    neg44 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 4 * h, w, h, subWindowImgIntegral)
                    neg54 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 4 * h, w, h, subWindowImgIntegral)
                    neg64 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 4 * h, w, h, subWindowImgIntegral)
                    pos74 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 4 * h, w, h, subWindowImgIntegral)
                    pos84 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 4 * h, w, h, subWindowImgIntegral)
                    pos94 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 4 * h, w, h, subWindowImgIntegral)
                    pos104 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 4 * h, w, h, subWindowImgIntegral)
                    neg114 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 4 * h, w, h, subWindowImgIntegral)
                    neg124 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 4 * h, w, h, subWindowImgIntegral)
                    neg134 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 4 * h, w, h, subWindowImgIntegral)
                    neg144 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 4 * h, w, h, subWindowImgIntegral)
                    neg154 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 4 * h, w, h, subWindowImgIntegral)
                    neg164 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 4 * h, w, h, subWindowImgIntegral)
                    neg174 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 4 * h, w, h, subWindowImgIntegral)
                    neg184 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 4 * h, w, h, subWindowImgIntegral)
                    neg194 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 4 * h, w, h, subWindowImgIntegral)
                    neg05 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 5 * h, w, h, subWindowImgIntegral)
                    neg15 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 5 * h, w, h, subWindowImgIntegral)
                    neg25 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 5 * h, w, h, subWindowImgIntegral)
                    neg35 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 5 * h, w, h, subWindowImgIntegral)
                    neg45 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 5 * h, w, h, subWindowImgIntegral)
                    neg55 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 5 * h, w, h, subWindowImgIntegral)
                    pos65 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 5 * h, w, h, subWindowImgIntegral)
                    pos75 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 5 * h, w, h, subWindowImgIntegral)
                    pos85 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 5 * h, w, h, subWindowImgIntegral)
                    pos95 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 5 * h, w, h, subWindowImgIntegral)
                    pos105 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 5 * h, w, h, subWindowImgIntegral)
                    neg115 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 5 * h, w, h, subWindowImgIntegral)
                    neg125 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 5 * h, w, h, subWindowImgIntegral)
                    neg135 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 5 * h, w, h, subWindowImgIntegral)
                    neg145 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 5 * h, w, h, subWindowImgIntegral)
                    neg155 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 5 * h, w, h, subWindowImgIntegral)
                    neg165 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 5 * h, w, h, subWindowImgIntegral)
                    neg175 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 5 * h, w, h, subWindowImgIntegral)
                    neg185 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 5 * h, w, h, subWindowImgIntegral)
                    neg195 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 5 * h, w, h, subWindowImgIntegral)
                    neg06 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 6 * h, w, h, subWindowImgIntegral)
                    neg16 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 6 * h, w, h, subWindowImgIntegral)
                    neg26 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 6 * h, w, h, subWindowImgIntegral)
                    neg36 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 6 * h, w, h, subWindowImgIntegral)
                    neg46 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 6 * h, w, h, subWindowImgIntegral)
                    pos56 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 6 * h, w, h, subWindowImgIntegral)
                    pos66 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 6 * h, w, h, subWindowImgIntegral)
                    pos76 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 6 * h, w, h, subWindowImgIntegral)
                    pos86 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 6 * h, w, h, subWindowImgIntegral)
                    pos96 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 6 * h, w, h, subWindowImgIntegral)
                    pos106 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 6 * h, w, h, subWindowImgIntegral)
                    neg116 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 6 * h, w, h, subWindowImgIntegral)
                    neg126 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 6 * h, w, h, subWindowImgIntegral)
                    neg136 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 6 * h, w, h, subWindowImgIntegral)
                    neg146 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 6 * h, w, h, subWindowImgIntegral)
                    neg156 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 6 * h, w, h, subWindowImgIntegral)
                    neg166 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 6 * h, w, h, subWindowImgIntegral)
                    neg176 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 6 * h, w, h, subWindowImgIntegral)
                    neg186 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 6 * h, w, h, subWindowImgIntegral)
                    neg196 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 6 * h, w, h, subWindowImgIntegral)
                    neg07 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 7 * h, w, h, subWindowImgIntegral)
                    neg17 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 7 * h, w, h, subWindowImgIntegral)
                    neg27 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 7 * h, w, h, subWindowImgIntegral)
                    neg37 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 7 * h, w, h, subWindowImgIntegral)
                    neg47 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 7 * h, w, h, subWindowImgIntegral)
                    pos57 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 7 * h, w, h, subWindowImgIntegral)
                    pos67 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 7 * h, w, h, subWindowImgIntegral)
                    pos77 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 7 * h, w, h, subWindowImgIntegral)
                    pos87 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 7 * h, w, h, subWindowImgIntegral)
                    pos97 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 7 * h, w, h, subWindowImgIntegral)
                    pos107 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 7 * h, w, h, subWindowImgIntegral)
                    pos117 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 7 * h, w, h, subWindowImgIntegral)
                    pos127 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 7 * h, w, h, subWindowImgIntegral)
                    neg137 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 7 * h, w, h, subWindowImgIntegral)
                    neg147 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 7 * h, w, h, subWindowImgIntegral)
                    neg157 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 7 * h, w, h, subWindowImgIntegral)
                    neg167 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 7 * h, w, h, subWindowImgIntegral)
                    neg177 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 7 * h, w, h, subWindowImgIntegral)
                    neg187 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 7 * h, w, h, subWindowImgIntegral)
                    neg197 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 7 * h, w, h, subWindowImgIntegral)
                    neg08 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 8 * h, w, h, subWindowImgIntegral)
                    neg18 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 8 * h, w, h, subWindowImgIntegral)
                    neg28 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 8 * h, w, h, subWindowImgIntegral)
                    neg38 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 8 * h, w, h, subWindowImgIntegral)
                    neg48 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 8 * h, w, h, subWindowImgIntegral)
                    pos58 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 8 * h, w, h, subWindowImgIntegral)
                    pos68 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 8 * h, w, h, subWindowImgIntegral)
                    pos78 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 8 * h, w, h, subWindowImgIntegral)
                    pos88 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 8 * h, w, h, subWindowImgIntegral)
                    pos98 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 8 * h, w, h, subWindowImgIntegral)
                    pos108 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 8 * h, w, h, subWindowImgIntegral)
                    pos118 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 8 * h, w, h, subWindowImgIntegral)
                    pos128 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 8 * h, w, h, subWindowImgIntegral)
                    pos138 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 8 * h, w, h, subWindowImgIntegral)
                    pos148 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 8 * h, w, h, subWindowImgIntegral)
                    neg158 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 8 * h, w, h, subWindowImgIntegral)
                    neg168 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 8 * h, w, h, subWindowImgIntegral)
                    neg178 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 8 * h, w, h, subWindowImgIntegral)
                    neg188 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 8 * h, w, h, subWindowImgIntegral)
                    neg198 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 8 * h, w, h, subWindowImgIntegral)
                    neg09 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 9 * h, w, h, subWindowImgIntegral)
                    neg19 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 9 * h, w, h, subWindowImgIntegral)
                    neg29 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 9 * h, w, h, subWindowImgIntegral)
                    neg39 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 9 * h, w, h, subWindowImgIntegral)
                    pos49 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 9 * h, w, h, subWindowImgIntegral)
                    pos59 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 9 * h, w, h, subWindowImgIntegral)
                    pos69 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 9 * h, w, h, subWindowImgIntegral)
                    pos79 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 9 * h, w, h, subWindowImgIntegral)
                    pos89 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 9 * h, w, h, subWindowImgIntegral)
                    pos99 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 9 * h, w, h, subWindowImgIntegral)
                    pos109 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 9 * h, w, h, subWindowImgIntegral)
                    pos119 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 9 * h, w, h, subWindowImgIntegral)
                    pos129 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 9 * h, w, h, subWindowImgIntegral)
                    pos139 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 9 * h, w, h, subWindowImgIntegral)
                    pos149 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 9 * h, w, h, subWindowImgIntegral)
                    neg159 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 9 * h, w, h, subWindowImgIntegral)
                    neg169 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 9 * h, w, h, subWindowImgIntegral)
                    neg179 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 9 * h, w, h, subWindowImgIntegral)
                    neg189 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 9 * h, w, h, subWindowImgIntegral)
                    neg199 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 9 * h, w, h, subWindowImgIntegral)
                    neg010 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 10 * h, w, h, subWindowImgIntegral)
                    neg110 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 10 * h, w, h, subWindowImgIntegral)
                    neg210 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 10 * h, w, h, subWindowImgIntegral)
                    neg310 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 10 * h, w, h, subWindowImgIntegral)
                    pos410 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 10 * h, w, h, subWindowImgIntegral)
                    pos510 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 10 * h, w, h, subWindowImgIntegral)
                    pos610 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 10 * h, w, h, subWindowImgIntegral)
                    pos710 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 10 * h, w, h, subWindowImgIntegral)
                    pos810 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 10 * h, w, h, subWindowImgIntegral)
                    pos910 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 10 * h, w, h, subWindowImgIntegral)
                    pos1010 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 10 * h, w, h, subWindowImgIntegral)
                    pos1110 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 10 * h, w, h, subWindowImgIntegral)
                    pos1210 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 10 * h, w, h, subWindowImgIntegral)
                    pos1310 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 10 * h, w, h, subWindowImgIntegral)
                    pos1410 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 10 * h, w, h, subWindowImgIntegral)
                    neg1510 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 10 * h, w, h, subWindowImgIntegral)
                    neg1610 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 10 * h, w, h, subWindowImgIntegral)
                    neg1710 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 10 * h, w, h, subWindowImgIntegral)
                    neg1810 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 10 * h, w, h, subWindowImgIntegral)
                    neg1910 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 10 * h, w, h, subWindowImgIntegral)
                    neg011 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 11 * h, w, h, subWindowImgIntegral)
                    neg111 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 11 * h, w, h, subWindowImgIntegral)
                    neg211 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 11 * h, w, h, subWindowImgIntegral)
                    neg311 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 11 * h, w, h, subWindowImgIntegral)
                    pos411 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 11 * h, w, h, subWindowImgIntegral)
                    pos511 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 11 * h, w, h, subWindowImgIntegral)
                    pos611 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 11 * h, w, h, subWindowImgIntegral)
                    pos711 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 11 * h, w, h, subWindowImgIntegral)
                    pos811 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 11 * h, w, h, subWindowImgIntegral)
                    pos911 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 11 * h, w, h, subWindowImgIntegral)
                    pos1011 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 11 * h, w, h, subWindowImgIntegral)
                    pos1111 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 11 * h, w, h, subWindowImgIntegral)
                    pos1211 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 11 * h, w, h, subWindowImgIntegral)
                    pos1311 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 11 * h, w, h, subWindowImgIntegral)
                    pos1411 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 11 * h, w, h, subWindowImgIntegral)
                    neg1511 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 11 * h, w, h, subWindowImgIntegral)
                    neg1611 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 11 * h, w, h, subWindowImgIntegral)
                    neg1711 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 11 * h, w, h, subWindowImgIntegral)
                    neg1811 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 11 * h, w, h, subWindowImgIntegral)
                    neg1911 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 11 * h, w, h, subWindowImgIntegral)
                    neg012 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 12 * h, w, h, subWindowImgIntegral)
                    neg112 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 12 * h, w, h, subWindowImgIntegral)
                    neg212 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 12 * h, w, h, subWindowImgIntegral)
                    neg312 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 12 * h, w, h, subWindowImgIntegral)
                    pos412 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 12 * h, w, h, subWindowImgIntegral)
                    pos512 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 12 * h, w, h, subWindowImgIntegral)
                    pos612 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 12 * h, w, h, subWindowImgIntegral)
                    pos712 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 12 * h, w, h, subWindowImgIntegral)
                    pos812 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 12 * h, w, h, subWindowImgIntegral)
                    pos912 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 12 * h, w, h, subWindowImgIntegral)
                    pos1012 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 12 * h, w, h, subWindowImgIntegral)
                    pos1112 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 12 * h, w, h, subWindowImgIntegral)
                    pos1212 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 12 * h, w, h, subWindowImgIntegral)
                    pos1312 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 12 * h, w, h, subWindowImgIntegral)
                    neg1412 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 12 * h, w, h, subWindowImgIntegral)
                    neg1512 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 12 * h, w, h, subWindowImgIntegral)
                    neg1612 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 12 * h, w, h, subWindowImgIntegral)
                    neg1712 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 12 * h, w, h, subWindowImgIntegral)
                    neg1812 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 12 * h, w, h, subWindowImgIntegral)
                    neg1912 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 12 * h, w, h, subWindowImgIntegral)
                    neg013 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 13 * h, w, h, subWindowImgIntegral)
                    neg113 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 13 * h, w, h, subWindowImgIntegral)
                    neg213 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 13 * h, w, h, subWindowImgIntegral)
                    pos313 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 13 * h, w, h, subWindowImgIntegral)
                    pos413 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 13 * h, w, h, subWindowImgIntegral)
                    pos513 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 13 * h, w, h, subWindowImgIntegral)
                    pos613 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 13 * h, w, h, subWindowImgIntegral)
                    pos713 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 13 * h, w, h, subWindowImgIntegral)
                    pos813 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 13 * h, w, h, subWindowImgIntegral)
                    pos913 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 13 * h, w, h, subWindowImgIntegral)
                    pos1013 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 13 * h, w, h, subWindowImgIntegral)
                    pos1113 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 13 * h, w, h, subWindowImgIntegral)
                    pos1213 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 13 * h, w, h, subWindowImgIntegral)
                    pos1313 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 13 * h, w, h, subWindowImgIntegral)
                    neg1413 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 13 * h, w, h, subWindowImgIntegral)
                    neg1513 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 13 * h, w, h, subWindowImgIntegral)
                    neg1613 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 13 * h, w, h, subWindowImgIntegral)
                    neg1713 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 13 * h, w, h, subWindowImgIntegral)
                    neg1813 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 13 * h, w, h, subWindowImgIntegral)
                    neg1913 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 13 * h, w, h, subWindowImgIntegral)
                    neg014 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 14 * h, w, h, subWindowImgIntegral)
                    neg114 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 14 * h, w, h, subWindowImgIntegral)
                    neg214 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 14 * h, w, h, subWindowImgIntegral)
                    pos314 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 14 * h, w, h, subWindowImgIntegral)
                    pos414 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 14 * h, w, h, subWindowImgIntegral)
                    pos514 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 14 * h, w, h, subWindowImgIntegral)
                    pos614 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 14 * h, w, h, subWindowImgIntegral)
                    pos714 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 14 * h, w, h, subWindowImgIntegral)
                    pos814 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 14 * h, w, h, subWindowImgIntegral)
                    pos914 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 14 * h, w, h, subWindowImgIntegral)
                    pos1014 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 14 * h, w, h, subWindowImgIntegral)
                    pos1114 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 14 * h, w, h, subWindowImgIntegral)
                    pos1214 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 14 * h, w, h, subWindowImgIntegral)
                    pos1314 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 14 * h, w, h, subWindowImgIntegral)
                    neg1414 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 14 * h, w, h, subWindowImgIntegral)
                    neg1514 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 14 * h, w, h, subWindowImgIntegral)
                    neg1614 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 14 * h, w, h, subWindowImgIntegral)
                    neg1714 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 14 * h, w, h, subWindowImgIntegral)
                    neg1814 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 14 * h, w, h, subWindowImgIntegral)
                    neg1914 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 14 * h, w, h, subWindowImgIntegral)
                    neg015 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 15 * h, w, h, subWindowImgIntegral)
                    neg115 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 15 * h, w, h, subWindowImgIntegral)
                    neg215 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 15 * h, w, h, subWindowImgIntegral)
                    pos315 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 15 * h, w, h, subWindowImgIntegral)
                    neg415 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 15 * h, w, h, subWindowImgIntegral)
                    pos515 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 15 * h, w, h, subWindowImgIntegral)
                    pos615 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 15 * h, w, h, subWindowImgIntegral)
                    pos715 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 15 * h, w, h, subWindowImgIntegral)
                    pos815 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 15 * h, w, h, subWindowImgIntegral)
                    pos915 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 15 * h, w, h, subWindowImgIntegral)
                    pos1015 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 15 * h, w, h, subWindowImgIntegral)
                    pos1115 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 15 * h, w, h, subWindowImgIntegral)
                    pos1215 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 15 * h, w, h, subWindowImgIntegral)
                    pos1315 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 15 * h, w, h, subWindowImgIntegral)
                    pos1415 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 15 * h, w, h, subWindowImgIntegral)
                    neg1515 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 15 * h, w, h, subWindowImgIntegral)
                    neg1615 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 15 * h, w, h, subWindowImgIntegral)
                    neg1715 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 15 * h, w, h, subWindowImgIntegral)
                    neg1815 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 15 * h, w, h, subWindowImgIntegral)
                    neg1915 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 15 * h, w, h, subWindowImgIntegral)
                    neg016 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 16 * h, w, h, subWindowImgIntegral)
                    neg116 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 16 * h, w, h, subWindowImgIntegral)
                    neg216 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 16 * h, w, h, subWindowImgIntegral)
                    neg316 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 16 * h, w, h, subWindowImgIntegral)
                    neg416 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 16 * h, w, h, subWindowImgIntegral)
                    pos516 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 16 * h, w, h, subWindowImgIntegral)
                    pos616 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 16 * h, w, h, subWindowImgIntegral)
                    neg716 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 16 * h, w, h, subWindowImgIntegral)
                    neg816 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 16 * h, w, h, subWindowImgIntegral)
                    pos916 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 16 * h, w, h, subWindowImgIntegral)
                    pos1016 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 16 * h, w, h, subWindowImgIntegral)
                    pos1116 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 16 * h, w, h, subWindowImgIntegral)
                    pos1216 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 16 * h, w, h, subWindowImgIntegral)
                    pos1316 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 16 * h, w, h, subWindowImgIntegral)
                    neg1416 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 16 * h, w, h, subWindowImgIntegral)
                    neg1516 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 16 * h, w, h, subWindowImgIntegral)
                    neg1616 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 16 * h, w, h, subWindowImgIntegral)
                    neg1716 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 16 * h, w, h, subWindowImgIntegral)
                    neg1816 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 16 * h, w, h, subWindowImgIntegral)
                    neg1916 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 16 * h, w, h, subWindowImgIntegral)
                    neg017 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 17 * h, w, h, subWindowImgIntegral)
                    neg117 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 17 * h, w, h, subWindowImgIntegral)
                    neg217 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 17 * h, w, h, subWindowImgIntegral)
                    neg317 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 17 * h, w, h, subWindowImgIntegral)
                    pos417 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 17 * h, w, h, subWindowImgIntegral)
                    pos517 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 17 * h, w, h, subWindowImgIntegral)
                    pos617 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 17 * h, w, h, subWindowImgIntegral)
                    neg717 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 17 * h, w, h, subWindowImgIntegral)
                    pos817 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 17 * h, w, h, subWindowImgIntegral)
                    pos917 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 17 * h, w, h, subWindowImgIntegral)
                    pos1017 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 17 * h, w, h, subWindowImgIntegral)
                    pos1117 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 17 * h, w, h, subWindowImgIntegral)
                    pos1217 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 17 * h, w, h, subWindowImgIntegral)
                    pos1317 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 17 * h, w, h, subWindowImgIntegral)
                    neg1417 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 17 * h, w, h, subWindowImgIntegral)
                    neg1517 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 17 * h, w, h, subWindowImgIntegral)
                    neg1617 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 17 * h, w, h, subWindowImgIntegral)
                    neg1717 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 17 * h, w, h, subWindowImgIntegral)
                    neg1817 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 17 * h, w, h, subWindowImgIntegral)
                    neg1917 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 17 * h, w, h, subWindowImgIntegral)
                    neg018 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 18 * h, w, h, subWindowImgIntegral)
                    neg118 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 18 * h, w, h, subWindowImgIntegral)
                    neg218 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 18 * h, w, h, subWindowImgIntegral)
                    pos318 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 18 * h, w, h, subWindowImgIntegral)
                    pos418 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 18 * h, w, h, subWindowImgIntegral)
                    pos518 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 18 * h, w, h, subWindowImgIntegral)
                    pos618 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 18 * h, w, h, subWindowImgIntegral)
                    pos718 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 18 * h, w, h, subWindowImgIntegral)
                    pos818 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 18 * h, w, h, subWindowImgIntegral)
                    pos918 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 18 * h, w, h, subWindowImgIntegral)
                    pos1018 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 18 * h, w, h, subWindowImgIntegral)
                    pos1118 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 18 * h, w, h, subWindowImgIntegral)
                    pos1218 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 18 * h, w, h, subWindowImgIntegral)
                    pos1318 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 18 * h, w, h, subWindowImgIntegral)
                    neg1418 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 18 * h, w, h, subWindowImgIntegral)
                    neg1518 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 18 * h, w, h, subWindowImgIntegral)
                    neg1618 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 18 * h, w, h, subWindowImgIntegral)
                    neg1718 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 18 * h, w, h, subWindowImgIntegral)
                    neg1818 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 18 * h, w, h, subWindowImgIntegral)
                    neg1918 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 18 * h, w, h, subWindowImgIntegral)
                    neg019 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 19 * h, w, h, subWindowImgIntegral)
                    neg119 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 19 * h, w, h, subWindowImgIntegral)
                    pos219 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 19 * h, w, h, subWindowImgIntegral)
                    pos319 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 19 * h, w, h, subWindowImgIntegral)
                    pos419 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 19 * h, w, h, subWindowImgIntegral)
                    pos519 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 19 * h, w, h, subWindowImgIntegral)
                    pos619 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 19 * h, w, h, subWindowImgIntegral)
                    pos719 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 19 * h, w, h, subWindowImgIntegral)
                    pos819 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 19 * h, w, h, subWindowImgIntegral)
                    pos919 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 19 * h, w, h, subWindowImgIntegral)
                    pos1019 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 19 * h, w, h, subWindowImgIntegral)
                    pos1119 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 19 * h, w, h, subWindowImgIntegral)
                    pos1219 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 19 * h, w, h, subWindowImgIntegral)
                    pos1319 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 19 * h, w, h, subWindowImgIntegral)
                    neg1419 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 19 * h, w, h, subWindowImgIntegral)
                    neg1519 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 19 * h, w, h, subWindowImgIntegral)
                    neg1619 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 19 * h, w, h, subWindowImgIntegral)
                    neg1719 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 19 * h, w, h, subWindowImgIntegral)
                    neg1819 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 19 * h, w, h, subWindowImgIntegral)
                    neg1919 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 19 * h, w, h, subWindowImgIntegral)
                    neg020 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 20 * h, w, h, subWindowImgIntegral)
                    neg120 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 20 * h, w, h, subWindowImgIntegral)
                    pos220 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 20 * h, w, h, subWindowImgIntegral)
                    pos320 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 20 * h, w, h, subWindowImgIntegral)
                    pos420 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 20 * h, w, h, subWindowImgIntegral)
                    pos520 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 20 * h, w, h, subWindowImgIntegral)
                    pos620 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 20 * h, w, h, subWindowImgIntegral)
                    pos720 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 20 * h, w, h, subWindowImgIntegral)
                    pos820 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 20 * h, w, h, subWindowImgIntegral)
                    pos920 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 20 * h, w, h, subWindowImgIntegral)
                    pos1020 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 20 * h, w, h, subWindowImgIntegral)
                    pos1120 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 20 * h, w, h, subWindowImgIntegral)
                    pos1220 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 20 * h, w, h, subWindowImgIntegral)
                    neg1320 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 20 * h, w, h, subWindowImgIntegral)
                    neg1420 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 20 * h, w, h, subWindowImgIntegral)
                    neg1520 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 20 * h, w, h, subWindowImgIntegral)
                    neg1620 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 20 * h, w, h, subWindowImgIntegral)
                    neg1720 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 20 * h, w, h, subWindowImgIntegral)
                    neg1820 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 20 * h, w, h, subWindowImgIntegral)
                    neg1920 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 20 * h, w, h, subWindowImgIntegral)
                    neg021 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 21 * h, w, h, subWindowImgIntegral)
                    neg121 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 21 * h, w, h, subWindowImgIntegral)
                    pos221 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 21 * h, w, h, subWindowImgIntegral)
                    pos321 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 21 * h, w, h, subWindowImgIntegral)
                    pos421 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 21 * h, w, h, subWindowImgIntegral)
                    pos521 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 21 * h, w, h, subWindowImgIntegral)
                    pos621 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 21 * h, w, h, subWindowImgIntegral)
                    pos721 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 21 * h, w, h, subWindowImgIntegral)
                    pos821 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 21 * h, w, h, subWindowImgIntegral)
                    pos921 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 21 * h, w, h, subWindowImgIntegral)
                    pos1021 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 21 * h, w, h, subWindowImgIntegral)
                    pos1121 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 21 * h, w, h, subWindowImgIntegral)
                    pos1221 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 21 * h, w, h, subWindowImgIntegral)
                    neg1321 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 21 * h, w, h, subWindowImgIntegral)
                    neg1421 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 21 * h, w, h, subWindowImgIntegral)
                    neg1521 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 21 * h, w, h, subWindowImgIntegral)
                    neg1621 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 21 * h, w, h, subWindowImgIntegral)
                    neg1721 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 21 * h, w, h, subWindowImgIntegral)
                    neg1821 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 21 * h, w, h, subWindowImgIntegral)
                    neg1921 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 21 * h, w, h, subWindowImgIntegral)
                    neg022 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 22 * h, w, h, subWindowImgIntegral)
                    pos122 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 22 * h, w, h, subWindowImgIntegral)
                    pos222 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 22 * h, w, h, subWindowImgIntegral)
                    pos322 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 22 * h, w, h, subWindowImgIntegral)
                    pos422 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 22 * h, w, h, subWindowImgIntegral)
                    pos522 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 22 * h, w, h, subWindowImgIntegral)
                    pos622 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 22 * h, w, h, subWindowImgIntegral)
                    pos722 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 22 * h, w, h, subWindowImgIntegral)
                    pos822 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 22 * h, w, h, subWindowImgIntegral)
                    pos922 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 22 * h, w, h, subWindowImgIntegral)
                    pos1022 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 22 * h, w, h, subWindowImgIntegral)
                    pos1122 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 22 * h, w, h, subWindowImgIntegral)
                    pos1222 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 22 * h, w, h, subWindowImgIntegral)
                    neg1322 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 22 * h, w, h, subWindowImgIntegral)
                    neg1422 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 22 * h, w, h, subWindowImgIntegral)
                    neg1522 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 22 * h, w, h, subWindowImgIntegral)
                    neg1622 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 22 * h, w, h, subWindowImgIntegral)
                    neg1722 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 22 * h, w, h, subWindowImgIntegral)
                    neg1822 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 22 * h, w, h, subWindowImgIntegral)
                    neg1922 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 22 * h, w, h, subWindowImgIntegral)
                    neg023 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 23 * h, w, h, subWindowImgIntegral)
                    pos123 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 23 * h, w, h, subWindowImgIntegral)
                    pos223 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 23 * h, w, h, subWindowImgIntegral)
                    pos323 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 23 * h, w, h, subWindowImgIntegral)
                    pos423 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 23 * h, w, h, subWindowImgIntegral)
                    pos523 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 23 * h, w, h, subWindowImgIntegral)
                    pos623 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 23 * h, w, h, subWindowImgIntegral)
                    pos723 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 23 * h, w, h, subWindowImgIntegral)
                    pos823 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 23 * h, w, h, subWindowImgIntegral)
                    pos923 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 23 * h, w, h, subWindowImgIntegral)
                    pos1023 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 23 * h, w, h, subWindowImgIntegral)
                    pos1123 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 23 * h, w, h, subWindowImgIntegral)
                    pos1223 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 23 * h, w, h, subWindowImgIntegral)
                    pos1323 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 23 * h, w, h, subWindowImgIntegral)
                    neg1423 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 23 * h, w, h, subWindowImgIntegral)
                    neg1523 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 23 * h, w, h, subWindowImgIntegral)
                    neg1623 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 23 * h, w, h, subWindowImgIntegral)
                    neg1723 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 23 * h, w, h, subWindowImgIntegral)
                    neg1823 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 23 * h, w, h, subWindowImgIntegral)
                    neg1923 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 23 * h, w, h, subWindowImgIntegral)
                    neg024 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 24 * h, w, h, subWindowImgIntegral)
                    pos124 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 24 * h, w, h, subWindowImgIntegral)
                    neg224 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 24 * h, w, h, subWindowImgIntegral)
                    neg324 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 24 * h, w, h, subWindowImgIntegral)
                    pos424 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 24 * h, w, h, subWindowImgIntegral)
                    pos524 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 24 * h, w, h, subWindowImgIntegral)
                    pos624 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 24 * h, w, h, subWindowImgIntegral)
                    pos724 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 24 * h, w, h, subWindowImgIntegral)
                    pos824 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 24 * h, w, h, subWindowImgIntegral)
                    pos924 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 24 * h, w, h, subWindowImgIntegral)
                    pos1024 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 24 * h, w, h, subWindowImgIntegral)
                    pos1124 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 24 * h, w, h, subWindowImgIntegral)
                    pos1224 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 24 * h, w, h, subWindowImgIntegral)
                    pos1324 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 24 * h, w, h, subWindowImgIntegral)
                    neg1424 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 24 * h, w, h, subWindowImgIntegral)
                    neg1524 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 24 * h, w, h, subWindowImgIntegral)
                    neg1624 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 24 * h, w, h, subWindowImgIntegral)
                    neg1724 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 24 * h, w, h, subWindowImgIntegral)
                    neg1824 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 24 * h, w, h, subWindowImgIntegral)
                    neg1924 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 24 * h, w, h, subWindowImgIntegral)
                    neg025 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 25 * h, w, h, subWindowImgIntegral)
                    neg125 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 25 * h, w, h, subWindowImgIntegral)
                    neg225 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 25 * h, w, h, subWindowImgIntegral)
                    neg325 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 25 * h, w, h, subWindowImgIntegral)
                    neg425 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 25 * h, w, h, subWindowImgIntegral)
                    pos525 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 25 * h, w, h, subWindowImgIntegral)
                    pos625 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 25 * h, w, h, subWindowImgIntegral)
                    pos725 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 25 * h, w, h, subWindowImgIntegral)
                    pos825 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 25 * h, w, h, subWindowImgIntegral)
                    pos925 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 25 * h, w, h, subWindowImgIntegral)
                    pos1025 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 25 * h, w, h, subWindowImgIntegral)
                    pos1125 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 25 * h, w, h, subWindowImgIntegral)
                    pos1225 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 25 * h, w, h, subWindowImgIntegral)
                    pos1325 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 25 * h, w, h, subWindowImgIntegral)
                    pos1425 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 25 * h, w, h, subWindowImgIntegral)
                    neg1525 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 25 * h, w, h, subWindowImgIntegral)
                    neg1625 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 25 * h, w, h, subWindowImgIntegral)
                    neg1725 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 25 * h, w, h, subWindowImgIntegral)
                    neg1825 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 25 * h, w, h, subWindowImgIntegral)
                    neg1925 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 25 * h, w, h, subWindowImgIntegral)
                    neg026 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 26 * h, w, h, subWindowImgIntegral)
                    neg126 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 26 * h, w, h, subWindowImgIntegral)
                    neg226 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 26 * h, w, h, subWindowImgIntegral)
                    neg326 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 26 * h, w, h, subWindowImgIntegral)
                    neg426 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 26 * h, w, h, subWindowImgIntegral)
                    pos526 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 26 * h, w, h, subWindowImgIntegral)
                    pos626 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 26 * h, w, h, subWindowImgIntegral)
                    pos726 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 26 * h, w, h, subWindowImgIntegral)
                    pos826 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 26 * h, w, h, subWindowImgIntegral)
                    pos926 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 26 * h, w, h, subWindowImgIntegral)
                    pos1026 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 26 * h, w, h, subWindowImgIntegral)
                    pos1126 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 26 * h, w, h, subWindowImgIntegral)
                    pos1226 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 26 * h, w, h, subWindowImgIntegral)
                    pos1326 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 26 * h, w, h, subWindowImgIntegral)
                    pos1426 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 26 * h, w, h, subWindowImgIntegral)
                    pos1526 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 26 * h, w, h, subWindowImgIntegral)
                    neg1626 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 26 * h, w, h, subWindowImgIntegral)
                    neg1726 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 26 * h, w, h, subWindowImgIntegral)
                    neg1826 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 26 * h, w, h, subWindowImgIntegral)
                    neg1926 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 26 * h, w, h, subWindowImgIntegral)
                    neg027 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 27 * h, w, h, subWindowImgIntegral)
                    neg127 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 27 * h, w, h, subWindowImgIntegral)
                    neg227 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 27 * h, w, h, subWindowImgIntegral)
                    neg327 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 27 * h, w, h, subWindowImgIntegral)
                    neg427 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 27 * h, w, h, subWindowImgIntegral)
                    pos527 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 27 * h, w, h, subWindowImgIntegral)
                    pos627 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 27 * h, w, h, subWindowImgIntegral)
                    pos727 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 27 * h, w, h, subWindowImgIntegral)
                    pos827 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 27 * h, w, h, subWindowImgIntegral)
                    pos927 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 27 * h, w, h, subWindowImgIntegral)
                    pos1027 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 27 * h, w, h, subWindowImgIntegral)
                    pos1127 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 27 * h, w, h, subWindowImgIntegral)
                    pos1227 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 27 * h, w, h, subWindowImgIntegral)
                    pos1327 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 27 * h, w, h, subWindowImgIntegral)
                    pos1427 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 27 * h, w, h, subWindowImgIntegral)
                    pos1527 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 27 * h, w, h, subWindowImgIntegral)
                    pos1627 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 27 * h, w, h, subWindowImgIntegral)
                    neg1727 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 27 * h, w, h, subWindowImgIntegral)
                    neg1827 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 27 * h, w, h, subWindowImgIntegral)
                    neg1927 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 27 * h, w, h, subWindowImgIntegral)
                    neg028 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 28 * h, w, h, subWindowImgIntegral)
                    neg128 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 28 * h, w, h, subWindowImgIntegral)
                    neg228 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 28 * h, w, h, subWindowImgIntegral)
                    neg328 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 28 * h, w, h, subWindowImgIntegral)
                    neg428 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 28 * h, w, h, subWindowImgIntegral)
                    pos528 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 28 * h, w, h, subWindowImgIntegral)
                    pos628 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 28 * h, w, h, subWindowImgIntegral)
                    pos728 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 28 * h, w, h, subWindowImgIntegral)
                    pos828 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 28 * h, w, h, subWindowImgIntegral)
                    neg928 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 28 * h, w, h, subWindowImgIntegral)
                    neg1028 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 28 * h, w, h, subWindowImgIntegral)
                    neg1128 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 28 * h, w, h, subWindowImgIntegral)
                    pos1228 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 28 * h, w, h, subWindowImgIntegral)
                    pos1328 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 28 * h, w, h, subWindowImgIntegral)
                    pos1428 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 28 * h, w, h, subWindowImgIntegral)
                    pos1528 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 28 * h, w, h, subWindowImgIntegral)
                    pos1628 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 28 * h, w, h, subWindowImgIntegral)
                    neg1728 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 28 * h, w, h, subWindowImgIntegral)
                    neg1828 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 28 * h, w, h, subWindowImgIntegral)
                    neg1928 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 28 * h, w, h, subWindowImgIntegral)
                    neg029 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 29 * h, w, h, subWindowImgIntegral)
                    neg129 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 29 * h, w, h, subWindowImgIntegral)
                    neg229 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 29 * h, w, h, subWindowImgIntegral)
                    neg329 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 29 * h, w, h, subWindowImgIntegral)
                    pos429 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 29 * h, w, h, subWindowImgIntegral)
                    pos529 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 29 * h, w, h, subWindowImgIntegral)
                    pos629 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 29 * h, w, h, subWindowImgIntegral)
                    pos729 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 29 * h, w, h, subWindowImgIntegral)
                    neg829 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 29 * h, w, h, subWindowImgIntegral)
                    neg929 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 29 * h, w, h, subWindowImgIntegral)
                    neg1029 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 29 * h, w, h, subWindowImgIntegral)
                    neg1129 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 29 * h, w, h, subWindowImgIntegral)
                    pos1229 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 29 * h, w, h, subWindowImgIntegral)
                    pos1329 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 29 * h, w, h, subWindowImgIntegral)
                    pos1429 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 29 * h, w, h, subWindowImgIntegral)
                    pos1529 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 29 * h, w, h, subWindowImgIntegral)
                    pos1629 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 29 * h, w, h, subWindowImgIntegral)
                    pos1729 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 29 * h, w, h, subWindowImgIntegral)
                    neg1829 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 29 * h, w, h, subWindowImgIntegral)
                    neg1929 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 29 * h, w, h, subWindowImgIntegral)
                    neg030 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 30 * h, w, h, subWindowImgIntegral)
                    neg130 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 30 * h, w, h, subWindowImgIntegral)
                    neg230 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 30 * h, w, h, subWindowImgIntegral)
                    neg330 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 30 * h, w, h, subWindowImgIntegral)
                    pos430 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 30 * h, w, h, subWindowImgIntegral)
                    pos530 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 30 * h, w, h, subWindowImgIntegral)
                    pos630 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 30 * h, w, h, subWindowImgIntegral)
                    neg730 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 30 * h, w, h, subWindowImgIntegral)
                    neg830 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 30 * h, w, h, subWindowImgIntegral)
                    neg930 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 30 * h, w, h, subWindowImgIntegral)
                    neg1030 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 30 * h, w, h, subWindowImgIntegral)
                    neg1130 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 30 * h, w, h, subWindowImgIntegral)
                    neg1230 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 30 * h, w, h, subWindowImgIntegral)
                    neg1330 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 30 * h, w, h, subWindowImgIntegral)
                    pos1430 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 30 * h, w, h, subWindowImgIntegral)
                    pos1530 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 30 * h, w, h, subWindowImgIntegral)
                    pos1630 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 30 * h, w, h, subWindowImgIntegral)
                    pos1730 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 30 * h, w, h, subWindowImgIntegral)
                    neg1830 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 30 * h, w, h, subWindowImgIntegral)
                    neg1930 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 30 * h, w, h, subWindowImgIntegral)
                    neg031 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 31 * h, w, h, subWindowImgIntegral)
                    neg131 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 31 * h, w, h, subWindowImgIntegral)
                    neg231 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 31 * h, w, h, subWindowImgIntegral)
                    pos331 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 31 * h, w, h, subWindowImgIntegral)
                    pos431 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 31 * h, w, h, subWindowImgIntegral)
                    neg531 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 31 * h, w, h, subWindowImgIntegral)
                    neg631 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 31 * h, w, h, subWindowImgIntegral)
                    neg731 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 31 * h, w, h, subWindowImgIntegral)
                    neg831 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 31 * h, w, h, subWindowImgIntegral)
                    neg931 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 31 * h, w, h, subWindowImgIntegral)
                    neg1031 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 31 * h, w, h, subWindowImgIntegral)
                    neg1131 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 31 * h, w, h, subWindowImgIntegral)
                    neg1231 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 31 * h, w, h, subWindowImgIntegral)
                    neg1331 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 31 * h, w, h, subWindowImgIntegral)
                    neg1431 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 31 * h, w, h, subWindowImgIntegral)
                    pos1531 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 31 * h, w, h, subWindowImgIntegral)
                    pos1631 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 31 * h, w, h, subWindowImgIntegral)
                    pos1731 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 31 * h, w, h, subWindowImgIntegral)
                    neg1831 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 31 * h, w, h, subWindowImgIntegral)
                    neg1931 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 31 * h, w, h, subWindowImgIntegral)
                    neg032 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 32 * h, w, h, subWindowImgIntegral)
                    neg132 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 32 * h, w, h, subWindowImgIntegral)
                    neg232 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 32 * h, w, h, subWindowImgIntegral)
                    pos332 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 32 * h, w, h, subWindowImgIntegral)
                    pos432 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 32 * h, w, h, subWindowImgIntegral)
                    neg532 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 32 * h, w, h, subWindowImgIntegral)
                    neg632 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 32 * h, w, h, subWindowImgIntegral)
                    neg732 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 32 * h, w, h, subWindowImgIntegral)
                    neg832 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 32 * h, w, h, subWindowImgIntegral)
                    neg932 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 32 * h, w, h, subWindowImgIntegral)
                    neg1032 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 32 * h, w, h, subWindowImgIntegral)
                    neg1132 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 32 * h, w, h, subWindowImgIntegral)
                    neg1232 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 32 * h, w, h, subWindowImgIntegral)
                    neg1332 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 32 * h, w, h, subWindowImgIntegral)
                    neg1432 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 32 * h, w, h, subWindowImgIntegral)
                    pos1532 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 32 * h, w, h, subWindowImgIntegral)
                    pos1632 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 32 * h, w, h, subWindowImgIntegral)
                    pos1732 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 32 * h, w, h, subWindowImgIntegral)
                    neg1832 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 32 * h, w, h, subWindowImgIntegral)
                    neg1932 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 32 * h, w, h, subWindowImgIntegral)
                    neg033 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 33 * h, w, h, subWindowImgIntegral)
                    neg133 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 33 * h, w, h, subWindowImgIntegral)
                    neg233 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 33 * h, w, h, subWindowImgIntegral)
                    neg333 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 33 * h, w, h, subWindowImgIntegral)
                    neg433 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 33 * h, w, h, subWindowImgIntegral)
                    neg533 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 33 * h, w, h, subWindowImgIntegral)
                    neg633 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 33 * h, w, h, subWindowImgIntegral)
                    neg733 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 33 * h, w, h, subWindowImgIntegral)
                    neg833 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 33 * h, w, h, subWindowImgIntegral)
                    neg933 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 33 * h, w, h, subWindowImgIntegral)
                    neg1033 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 33 * h, w, h, subWindowImgIntegral)
                    neg1133 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 33 * h, w, h, subWindowImgIntegral)
                    neg1233 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 33 * h, w, h, subWindowImgIntegral)
                    neg1333 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 33 * h, w, h, subWindowImgIntegral)
                    neg1433 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 33 * h, w, h, subWindowImgIntegral)
                    neg1533 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 33 * h, w, h, subWindowImgIntegral)
                    neg1633 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 33 * h, w, h, subWindowImgIntegral)
                    pos1733 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 33 * h, w, h, subWindowImgIntegral)
                    neg1833 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 33 * h, w, h, subWindowImgIntegral)
                    neg1933 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 33 * h, w, h, subWindowImgIntegral)
                    neg034 = self.haar.getPixelValInIntegralMat(x + 0 * w, y + 34 * h, w, h, subWindowImgIntegral)
                    neg134 = self.haar.getPixelValInIntegralMat(x + 1 * w, y + 34 * h, w, h, subWindowImgIntegral)
                    neg234 = self.haar.getPixelValInIntegralMat(x + 2 * w, y + 34 * h, w, h, subWindowImgIntegral)
                    neg334 = self.haar.getPixelValInIntegralMat(x + 3 * w, y + 34 * h, w, h, subWindowImgIntegral)
                    neg434 = self.haar.getPixelValInIntegralMat(x + 4 * w, y + 34 * h, w, h, subWindowImgIntegral)
                    neg534 = self.haar.getPixelValInIntegralMat(x + 5 * w, y + 34 * h, w, h, subWindowImgIntegral)
                    neg634 = self.haar.getPixelValInIntegralMat(x + 6 * w, y + 34 * h, w, h, subWindowImgIntegral)
                    neg734 = self.haar.getPixelValInIntegralMat(x + 7 * w, y + 34 * h, w, h, subWindowImgIntegral)
                    neg834 = self.haar.getPixelValInIntegralMat(x + 8 * w, y + 34 * h, w, h, subWindowImgIntegral)
                    neg934 = self.haar.getPixelValInIntegralMat(x + 9 * w, y + 34 * h, w, h, subWindowImgIntegral)
                    neg1034 = self.haar.getPixelValInIntegralMat(x + 10 * w, y + 34 * h, w, h, subWindowImgIntegral)
                    neg1134 = self.haar.getPixelValInIntegralMat(x + 11 * w, y + 34 * h, w, h, subWindowImgIntegral)
                    neg1234 = self.haar.getPixelValInIntegralMat(x + 12 * w, y + 34 * h, w, h, subWindowImgIntegral)
                    neg1334 = self.haar.getPixelValInIntegralMat(x + 13 * w, y + 34 * h, w, h, subWindowImgIntegral)
                    neg1434 = self.haar.getPixelValInIntegralMat(x + 14 * w, y + 34 * h, w, h, subWindowImgIntegral)
                    neg1534 = self.haar.getPixelValInIntegralMat(x + 15 * w, y + 34 * h, w, h, subWindowImgIntegral)
                    neg1634 = self.haar.getPixelValInIntegralMat(x + 16 * w, y + 34 * h, w, h, subWindowImgIntegral)
                    neg1734 = self.haar.getPixelValInIntegralMat(x + 17 * w, y + 34 * h, w, h, subWindowImgIntegral)
                    neg1834 = self.haar.getPixelValInIntegralMat(x + 18 * w, y + 34 * h, w, h, subWindowImgIntegral)
                    neg1934 = self.haar.getPixelValInIntegralMat(x + 19 * w, y + 34 * h, w, h, subWindowImgIntegral)
                    scaledWindowsMat[window][dimension] = (( \
                                                                       neg00 + neg10 + neg20 + neg30 + neg40 + neg50 + neg60 + neg70 + neg80 + neg90 + neg100 + neg110 + neg120 + neg130 + neg140 + neg150 + neg160 + neg170 + neg180 + neg190 + neg01 + neg11 + neg21 + neg31 + neg41 + neg51 + neg61 + neg71 + neg81 + neg91 + neg101 + neg111 + neg121 + neg131 + neg141 + neg151 + neg161 + neg171 + neg181 + neg191 + neg02 + neg12 + neg22 + neg32 + neg42 + neg52 + neg62 + neg72 + neg102 + neg112 + neg122 + neg132 + neg142 + neg152 + neg162 + neg172 + neg182 + neg192 + neg03 + neg13 + neg23 + neg33 + neg43 + neg53 + neg63 + neg113 + neg123 + neg133 + neg143 + neg153 + neg163 + neg173 + neg183 + neg193 + neg04 + neg14 + neg24 + neg34 + neg44 + neg54 + neg64 + neg114 + neg124 + neg134 + neg144 + neg154 + neg164 + neg174 + neg184 + neg194 + neg05 + neg15 + neg25 + neg35 + neg45 + neg55 + neg115 + neg125 + neg135 + neg145 + neg155 + neg165 + neg175 + neg185 + neg195 + neg06 + neg16 + neg26 + neg36 + neg46 + neg116 + neg126 + neg136 + neg146 + neg156 + neg166 + neg176 + neg186 + neg196 + neg07 + neg17 + neg27 + neg37 + neg47 + neg137 + neg147 + neg157 + neg167 + neg177 + neg187 + neg197 + neg08 + neg18 + neg28 + neg38 + neg48 + neg158 + neg168 + neg178 + neg188 + neg198 + neg09 + neg19 + neg29 + neg39 + neg159 + neg169 + neg179 + neg189 + neg199 + neg010 + neg110 + neg210 + neg310 + neg1510 + neg1610 + neg1710 + neg1810 + neg1910 + neg011 + neg111 + neg211 + neg311 + neg1511 + neg1611 + neg1711 + neg1811 + neg1911 + neg012 + neg112 + neg212 + neg312 + neg1412 + neg1512 + neg1612 + neg1712 + neg1812 + neg1912 + neg013 + neg113 + neg213 + neg1413 + neg1513 + neg1613 + neg1713 + neg1813 + neg1913 + neg014 + neg114 + neg214 + neg1414 + neg1514 + neg1614 + neg1714 + neg1814 + neg1914 + neg015 + neg115 + neg215 + neg415 + neg1515 + neg1615 + neg1715 + neg1815 + neg1915 + neg016 + neg116 + neg216 + neg316 + neg416 + neg716 + neg816 + neg1416 + neg1516 + neg1616 + neg1716 + neg1816 + neg1916 + neg017 + neg117 + neg217 + neg317 + neg717 + neg1417 + neg1517 + neg1617 + neg1717 + neg1817 + neg1917 + neg018 + neg118 + neg218 + neg1418 + neg1518 + neg1618 + neg1718 + neg1818 + neg1918 + neg019 + neg119 + neg1419 + neg1519 + neg1619 + neg1719 + neg1819 + neg1919 + neg020 + neg120 + neg1320 + neg1420 + neg1520 + neg1620 + neg1720 + neg1820 + neg1920 + neg021 + neg121 + neg1321 + neg1421 + neg1521 + neg1621 + neg1721 + neg1821 + neg1921 + neg022 + neg1322 + neg1422 + neg1522 + neg1622 + neg1722 + neg1822 + neg1922 + neg023 + neg1423 + neg1523 + neg1623 + neg1723 + neg1823 + neg1923 + neg024 + neg224 + neg324 + neg1424 + neg1524 + neg1624 + neg1724 + neg1824 + neg1924 + neg025 + neg125 + neg225 + neg325 + neg425 + neg1525 + neg1625 + neg1725 + neg1825 + neg1925 + neg026 + neg126 + neg226 + neg326 + neg426 + neg1626 + neg1726 + neg1826 + neg1926 + neg027 + neg127 + neg227 + neg327 + neg427 + neg1727 + neg1827 + neg1927 + neg028 + neg128 + neg228 + neg328 + neg428 + neg928 + neg1028 + neg1128 + neg1728 + neg1828 + neg1928 + neg029 + neg129 + neg229 + neg329 + neg829 + neg929 + neg1029 + neg1129 + neg1829 + neg1929 + neg030 + neg130 + neg230 + neg330 + neg730 + neg830 + neg930 + neg1030 + neg1130 + neg1230 + neg1330 + neg1830 + neg1930 + neg031 + neg131 + neg231 + neg531 + neg631 + neg731 + neg831 + neg931 + neg1031 + neg1131 + neg1231 + neg1331 + neg1431 + neg1831 + neg1931 + neg032 + neg132 + neg232 + neg532 + neg632 + neg732 + neg832 + neg932 + neg1032 + neg1132 + neg1232 + neg1332 + neg1432 + neg1832 + neg1932 + neg033 + neg133 + neg233 + neg333 + neg433 + neg533 + neg633 + neg733 + neg833 + neg933 + neg1033 + neg1133 + neg1233 + neg1333 + neg1433 + neg1533 + neg1633 + neg1833 + neg1933 + neg034 + neg134 + neg234 + neg334 + neg434 + neg534 + neg634 + neg734 + neg834 + neg934 + neg1034 + neg1134 + neg1234 + neg1334 + neg1434 + neg1534 + neg1634 + neg1734 + neg1834 + neg1934
                                                           ) - (
                                                                   pos82 + pos92 + pos73 + pos83 + pos93 + pos103 + pos74 + pos84 + pos94 + pos104 + pos65 + pos75 + pos85 + pos95 + pos105 + pos56 + pos66 + pos76 + pos86 + pos96 + pos106 + pos57 + pos67 + pos77 + pos87 + pos97 + pos107 + pos117 + pos127 + pos58 + pos68 + pos78 + pos88 + pos98 + pos108 + pos118 + pos128 + pos138 + pos148 + pos49 + pos59 + pos69 + pos79 + pos89 + pos99 + pos109 + pos119 + pos129 + pos139 + pos149 + pos410 + pos510 + pos610 + pos710 + pos810 + pos910 + pos1010 + pos1110 + pos1210 + pos1310 + pos1410 + pos411 + pos511 + pos611 + pos711 + pos811 + pos911 + pos1011 + pos1111 + pos1211 + pos1311 + pos1411 + pos412 + pos512 + pos612 + pos712 + pos812 + pos912 + pos1012 + pos1112 + pos1212 + pos1312 + pos313 + pos413 + pos513 + pos613 + pos713 + pos813 + pos913 + pos1013 + pos1113 + pos1213 + pos1313 + pos314 + pos414 + pos514 + pos614 + pos714 + pos814 + pos914 + pos1014 + pos1114 + pos1214 + pos1314 + pos315 + pos515 + pos615 + pos715 + pos815 + pos915 + pos1015 + pos1115 + pos1215 + pos1315 + pos1415 + pos516 + pos616 + pos916 + pos1016 + pos1116 + pos1216 + pos1316 + pos417 + pos517 + pos617 + pos817 + pos917 + pos1017 + pos1117 + pos1217 + pos1317 + pos318 + pos418 + pos518 + pos618 + pos718 + pos818 + pos918 + pos1018 + pos1118 + pos1218 + pos1318 + pos219 + pos319 + pos419 + pos519 + pos619 + pos719 + pos819 + pos919 + pos1019 + pos1119 + pos1219 + pos1319 + pos220 + pos320 + pos420 + pos520 + pos620 + pos720 + pos820 + pos920 + pos1020 + pos1120 + pos1220 + pos221 + pos321 + pos421 + pos521 + pos621 + pos721 + pos821 + pos921 + pos1021 + pos1121 + pos1221 + pos122 + pos222 + pos322 + pos422 + pos522 + pos622 + pos722 + pos822 + pos922 + pos1022 + pos1122 + pos1222 + pos123 + pos223 + pos323 + pos423 + pos523 + pos623 + pos723 + pos823 + pos923 + pos1023 + pos1123 + pos1223 + pos1323 + pos124 + pos424 + pos524 + pos624 + pos724 + pos824 + pos924 + pos1024 + pos1124 + pos1224 + pos1324 + pos525 + pos625 + pos725 + pos825 + pos925 + pos1025 + pos1125 + pos1225 + pos1325 + pos1425 + pos526 + pos626 + pos726 + pos826 + pos926 + pos1026 + pos1126 + pos1226 + pos1326 + pos1426 + pos1526 + pos527 + pos627 + pos727 + pos827 + pos927 + pos1027 + pos1127 + pos1227 + pos1327 + pos1427 + pos1527 + pos1627 + pos528 + pos628 + pos728 + pos828 + pos1228 + pos1328 + pos1428 + pos1528 + pos1628 + pos429 + pos529 + pos629 + pos729 + pos1229 + pos1329 + pos1429 + pos1529 + pos1629 + pos1729 + pos430 + pos530 + pos630 + pos1430 + pos1530 + pos1630 + pos1730 + pos331 + pos431 + pos1531 + pos1631 + pos1731 + pos332 + pos432 + pos1532 + pos1632 + pos1732 + pos1733 ) / (
                                                                       700 * w * h))
                    # scaledWindowsMat[window][dimension] = (pos - neg) / (50 * w * h)

                # if h_type == "HAAR_TYPE_I":
                #      pos = self.haar.getPixelValInIntegralMat(x, y, w, h, subWindowImgIntegral)
                #      neg = self.haar.getPixelValInIntegralMat(x, y + h, w, h, subWindowImgIntegral)
                #      scaledWindowsMat[window][dimension] = (pos - neg) / (2 * w * h)

                if h_type == "HAAR_TYPE_II":
                    neg = self.haar.getPixelValInIntegralMat(x, y, w, h, subWindowImgIntegral)
                    pos = self.haar.getPixelValInIntegralMat(x + w, y, w, h, subWindowImgIntegral)

                    scaledWindowsMat[window][dimension] = (pos - neg) / (2 * w * h)
                # elif h_type == "HAAR_TYPE_III":
                #     neg1 = self.haar.getPixelValInIntegralMat(x, y, w, h, subWindowImgIntegral)
                #     pos  = self.haar.getPixelValInIntegralMat(x + w, y, w, h, subWindowImgIntegral)
                #     neg2 = self.haar.getPixelValInIntegralMat(x + 2 * w, y, w, h, subWindowImgIntegral)
                #
                #     scaledWindowsMat[window][dimension] = (pos - neg1 - neg2) / (3 * w * h)
                #
                # elif h_type == "HAAR_TYPE_IV":
                #     neg1 = self.haar.getPixelValInIntegralMat(x, y, w, h, subWindowImgIntegral)
                #     pos  = self.haar.getPixelValInIntegralMat(x, y + h, w, h, subWindowImgIntegral)
                #     neg2 = self.haar.getPixelValInIntegralMat(x, y + 2 * h, w, h, subWindowImgIntegral)
                #
                #     scaledWindowsMat[window][dimension] = (pos - neg1 - neg2) / (3 * w * h)
                #
                # elif h_type == "HAAR_TYPE_V":
                #     neg1 = self.haar.getPixelValInIntegralMat(x, y, w, h, subWindowImgIntegral)
                #     pos1 = self.haar.getPixelValInIntegralMat(x + w, y, w, h, subWindowImgIntegral)
                #     pos2 = self.haar.getPixelValInIntegralMat(x, y + h, w, h, subWindowImgIntegral)
                #     neg2 = self.haar.getPixelValInIntegralMat(x + w, y + h, w, h, subWindowImgIntegral)
                #
                #     scaledWindowsMat[window][dimension] = (pos1 + pos2 - neg1 - neg2) / (4 * w * h)

        pred = self.model.predict_prob(scaledWindowsMat)
        indexs = np.where(pred > 0)[0]
        #print(indexs)
        predWindow = np.zeros((len(indexs), scaledWindows.shape[1]+1), dtype=object)
        for i in range(len(indexs)):
            predWindow[i] = np.append(scaledWindows[indexs[i]], pred[indexs[i]])


        return predWindow

    def _optimalWindow(self, predWindow):
        """optimize the windows according to the situations of overlapping...
        :param predWindow: (x, y, w, h, scale, prob)
        :return:
        """
        optimalWindowMap = np.array([i for i in range(predWindow.shape[0])])
        print(optimalWindowMap)
        for i in range(predWindow.shape[0]):
            for j in range(i+1, predWindow.shape[0]):
                overlap = False
                contain = False

                if self._windowInAnotherWindow(predWindow[i], predWindow[j]):
                    # optimalWindowMap[np.where(optimalWindowMap == optimalWindowMap[i])] = optimalWindowMap[j]
                    contain = True
                elif self._windowInAnotherWindow(predWindow[j], predWindow[i]):
                    # optimalWindowMap[np.where(optimalWindowMap == optimalWindowMap[j])] = optimalWindowMap[i]
                    contain = True
                else:
                    for x in [predWindow[i][0], predWindow[i][0] + predWindow[i][2]]:
                        for y in [predWindow[i][1], predWindow[i][1] + predWindow[i][3]]:
                            if self._pointInWindow((x, y), predWindow[j]):
                                overlap = True
                                break
                    for x in [predWindow[j][0], predWindow[j][0] + predWindow[j][2]]:
                        for y in [predWindow[j][1], predWindow[j][1] + predWindow[j][3]]:
                            if self._pointInWindow((x, y), predWindow[i]):
                                overlap = True
                                break

                if overlap or contain:
                    if predWindow[i][-1] == max(predWindow[i][-1], predWindow[j][-1]):
                        optimalWindowMap[np.where(optimalWindowMap == optimalWindowMap[j])] = optimalWindowMap[i]

                    else:
                        optimalWindowMap[np.where(optimalWindowMap == optimalWindowMap[i])] = optimalWindowMap[j]

        optimalWindow = np.zeros(len(set(optimalWindowMap)), dtype=object)
        index = 0
        for i in set(optimalWindowMap):
            optimalWindow[index] = predWindow[i]
            index = index + 1
        return optimalWindow

    def _pointInWindow(self, point, window):
        """
        :param point: (x, y)
        :param window: (x, y, w, h, scale, prob)
        :return:
        """
        if point[0] >= window[0] and point[0] <= window[0] + window[2]:
            if point[1] >= window[1] and point[1] <= window[1] + window[3]:
                return True
        return False

    def _windowInAnotherWindow(self, window, anotherWindow):
        """
        :param window: (x, y, w, h, scale, prob)
        :param anotherWindow: (x, y, w, h, scale, prob)
        :return:
        """
        if self._pointInWindow((window[0], window[1]), anotherWindow):
            if self._pointInWindow((window[0]+window[2], window[1]), anotherWindow):
                if self._pointInWindow((window[0], window[1]+window[3]), anotherWindow):
                    if self._pointInWindow((window[0]+window[2], window[1]+window[3]), anotherWindow):
                        return True
        return False


