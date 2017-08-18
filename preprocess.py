import cPickle
import cv2
import math
import numpy as np
import os
import sys
import xml.etree.ElementTree as ET

indexFile = '../VOC2007/train/ImageSets/Main/train.txt'

imageWidth = 448
imageHeight = 448

numOfClasses = 20
numOfGrids = 7
numOfBoxes = 2

widthOfGrid = imageWidth / numOfGrids
heightOfGrid = imageHeight / numOfGrids


classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


data = []


def preprocess_dataset():

    fileHandle = open(indexFile, 'rb')
    numLines = sum(1 for line in open(indexFile, 'rb'))
    linesProcessed = 0

    totalImages = 0
    totalObjects = 0
    for line in fileHandle:
        # per image
        totalImages += 1
        index = line.split(' ')[0].split('\n')[0]

        imagePath = '../VOC2007/train/JPEGImages/' + index + '.jpg'
        imageMatrix = cv2.imread(imagePath)
        originalImageHeight, originalImageWidth, _ = imageMatrix.shape

        # transform the image
        imageMatrix = cv2.resize(imageMatrix, (448, 448))
        imageMatrix = cv2.cvtColor(imageMatrix, cv2.COLOR_BGR2RGB)
        imageMatrix = np.asarray(imageMatrix, dtype='float32')

        # should the matrix be normalised ?
        xmlPath = '../VOC2007/train/Annotations/' + index + '.xml'
        xmlParse = ET.parse(xmlPath)

        # find the annotations of objects in the list
        xmlObject = xmlParse.findall('object')
        for item in xmlObject:

            # per object per image
            totalObjects += 1
            category = item.find('name').text
            categoryIndex = classes.index(category)
            xmin = int(item.find('bndbox').find('xmin').text)
            xmax = int(item.find('bndbox').find('xmax').text)
            ymin = int(item.find('bndbox').find('ymin').text)
            ymax = int(item.find('bndbox').find('ymax').text)
            # print(xmin, xmax, ymin, ymax)

            # take the midpoint and scale according to new size
            xC = ((xmin + xmax) * 1.0 / 2.0) * \
                (imageWidth * 1.0 / originalImageWidth)
            yC = ((ymin + ymax) * 1.0 / 2.0) * \
                (imageHeight * 1.0 / originalImageHeight)
            w = (xmax - xmin) * 1.0
            h = (ymax - ymin) * 1.0
            # print((ymin + ymax) / 2.0, (imageHeight * 1.0 / originalImageHeight))

            # take xC and yC as offsets to the original image
            xOffset = ((xC % widthOfGrid) * 1.0) / widthOfGrid
            yOffset = ((yC % heightOfGrid) * 1.0) / heightOfGrid

            # normalised squareroot of width and height ?
            # i think it should bot be in preprocessing
            wSqrt = math.sqrt(w * 1.0) / originalImageWidth
            hSqrt = math.sqrt(h * 1.0) / originalImageHeight
            # w = (w * 1.0) / originalImageWidth
            # h = (h * 1.0) / originalImageHeight

            boxData = [xOffset, yOffset, wSqrt, hSqrt]

            # one hot encoding to be expected as NN output
            classProbability = np.zeros([20])
            classProbability[categoryIndex] = 1

            # marking the grid responsible for detection
            responsibleGridX = int(xC / widthOfGrid)
            responsibleGridY = int(yC / heightOfGrid)
            responsibleGrid = np.zeros([7, 7])

            # print(xmin, xmax, ymin, ymax, originalImageWidth, originalImageHeight, xC, yC, w, h, responsibleGridX, responsibleGridY)

            responsibleGrid[responsibleGridX][responsibleGridY] = 1
            responsibleGrid = np.reshape(responsibleGrid, [49])

            outputLabels = np.hstack(
                [classProbability, boxData, responsibleGrid])
            data.append({'inputImage': imageMatrix,
                         'outputLabels': outputLabels})

        # Print something after processing 5 percent of files
        linesProcessed = linesProcessed + 1
        if linesProcessed % 100 == 0:
            print('Processed ' + str(linesProcessed) + ' images out of ' + str(numLines))
            fileName = 'data/data-' + str(linesProcessed / 100) + '.pkl'

            with open(fileName, 'w') as f:
                cPickle.dump(data, f)
            
            print('Saved ' + fileName)
            del data[:]

    fileHandle.close()
    print('totalObjects = ' + totalObjects)
    print('totalImages = ' + totalImages)

if __name__ == "__main__":
    preprocess_dataset()
