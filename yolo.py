import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import sys
import tensorflow as tf
import time
import xml.etree.cElementTree as ET

from six.moves import cPickle as pickle
from xml.dom import minidom

# build a grpah for YOLO tiny architecture
# define a general conv layer, pooling layer and fc layer
class Yolo:

    weightFile = 'weights/yolo_small.ckpt'

    numOfClasses = 20
    numOfBoxes = 2
    numOfGrids = 7

    # parameter for leaky relu
    alpha = 0.1
    # momentum to be used during training
    momentum = 0.9
    # batch size during training
    batchSize = 64

    # used to disregard bounding boxes
    scoreThreshold = 0.2
    # used for non-maximum supression
    IoUThreshold = 0.5
    # used to increase contribution of localisation in the error pipeline
    lambdaCoordinate = 5.0
    # used to decrease contribution of cells which do not contain an object
    lambdaNoObject = 0.5
    # to display updated status of the program
    displayConsole = False

    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    # flags to be set while testing the system
    mode = 'testLive'

    # bounding box seeds
    seed = [random.randint(1, 1000) for i in range(3)]
    # test on a database
    if mode == 'testDB':
        saveAnnotatedImage = True
        saveAnnotatedXML = True
        showImage = False
        inputFolder = '../VOC2007/test/JPEGImages/'
        outputFolder = 'VOC2007/test/outputImages/'
        textOutputFolder = 'VOC2007/test/outputAnnotations/'
    # test on a webcam stream
    elif mode == 'testLive':
        saveAnnotatedImage = False
        saveAnnotatedXML = False
        showImage = True
    elif mode == 'testFile':
        saveAnnotatedImage = True
        saveAnnotatedXML = False
        showImage = True
        inputFile = 'test/006656.jpg'
        outputFile = 'test/output.jpg'
        textOutputFile = 'test/outputAnnotations.txt'
    else:
        pass

    def __init__(self):
        self.build_graph()
        if self.mode == 'testDB':
            for fileName in os.listdir(self.inputFolder):
                inputFile = os.path.join(self.inputFolder, fileName)
                annotatedImage, predictedObjects = self.detect_from_file(
                    inputFile)

                # show tht image as output
                if self.showImage:
                    cv2.imshow('YOLO Detection', annotatedImage)
                    cv2.waitKey(1)

                # save image and save the parameters for detected object
                if self.saveAnnotatedImage:
                    outputFile = os.path.join(self.outputFolder, fileName)
                    cv2.imwrite(outputFile, annotatedImage)

                # storing output in xml format
                if self.saveAnnotatedXML:
                    outputTextFile = os.path.join(
                        self.textOutputFolder, fileName)
                    root = ET.Element("annotation")
                    ET.SubElement(root, "filename").text = fileName
                    for i in range(len(predictedObjects)):
                        object = ET.SubElement(root, "object")
                        ET.SubElement(
                            object, "name").text = predictedObjects[i][0]
                        ET.SubElement(object, "confidence").text = str(
                            predictedObjects[i][5])
                        bndbox = ET.SubElement(object, "bndbox")
                        ET.SubElement(bndbox, "xmin").text = str(
                            predictedObjects[i][1])
                        ET.SubElement(bndbox, "ymin").text = str(
                            predictedObjects[i][2])
                        ET.SubElement(bndbox, "xmax").text = str(
                            predictedObjects[i][3])
                        ET.SubElement(bndbox, "ymax").text = str(
                            predictedObjects[i][4])
                    xmlString = minidom.parseString(
                        ET.tostring(root)).toprettyxml(indent="   ")
                    with open(outputTextFile.split('.')[0] + '.xml', 'w') as f:
                        f.write(xmlString.encode('utf-8'))

        elif self.mode == 'testLive':
            # detect from video
            cap = cv2.VideoCapture(0)
            while(True):
                # capture frames
                ret, frame = cap.read()
                annotatedImage, predictedObjects = self.detect_from_matrix(
                    frame)

                # show tht image as output
                if self.showImage:
                    cv2.imshow('YOLO Detection', annotatedImage)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cap.release()
            cv2.destroyAllWindows()

        elif self.mode == 'testFile':
            annotatedImage, predictedObjects = self.detect_from_file(
                self.inputFile)

            # show tht image as output
            if self.showImage:
                cv2.imshow('YOLO Detection', annotatedImage)
                cv2.waitKey(10)

            # save image and save the parameters for detected object
            if self.saveAnnotatedImage:
                cv2.imwrite(self.outputFile, annotatedImage)
        else:
            # train mode
            pass

    # Builds the computational graph for the network
    def build_graph(self):
        if self.displayConsole:
            print('Building Yolo Graph....')
        self.x = tf.placeholder('float32', [None, 448, 448, 3])

        self.conv1 = self.conv_layer(1, self.x, 64, 7, 2)
        self.pool1 = self.maxpool_layer(2, self.conv1, 2, 2)

        # size reduced to 64x112x112
        self.conv2 = self.conv_layer(3, self.pool1, 192, 3, 1)
        self.pool2 = self.maxpool_layer(4, self.conv2, 2, 2)

        # size reduced to 192x56x56
        self.conv3 = self.conv_layer(5, self.pool2, 128, 1, 1)
        self.conv4 = self.conv_layer(6, self.conv3, 256, 3, 1)
        self.conv5 = self.conv_layer(7, self.conv4, 256, 1, 1)
        self.conv6 = self.conv_layer(8, self.conv5, 512, 3, 1)
        self.pool3 = self.maxpool_layer(9, self.conv6, 2, 2)

        # size reduced to 512x28x28
        self.conv7 = self.conv_layer(10, self.pool3, 256, 1, 1)
        self.conv8 = self.conv_layer(11, self.conv7, 512, 3, 1)
        self.conv9 = self.conv_layer(12, self.conv8, 256, 1, 1)
        self.conv10 = self.conv_layer(13, self.conv9, 512, 3, 1)

        self.conv11 = self.conv_layer(14, self.conv10, 256, 1, 1)
        self.conv12 = self.conv_layer(15, self.conv11, 512, 3, 1)
        self.conv13 = self.conv_layer(16, self.conv12, 256, 1, 1)
        self.conv14 = self.conv_layer(17, self.conv13, 512, 3, 1)

        # size reduced to 512x28x28
        self.conv15 = self.conv_layer(18, self.conv14, 512, 1, 1)
        self.conv16 = self.conv_layer(19, self.conv15, 1024, 3, 1)
        self.pool4 = self.maxpool_layer(20, self.conv16, 2, 2)

        # size reduced to 1024x14x14
        self.conv17 = self.conv_layer(21, self.pool4, 512, 1, 1)
        self.conv18 = self.conv_layer(22, self.conv17, 1024, 3, 1)
        self.conv19 = self.conv_layer(23, self.conv18, 512, 1, 1)
        self.conv20 = self.conv_layer(24, self.conv19, 1024, 3, 1)

        self.conv21 = self.conv_layer(25, self.conv20, 1024, 3, 1)
        self.conv22 = self.conv_layer(26, self.conv21, 1024, 3, 2)
        self.conv23 = self.conv_layer(27, self.conv22, 1024, 3, 1)
        self.conv24 = self.conv_layer(28, self.conv23, 1024, 3, 1)

        # print(self.conv24.get_shape())
        # size reduced to 1024x7x7
        self.fc1 = self.fc_layer(29, self.conv24, 512,
                                 flatten=True, linear=False)
        self.fc2 = self.fc_layer(
            30, self.fc1, 4096, flatten=False, linear=False)
        self.fc3 = self.fc_layer(
            31, self.fc2, 1470, flatten=False, linear=True)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weightFile)

        if self.displayConsole:
            print('Loading Complete \n')

    # complete the training function
    def train_network(self):
        # determine which bounding box is responsible for a particular prediction
        # save the weights after each epoch
        if self.trainData:
            if self.displayConsole:
                print('Started training...')

            for epoch in range(135):
                pass
                # save the model
        else:
            if self.displayConsole:
                print('No train data available')

    def calculate_loss_function(self, predicted, groundTruth):
        '''
        Calculate the total loss for gradient descent
        For each ground truth object, loss needs to be calculated
        Is it assument that each image consists of only one object 

        Predicted
        0-19 CLass prediction
        20-21 Confidence that an object exists in bbox1 or bbox2 of a cell
        22-29 Coordinates for box 1 followed by box 2 

        Real
        0-19 Class prediction ( One-Hot Encoded )
        20-23 Ground truth coordinates for that box
        24-72 Cell has an object/ no object ( Only one of these would be 1 )
        '''

        predictedParameters = np.reshape(
            predicted, [-1, self.numOfGrids, self.numOfGrids, 30])
        predictedClasses = predictedParameters[:, :, :, :20]
        predictedObjectConfidence = predictedParameters[:, :, :, 20:22]
        predictedBoxes = predictedParameters[:, :, :, 22:]

        groundTruthClasses = np.reshape(groundTruth[:, :20], [-1, 1, 1, 20])
        groundTruthBoxes = np.reshape(groundTruth[:, 20:24], [-1, 1, 1, 4])
        groundTruthGrid = np.reshape(groundTruth[:, 24:], [-1, 7, 7, 1])

        predictedFirstBoxes = predictedBoxes[:, :, :, :4]
        predictedSecondBoxes = predictedBoxes[:, :, :, 5:]

        # calulate loss along the 4th axis, localFirstBoxes -1x7x7x1
        # think there should be a simpler method to do this
        lossFirstBoxes = tf.reduce_sum(
            tf.square(predictedFirstBoxes - groundTruthBoxes), 3)
        lossSecondBoxes = tf.reduce_sum(
            tf.square(predictedSecondBoxes - groundTruthBoxes), 3)

        # getting which box ( bbox1 or bbox2 ) is responsible for the detection
        IOU = iouTrain(predictedFirstBoxes,
                       predictedSecondBoxes, groundTruthBoxes)
        responsbileBox = tf.greater(IOU[:, :, :, 0], IOU[:, :, :, 1])
        # suppose it is known that which iou is greater

        # coordinate loss ( loss due to difference in coordinates of
        # predicted-responsible and real box )
        coordinateLoss = tf.where(
            responsibleBox, lossFirstBoxes, lossSecondBoxes)
        # why do we need to reshape it
        coordinateLoss = tf.reshape(coordinateLoss, [-1, 7, 7, 1])
        # count the loss only if the object is in the groundTruth grid
        # gives a sparse -1x7x7x1 matrix, only one element would be nonzero in
        # each slice
        coorinateLoss = self.lambdaCoordinate * \
            tf.multiply(groundTruthGrid, coordinateLoss)

        # object loss ( loss due to difference in object confidence )
        # only take the objectLoss of the predicted grid with higher IoU is
        # responsible for the object
        objectLoss = tf.square(predictedObjectConfidence - groundTruthGrid)
        objectLoss = tf.where(responsibleBox, objectLoss[
                              :, :, :, 0], objectLoss[:, :, :, 1])
        tempObjectLoss = tf.reshape(objectLoss, [-1, 7, 7, 1])
        objectLoss = tf.multiply(groundTruthGrid, tempObjectLoss)

        # class loss ( loss due to misjudgement in class of the object detected
        # )
        classLoss = tf.square(predictedClasses - groundTruthClasses)
        classLoss = tf.reduce_sum(
            tf.mul(groundTruthGrid, classLoss), reduction_indices=3)
        classLoss = tf.reshape(classLoss, [-1, 7, 7, 1])

        # no-object loss, decrease the confidence where there is no object in
        # the ground truth
        noObjectLoss = self.lambdaNoObject * \
            tf.multiply(1 - groundTruthGrid, tempObjectLoss)
        # total loss
        totalLoss = coordinateLoss + objectLoss + classLoss + noObjectLoss
        totalLoss = tf.reduce_mean(tf.reduce_sum(
            totalLoss, reduction_indeces=[1, 2, 3]), reduction_indices=0)

        return totalLoss

    def conv_layer(self, index, inputMatrix, numberOfFilters, sizeOfFilter, stride):
        '''
        Input
        Index : index of the layer within the network
        inputMatrix : self-Explainatory, the input 
        numberOfFilters : number of channel outputs
        sizeOfFilter : defines the receptive field of a particular neuron
        stride : self-Exmplainatory, pixels to skip 

        Output
        Matrix the size of input[0]xinput[1]xnoOfFilters

        '''
        numberOfChannels = inputMatrix.get_shape()[3]

        # int with numberOfChannels
        weight = tf.Variable(tf.truncated_normal(
            [sizeOfFilter, sizeOfFilter, int(numberOfChannels), numberOfFilters], stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=[numberOfFilters]))

        padSize = sizeOfFilter // 2
        # print(inputMatrix.get_shape())
        #paddedInput = np.lib.pad(inputMatrix,((0,0),(padSize,padSize),(padSize,padSize),(0,0)),'constant', constant_values = (0,0))
        paddedInput = tf.pad(
            inputMatrix, ([[0, 0], [padSize, padSize], [padSize, padSize], [0, 0]]))

        conv = tf.nn.conv2d(paddedInput, weight, strides=[
                            1, stride, stride, 1], padding='VALID', name=str(index) + '_conv')
        conv_bias = tf.add(conv, bias, name=str(index) + '_conv')

        if self.displayConsole:
            print(' Layer %d Type: Conv Size: %dx%d Stride: %d No.Filters: %d Input Channels : %d' % (index, sizeOfFilter, sizeOfFilter, stride, numberOfFilters, numberOfChannels))
        # leaky relu as mentioned in YOLO paper
        return tf.maximum(self.alpha * conv_bias, conv_bias, name=str(index) + '_leaky_relu')

    def maxpool_layer(self, index, inputMatrix, sizeOfFilter, stride):

        if self.displayConsole:
            print(' Layer %d Type: Maxpool Size: %dx%d Stride: %d' % (index, sizeOfFilter, sizeOfFilter, stride))
        maxpool = tf.nn.max_pool(inputMatrix, ksize=[1, sizeOfFilter, sizeOfFilter, 1], strides=[
                                 1, sizeOfFilter, sizeOfFilter, 1], padding='SAME', name=str(index) + '_maxpool')
        return maxpool

    def fc_layer(self, index, inputMatrix, outputNeurons, flatten, linear):

        inputShape = inputMatrix.get_shape().as_list()
        if flatten:
            # flatten the matrix
            inputDimension = inputShape[1] * inputShape[2] * inputShape[3]
            # change it to the input as required by fully connected layer
            inputMatrixAdjust = tf.transpose(inputMatrix, (0, 3, 1, 2))
            inputMatrixAdjust = tf.reshape(
                inputMatrixAdjust, [-1, inputDimension])
        else:
            inputDimension = inputShape[1]
            inputMatrixAdjust = inputMatrix

        weight = tf.Variable(tf.truncated_normal(
            [inputDimension, outputNeurons], stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=[outputNeurons]))

        if self.displayConsole:
            print(' Layer %d Type: FullyConnected InSize: %d OutSize %d Linear: %d' % (index, inputDimension, outputNeurons, int(linear)))

        # linear or leaky relu activation
        if linear:
            return tf.add(tf.matmul(inputMatrixAdjust, weight), bias, name=str(index) + '_fc')
        else:
            answer = tf.add(tf.matmul(inputMatrixAdjust, weight),
                            bias, name=str(index) + '_fc')
            return tf.maximum(self.alpha * answer, answer, name=str(index) + '_fc')

    def detect_from_file(self, fileName):

        if self.displayConsole:
            print('Detecting object from :' + fileName)
        imageMatrix = cv2.imread(fileName)
        return self.detect_from_matrix(imageMatrix)

    def detect_from_matrix(self, imageMatrix):

        image = imageMatrix
        self.imageHeight, self.imageWidth, _ = imageMatrix.shape

        # Resize the image as required by network
        imageMatrix = cv2.resize(imageMatrix, (448, 448))
        imageMatrix = cv2.cvtColor(imageMatrix, cv2.COLOR_BGR2RGB)

        imageMatrix = np.asarray(imageMatrix, dtype='float32')

        # Normalize the image vlaues between 0 and 1
        normMatrix = np.zeros((1, 448, 448, 3), dtype='float32')
        normMatrix[0] = (imageMatrix / 255.0) * 2.0 - 1.0

        netOutput = self.sess.run(self.fc3, feed_dict={self.x: normMatrix})
        self.result = self.interpret_output(netOutput)
        return self.annotate_image(image, self.result)

    def interpret_output(self, netOutput):
        '''
        Threshold the confidence for all classes and apply Non-Maximum supression
        '''
        # print(sum(sum(netOutput)))
        # to fill in the probability of every class
        classProbability = np.zeros(
            [self.numOfGrids, self.numOfGrids, self.numOfBoxes, self.numOfClasses])
        # print(netOutput[0:980].shape)

        # this should be called objectProbability
        classConditionalProbability = np.reshape(
            netOutput[:, 0:980], [self.numOfGrids, self.numOfGrids, self.numOfClasses])
        classConfidence = np.reshape(netOutput[:, 980:1078], [
                                     self.numOfGrids, self.numOfGrids, self.numOfBoxes])

        #(x,y,w,h) of the two bounding boxes predicted by the network
        boxData = np.reshape(netOutput[:, 1078:], [
                             self.numOfGrids, self.numOfGrids, self.numOfBoxes, 4])

        for i in range(self.numOfBoxes):
            for j in range(self.numOfClasses):
                classProbability[:, :, i, j] = np.multiply(
                    classConditionalProbability[:, :, j], classConfidence[:, :, i])

        offset = np.transpose(np.reshape([np.arange(
            7)] * 14, (self.numOfBoxes, self.numOfGrids, self.numOfGrids)), (1, 2, 0))

        # changing x and y coordinates from model representation to image
        # representation
        boxData[:, :, :, 0] = (
            (boxData[:, :, :, 0] + offset[:, :, :]) * 1.0 / self.numOfGrids) * self.imageWidth
        boxData[:, :, :, 1] = ((boxData[:, :, :, 1] + np.transpose(offset,
                                                                   (1, 0, 2))) * 1.0 / self.numOfGrids) * self.imageHeight

        # changing width and height from model representation to image representation, square
        # root of the width and height is predicted because small error in small boxes matter
        # much than small errors in large boxes

        boxData[:, :, :, 2] = np.multiply(boxData[:, :, :, 2], boxData[
                                          :, :, :, 2]) * self.imageWidth
        boxData[:, :, :, 3] = np.multiply(boxData[:, :, :, 3], boxData[
                                          :, :, :, 3]) * self.imageHeight

        # threhold the values
        filterClassProbability = np.array(
            classProbability >= self.scoreThreshold, dtype='bool')
        filteredProbability = classProbability[filterClassProbability]

        filterProbabilityIndex = np.nonzero(filterClassProbability)
        filteredBoxes = boxData[filterProbabilityIndex[
            0], filterProbabilityIndex[1], filterProbabilityIndex[2]]

        filteredClasses = np.argmax(filterClassProbability, axis=3)[filterProbabilityIndex[
            0], filterProbabilityIndex[1], filterProbabilityIndex[2]]

        # sort the values based on scores
        sort = np.array(np.argsort(filteredProbability))[::-1]
        filteredProbability = filteredProbability[sort]
        filteredBoxes = filteredBoxes[sort]
        filteredClasses = filteredClasses[sort]

        # print(sum(filteredProbability))
        # non-maximum supression
        for i in range(len(filteredBoxes)):
            if filteredProbability[i] == 0:
                continue
            for j in range(i + 1, len(filteredBoxes)):
                if self.iou(filteredBoxes[i], filteredBoxes[j]) > self.IoUThreshold:
                    # print('Rejecting Box' + str(j))
                    filteredProbability[j] = 0.0

        filterIoU = np.array(filteredProbability > 0.0, dtype='bool')
        filteredProbability = filteredProbability[filterIoU]
        filteredBoxes = filteredBoxes[filterIoU]
        filteredClasses = filteredClasses[filterIoU]

        # list of bounding boxes with class,x,y,w,h,probability
        result = []
        for i in range(len(filteredBoxes)):
            result.append([self.classes[filteredClasses[i]], filteredBoxes[i][0], filteredBoxes[
                          i][1], filteredBoxes[i][2], filteredBoxes[i][3], filteredProbability[i]])
        return result

    def annotate_image(self, image, results):

        predictedObjects = []
        for i in range(len(results)):
            objectParameters = []

            x = int(results[i][1])
            y = int(results[i][2])
            w = int(results[i][3])
            h = int(results[i][4])

            # print(x, y, w, h, results[i][0])
            imageHeight, imageWidth, _ = image.shape

            w = w // 2
            h = h // 2

            # change to truncate boxes which go outside the image
            xmin, xmax, ymin, ymax = 0, 0, 0, 0
            xmin = 3 if not max(x - w, 0) else (x - w)
            xmax = imageWidth - \
                3 if not min(x + w - imageWidth, 0) else (x + w)
            ymin = 1 if not max(y - h, 0) else (y - h)
            ymax = imageHeight - \
                3 if not min(y + h - imageHeight, 0) else (y + h)

            if self.displayConsole:
                print('Class : ' + results[i][0] + ', [x, y, w, h] [' + str(x) + ', ' + str(y) + ', ' + str(w) + ', ' + str(h) + '] Confidence : ' + str(results[i][5]))
                # each class must have a unique color
            color = tuple(
                [(j * (1 + self.classes.index(results[i][0])) % 255) for j in self.seed])

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            if ymin <= 20:
                cv2.rectangle(image, (xmin, ymin),
                              (xmax, ymin + 20), color, -1)
                cv2.putText(image, results[i][0] + ': %.2f' % results[i][
                            5], (xmin + 5, ymin + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                cv2.rectangle(image, (xmin, ymin),
                              (xmax, ymin - 20), color, -1)
                cv2.putText(image, results[i][0] + ': %.2f' % results[i][
                            5], (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            ''' 
            cv2.rectangle(image, (x - w, y - h), (x + w, y + h), color, 3)
            cv2.rectangle(image, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
            cv2.putText(image, results[i][0] + ': %.2f' % results[i][5], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            '''

            objectParameters = [results[i][0], xmin,
                                ymin, xmax, ymax, results[i][5]]
            predictedObjects.append(objectParameters)
        return image, predictedObjects

        # if self.outputFile:
        #    cv2.imwrite(self.outputFile,image)

    def iou(self, boxA, boxB):

        intersectionX = max(0, min(boxA[0] + boxA[2] * 0.5, boxB[0] + boxB[
                            2] * 0.5) - max(boxA[0] - boxA[2] * 0.5, boxB[0] - boxB[2] * 0.5))
        intersectionY = max(0, min(boxA[1] + boxA[3] * 0.5, boxB[1] + boxB[
                            3] * 0.5) - max(boxA[1] - boxA[3] * 0.5, boxB[1] - boxB[3] * 0.5))
        intersection = intersectionX * intersectionY
        union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - intersection
        # print(intersection, union, intersection / union)
        return intersection / union

    def iouTrain(self, boxA, boxB, realBox):
        '''
        Calculate the IoU in training phase to get the box ( out of N boxes per grid ) responsible for ground truth box
        '''
        iou1 = tf.reshape(iouTrainUnit(boxA, realBox), [-1, 7, 7, 1])
        iou2 = tf.reshape(iouTrainUnit(boxB, realBox), [-1, 7, 7, 1])
        return tf.concat([iou1, iou2], 3)

    def iouTrainUnit(self, boxA, realBox):

        # make sure that the representation of box matches input
        intersectionX = tf.minimum(boxA[:, :, :, 0] + 0.5 * boxA[:, :, :, 2], realBox[:, :, :, 0] + 0.5 * realBox[
                                   :, :, :, 2]) - tf.maximum(boxA[:, :, :, 0] - 0.5 * boxA[:, :, :, 2], realBox[:, :, :, 0] - 0.5 * realBox[:, :, :, 2])
        intersectionY = tf.minimum(boxA[:, :, :, 1] + 0.5 * boxA[:, :, :, 3], realBox[:, :, :, 1] + 0.5 * realBox[
                                   :, :, :, 3]) - tf.maximum(boxA[:, :, :, 1] - 0.5 * boxA[:, :, :, 3], realBox[:, :, :, 1] - 0.5 * realBox[:, :, :, 3])
        intersection = tf.multiply(tf.maximum(
            0, intersectionX), tf.maximum(0, intersectionY))
        union = tf.subtract(tf.multiply(boxA[:, :, :, 1], boxA[
                            :, :, :, 3]) + tf.multiply(realBox[:, :, :, 1], realBox[:, :, :, 3]), intersection)
        iou = tf.divide(intersection, union)
        return iou


def main():
    yolo = Yolo()
    # cv2.waitKey(1000)

if __name__ == '__main__':
    main()
