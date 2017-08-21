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

    numOfClasses = 20

    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
               "cat", "chair", "cow", "diningtable", "dog", "horse",
               "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
               "tvmonitor"]

    # bounding box seeds
    seed = [random.randint(1, 1000) for i in range(3)]

    def __init__(self,
                 mode='testLive',
                 weightFile='weights/yolo_small.ckpt',
                 showImage=None,
                 saveAnnotatedImage=None,
                 saveAnnotatedXML=None,
                 numOfGridsIn1D=7,
                 numOfBoxesPerGrid=2,
                 batchSize=64,
                 verbose=False,
                 minClassProbability=0.2,
                 IoUThreshold=0.5,
                 lambdaCoordinate=5.0,
                 lambdaNoObject=0.5,
                 leakyReLUAlpha=0.9,
                 inputFile=None,
                 outputFile=None,
                 textOutputFile=None,
                 inputFolder=None,
                 outputFolder=None,
                 textOutputFolder=None):
        # Mode to run the Yolo code in
        # {testLive, testFile, testDB, train}
        self.mode = mode
        # Weights file
        self.weightFile = weightFile
        # To save annotated images
        self.saveAnnotatedImage = saveAnnotatedImage
        # To save annotated XML
        self.saveAnnotatedXML = saveAnnotatedXML
        # To show images
        self.showImage = showImage
        # Number of grids in each dimension to divide image into
        self.numOfGridsIn1D = numOfGridsIn1D
        # Number of bounding boxes per grid
        self.numOfBoxesPerGrid = numOfBoxesPerGrid
        # Batch size during training
        self.batchSize = batchSize
        # To display logs of the program
        self.verbose = verbose
        # Used to disregard bounding boxes
        self.minClassProbability = minClassProbability
        # Used for non-maximum supression
        self.IoUThreshold = IoUThreshold
        # Used to increase contribution of localisation in the error pipeline
        self.lambdaCoordinate = lambdaCoordinate
        # Used to decrease contribution of cells which do not contain an object
        self.lambdaNoObject = lambdaNoObject
        # Parameter for leaky relu
        self.leakyReLUAlpha = leakyReLUAlpha
        # Input file
        self.inputFile = inputFile
        # Output file
        self.outputFile = outputFile
        # textOutputFile file
        self.textOutputFile = textOutputFile
        # inputFolder file
        self.inputFolder = inputFolder
        # outputFolder file
        self.outputFolder = outputFolder
        # textOutputFolder file
        self.textOutputFolder = textOutputFolder
        # Input file
        if self.inputFile is None:
            self.inputFile = 'test/006656.jpg'
        # Output file
        if self.outputFile is None:
            self.outputFile = 'test/output.jpg'
        # Text output file
        if self.textOutputFile is None:
            textOutputFile = 'test/outputAnnotations.txt'
        # Input folder of DB
        if self.inputFolder is None:
            self.inputFolder = '../VOC2007/test/JPEGImages/'
        # Output folder of DB
        if self.outputFolder is None:
            self.outputFolder = 'VOC2007/test/outputImages/'
        # Text output folder of DB
        if self.textOutputFolder is None:
            self.textOutputFolder = 'VOC2007/test/outputAnnotations/'
        # Build the YOLO network
        self.buildGraph()
        # If YOLO is to be tested live
        if self.mode == 'testLive':
            # By default, show annotated images, but don't save annotated image
            # or details of predicted objects
            # To show image
            if self.showImage is None:
                self.showImage = True
            # To save annotated images
            if self.saveAnnotatedImage is None:
                self.saveAnnotatedImage = False
            # To save annotated XML
            if self.saveAnnotatedXML is None:
                self.saveAnnotatedXML = False
            # Test YOLO live
            self.yoloTestLive()
        # Else, if YOLO is to be tested on a file
        elif self.mode == 'testFile':
            # By default, show annotated image, save the annotated image, but
            # don't save details of predicted objects
            # To show image
            if self.showImage is None:
                self.showImage = True
            # To save annotated images
            if self.saveAnnotatedImage is None:
                self.saveAnnotatedImage = True
            # To save annotated XML
            if self.saveAnnotatedXML is None:
                self.saveAnnotatedXML = False
            # Test YOLO on self.inputFile
            self.yoloTestFile()
        # Else, if YOLO is to be tested on a database
        elif self.mode == 'testDB':
            # By default, don't show annotated image, but save the annotated
            #  image and details of predicted objects
            # To show image
            if self.showImage is None:
                self.showImage = False
            # To save annotated images
            if self.saveAnnotatedImage is None:
                self.saveAnnotatedImage = True
            # To save annotated XML
            if self.saveAnnotatedXML is None:
                self.saveAnnotatedXML = True
            # Test YOLO on all files in self.inputFolder
            self.yoloTestDB()
        else:
            # TODO: train mode
            pass

    # Builds the computational graph for the network
    def buildGraph(self):
        # Print
        if self.verbose:
            print('Building Yolo Graph....')
        # Input placeholder
        self.x = tf.placeholder('float32', [None, 448, 448, 3])
        # conv1, pool1
        self.conv1 = self.conv_layer(1, self.x, 64, 7, 2)
        self.pool1 = self.maxpool_layer(2, self.conv1, 2, 2)
        # size reduced to 64x112x112
        # conv2, pool2
        self.conv2 = self.conv_layer(3, self.pool1, 192, 3, 1)
        self.pool2 = self.maxpool_layer(4, self.conv2, 2, 2)
        # size reduced to 192x56x56
        # conv3, conv4, conv5, conv6, pool3
        self.conv3 = self.conv_layer(5, self.pool2, 128, 1, 1)
        self.conv4 = self.conv_layer(6, self.conv3, 256, 3, 1)
        self.conv5 = self.conv_layer(7, self.conv4, 256, 1, 1)
        self.conv6 = self.conv_layer(8, self.conv5, 512, 3, 1)
        self.pool3 = self.maxpool_layer(9, self.conv6, 2, 2)
        # size reduced to 512x28x28
        # conv7 - conv16, pool4
        self.conv7 = self.conv_layer(10, self.pool3, 256, 1, 1)
        self.conv8 = self.conv_layer(11, self.conv7, 512, 3, 1)
        self.conv9 = self.conv_layer(12, self.conv8, 256, 1, 1)
        self.conv10 = self.conv_layer(13, self.conv9, 512, 3, 1)
        self.conv11 = self.conv_layer(14, self.conv10, 256, 1, 1)
        self.conv12 = self.conv_layer(15, self.conv11, 512, 3, 1)
        self.conv13 = self.conv_layer(16, self.conv12, 256, 1, 1)
        self.conv14 = self.conv_layer(17, self.conv13, 512, 3, 1)
        self.conv15 = self.conv_layer(18, self.conv14, 512, 1, 1)
        self.conv16 = self.conv_layer(19, self.conv15, 1024, 3, 1)
        self.pool4 = self.maxpool_layer(20, self.conv16, 2, 2)
        # size reduced to 1024x14x14
        # conv17 - conv24
        self.conv17 = self.conv_layer(21, self.pool4, 512, 1, 1)
        self.conv18 = self.conv_layer(22, self.conv17, 1024, 3, 1)
        self.conv19 = self.conv_layer(23, self.conv18, 512, 1, 1)
        self.conv20 = self.conv_layer(24, self.conv19, 1024, 3, 1)
        self.conv21 = self.conv_layer(25, self.conv20, 1024, 3, 1)
        self.conv22 = self.conv_layer(26, self.conv21, 1024, 3, 2)
        self.conv23 = self.conv_layer(27, self.conv22, 1024, 3, 1)
        self.conv24 = self.conv_layer(28, self.conv23, 1024, 3, 1)
        # size reduced to 1024x7x7
        # fc1, fc2, fc3
        self.fc1 = self.fc_layer(29, self.conv24, 512,
                                 flatten=True, linear=False)
        self.fc2 = self.fc_layer(
            30, self.fc1, 4096, flatten=False, linear=False)
        self.fc3 = self.fc_layer(
            31, self.fc2, 1470, flatten=False, linear=True)
        # Run session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weightFile)
        # Print
        if self.verbose:
            print('Loading Complete \n')

    # Test YOLO live
    def yoloTestLive(self):
        # To capture video
        cap = cv2.VideoCapture(0)
        # Try capturing video and performing YOLO on frames
        try:
            # Capture video in a loop
            while(True):
                # Capture a frame
                ret, frame = cap.read()
                # Detect objects
                annotatedImage, predictedObjects = self.detectFromImage(
                    frame)
                # Show image
                if self.showImage:
                    cv2.imshow('YOLO Detection', annotatedImage)
                    # Press 'q' to quit
                    if cv2.waitKey(1) and 0xFF == ord('q'):
                        print("YOLO stopped by pressing 'q'.")
                        break
                # Save annotated image
                if self.saveAnnotatedImage:
                    cv2.imwrite('liveImageAnnotations.jpg', annotatedImage)
                # Save the parameters of detected objects in xml format
                if self.saveAnnotatedXML:
                    xmlFileName = 'liveImagePredictions.xml'
                    self.saveXML(xmlFileName, predictedObjects)
        # Press Ctrl+C to quit
        except KeyboardInterrupt:
            print("YOLO stopped via keyboard interrupt.")
        # If video could not be captured
        except:
            print("Could not capture video...!")
        cap.release()
        cv2.destroyAllWindows()

    # Test YOLO on a file
    def yoloTestFile(self):
        # Detect objects
        annotatedImage, predictedObjects = self.detectFromFile(
            self.inputFile)
        # Show image
        if self.showImage:
            cv2.imshow('YOLO Detection', annotatedImage)
            cv2.waitKey(10)
        # Save annotated image
        if self.saveAnnotatedImage:
            cv2.imwrite(self.outputFile, annotatedImage)
        # Save the parameters of detected objects in xml format
        if self.saveAnnotatedXML:
            xmlFileName = os.path.join(
                self.textOutputFolder,
                self.outputFile.split('.')[0] + '.xml')
            self.saveXML(xmlFileName, predictedObjects)

    # Test YOLO on a database
    def yoloTestDB(self):
        # For each file in database
        for fileName in os.listdir(self.inputFolder):
            # File path
            inputFile = os.path.join(self.inputFolder, fileName)
            # Detect object
            annotatedImage, predictedObjects = self.detectFromFile(
                inputFile)
            # Show image
            if self.showImage:
                cv2.imshow('YOLO Detection', annotatedImage)
                cv2.waitKey(1)
            # Save annotated image
            if self.saveAnnotatedImage:
                outputFileName = os.path.join(self.outputFolder, fileName)
                cv2.imwrite(outputFileName, annotatedImage)
            # Save the parameters of detected objects in xml format
            if self.saveAnnotatedXML:
                xmlFileName = os.path.join(
                    self.textOutputFolder, fileName.split('.')[0] + '.xml')
                self.saveXML(xmlFileName, predictedObjects)

    # To save XML file with details of predicted object
    def saveXML(outputTextFileName, predictedObjects):
        if self.verbose:
            print('Saving xml file', outputTextFileName)
        # root element
        root = ET.Element("annotation")
        # annotation.filename
        ET.SubElement(root, "filename").text = fileName
        # For each predicted object
        for i in range(len(predictedObjects)):
            # annotation.object
            predObject = ET.SubElement(root, "object")
            # annotation.object.name
            ET.SubElement(
                predObject, "name").text = predictedObjects[i][0]
            # annotation.object.confidence
            ET.SubElement(predObject, "confidence").text = str(
                predictedObjects[i][5])
            # annotation.object.bndBox
            bndBox = ET.SubElement(predObject, "bndBox")
            # annotation.object.bndBox.xmin
            ET.SubElement(bndBox, "xmin").text = str(
                predictedObjects[i][1])
            # annotation.object.bndBox.ymin
            ET.SubElement(bndBox, "ymin").text = str(
                predictedObjects[i][2])
            # annotation.object.bndBox.xmax
            ET.SubElement(bndBox, "xmax").text = str(
                predictedObjects[i][3])
            # annotation.object.bndBox.ymax
            ET.SubElement(bndBox, "ymax").text = str(
                predictedObjects[i][4])
        # Making the xml string
        xmlString = minidom.parseString(
            ET.tostring(root)).toprettyxml(indent="   ")
        # Saving the xml file
        with open(outputTextFileName, 'w') as f:
            f.write(xmlString.encode('utf-8'))

    # Annotate image read from file
    def detectFromFile(self, fileName):
        # Print
        if self.verbose:
            print('Detecting object from :', fileName)
        # Read image from file
        imageMatrix = cv2.imread(fileName)
        # Detect objects in image
        return self.detectFromImage(imageMatrix)

    # Detect objects in image
    def detectFromImage(self, imageMatrix):
        image = imageMatrix
        self.imageHeight, self.imageWidth, _ = imageMatrix.shape
        # Resize the image as required by network
        # Make image shape 1x448x448x3
        # Normalize the image values between -1 and 1
        imageMatrix = np.expand_dims((np.asarray(cv2.cvtColor(cv2.resize(
                                imageMatrix, (448, 448)), cv2.COLOR_BGR2RGB),
                                dtype='float32') / 255.) * 2. - 1., axis=0)
        netOutput = self.sess.run(self.fc3, feed_dict={self.x: imageMatrix})
        self.result = self.interpretOutput(netOutput)
        return self.annotateImage(image, self.result)

    def interpretOutput(self, netOutput):
        '''
        Threshold the confidence for all classes, apply Non-Maximum supression
        '''
        # print(sum(sum(netOutput)))
        # to fill in the probability of every class
        classProbability = np.zeros(
            [self.numOfGridsIn1D, self.numOfGridsIn1D, self.numOfBoxesPerGrid,
            self.numOfClasses])
        # print(netOutput[0:980].shape)
        # this should be called objectProbability
        classConditionalProbability = np.reshape(
            netOutput[:, 0:980], [self.numOfGridsIn1D, self.numOfGridsIn1D,
            self.numOfClasses])
        classConfidence = np.reshape(netOutput[:, 980:1078], [
                                     self.numOfGridsIn1D, self.numOfGridsIn1D,
                                     self.numOfBoxesPerGrid])
        # (x, y, w, h) of the two bounding boxes predicted by the network
        boxData = np.reshape(netOutput[:, 1078:], [
                             self.numOfGridsIn1D, self.numOfGridsIn1D,
                             self.numOfBoxesPerGrid, 4])
        for i in range(self.numOfBoxesPerGrid):
            for j in range(self.numOfClasses):
                classProbability[:, :, i, j] = np.multiply(
                    classConditionalProbability[:, :, j],
                    classConfidence[:, :, i])
        offset = np.transpose(np.reshape([np.arange(7)] * 14,
                                         (self.numOfBoxesPerGrid,
                                          self.numOfGridsIn1D,
                                          self.numOfGridsIn1D)), (1, 2, 0))
        # Changing x and y coordinates from model representation to image
        # representation
        boxData[:, :, :, 0] = ((boxData[:, :, :, 0] + offset[:, :, :]) \
            * 1.0 / self.numOfGridsIn1D) * self.imageWidth
        boxData[:, :, :, 1] = ((boxData[:, :, :, 1] + \
            np.transpose(offset, (1, 0, 2))) * 1.0 / self.numOfGridsIn1D) * \
            self.imageHeight
        # Changing width and height from model representation to image
        # representation, square root of the width and height is predicted
        # because small error in small boxes matter much more than small errors
        # in large boxes
        boxData[:, :, :, 2] = np.multiply(boxData[:, :, :, 2], boxData[
                                          :, :, :, 2]) * self.imageWidth
        boxData[:, :, :, 3] = np.multiply(boxData[:, :, :, 3], boxData[
                                          :, :, :, 3]) * self.imageHeight
        # Threhold the class probability
        filterClassProbability = np.array(
            classProbability >= self.minClassProbability, dtype='bool')
        filteredProbability = classProbability[filterClassProbability]
        filterProbabilityIndex = np.nonzero(filterClassProbability)
        filteredBoxes = boxData[filterProbabilityIndex[
            0], filterProbabilityIndex[1], filterProbabilityIndex[2]]
        filteredClasses = np.argmax(filterClassProbability,
                                    axis=3)[filterProbabilityIndex[0],
                                            filterProbabilityIndex[1],
                                            filterProbabilityIndex[2]]
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
                if self.iou(filteredBoxes[i],
                            filteredBoxes[j]) > self.IoUThreshold:
                    # print('Rejecting Box' + str(j))
                    filteredProbability[j] = 0.0
        filterIoU = np.array(filteredProbability > 0.0, dtype='bool')
        filteredProbability = filteredProbability[filterIoU]
        filteredBoxes = filteredBoxes[filterIoU]
        filteredClasses = filteredClasses[filterIoU]
        # list of bounding boxes with class,x,y,w,h,probability
        result = []
        for i in range(len(filteredBoxes)):
            result.append([self.classes[filteredClasses[i]],
                                        filteredBoxes[i][0],
                                        filteredBoxes[i][1],
                                        filteredBoxes[i][2],
                                        filteredBoxes[i][3],
                                        filteredProbability[i]])
        return result

    def annotateImage(self, image, results):
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
            if self.verbose:
                print('Class : ' + results[i][0] + ', [x, y, w, h] [' +
                    str(x) + ', ' + str(y) + ', ' + str(w) + ', ' + str(h) +
                    '] Confidence : ' + str(results[i][5]))
            # Each class must have a unique color
            color = tuple([(j * (1 + self.classes.index(results[i][0])) % 255) \
                    for j in self.seed])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            if ymin <= 20:
                cv2.rectangle(image, (xmin, ymin),
                              (xmax, ymin + 20), color, -1)
                cv2.putText(image, results[i][0] + ': %.2f' % results[i][5],
                            (xmin + 5, ymin + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                cv2.rectangle(image, (xmin, ymin),
                              (xmax, ymin - 20), color, -1)
                cv2.putText(image, results[i][0] + ': %.2f' % results[i][5],
                            (xmin + 5, ymin - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # cv2.rectangle(image, (x - w, y - h), (x + w, y + h), color, 3)
            # cv2.rectangle(image, (x - w, y - h - 20), (x + w, y - h),
            #               (125, 125, 125), -1)
            # cv2.putText(image, results[i][0] + ': %.2f' % results[i][5],
            #             (x - w + 5, y - h - 7),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            objectParameters = [results[i][0], xmin,
                                ymin, xmax, ymax, results[i][5]]
            predictedObjects.append(objectParameters)
        return image, predictedObjects
        # if self.outputFile:
        #    cv2.imwrite(self.outputFile,image)

    def iou(self, boxA, boxB):
        intersectionX = max(0, min(boxA[0] + boxA[2] * 0.5,
                                   boxB[0] + boxB[2] * 0.5) - \
                                max(boxA[0] - boxA[2] * 0.5,
                                    boxB[0] - boxB[2] * 0.5))
        intersectionY = max(0, min(boxA[1] + boxA[3] * 0.5,
                                   boxB[1] + boxB[3] * 0.5) - \
                                max(boxA[1] - boxA[3] * 0.5,
                                    boxB[1] - boxB[3] * 0.5))
        intersection = intersectionX * intersectionY
        union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - intersection
        # print(intersection, union, intersection / union)
        return intersection / union

    def iouTrain(self, boxA, boxB, realBox):
        '''
        Calculate the IoU in training phase, to get the box
        (out of N boxes per grid) responsible for ground truth box
        '''
        iou1 = tf.reshape(iouTrainUnit(boxA, realBox), [-1, 7, 7, 1])
        iou2 = tf.reshape(iouTrainUnit(boxB, realBox), [-1, 7, 7, 1])
        return tf.concat([iou1, iou2], 3)

    # Calculate IoU between boxA and realBox
    def iouTrainUnit(self, boxA, realBox):
        # make sure that the representation of box matches input
        intersectionX = tf.minimum(boxA[:, :, :, 0] + 0.5 * boxA[:, :, :, 2],
                            realBox[:, :, :, 0] + 0.5 * realBox[:, :, :, 2]) - \
                        tf.maximum(boxA[:, :, :, 0] - 0.5 * boxA[:, :, :, 2],
                           realBox[:, :, :, 0] - 0.5 * realBox[:, :, :, 2])
        intersectionY = tf.minimum(boxA[:, :, :, 1] + 0.5 * boxA[:, :, :, 3],
                            realBox[:, :, :, 1] + 0.5 * realBox[:, :, :, 3]) - \
                        tf.maximum(boxA[:, :, :, 1] - 0.5 * boxA[:, :, :, 3],
                            realBox[:, :, :, 1] - 0.5 * realBox[:, :, :, 3])
        intersection = tf.multiply(tf.maximum(
            0, intersectionX), tf.maximum(0, intersectionY))
        union = tf.subtract(
                    tf.multiply(boxA[:, :, :, 1], boxA[:, :, :, 3]) + \
                        tf.multiply(realBox[:, :, :, 1], realBox[:, :, :, 3]),
                    intersection)
        iou = tf.divide(intersection, union)
        return iou

    # TODO
    def train_network(self):
        # determine which bounding box is responsible for prediction
        # save the weights after each epoch
        if self.trainData:
            if self.verbose:
                print('Started training...')

            for epoch in range(135):
                pass
                # save the model
        else:
            if self.verbose:
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
            predicted, [-1, self.numOfGridsIn1D, self.numOfGridsIn1D, 30])
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
        # class loss (loss due to misjudgement in class of the object detected)
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

    # Conv layer
    def conv_layer(self, index, inputMatrix, numOfFilters, sizeOfFilter,
        stride):
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
        numOfChannels = inputMatrix.get_shape()[3]
        # int with numberOfChannels
        weight = tf.Variable(tf.truncated_normal(
            [sizeOfFilter, sizeOfFilter, int(numOfChannels), numOfFilters],
            stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=[numOfFilters]))
        padSize = sizeOfFilter // 2
        paddedInput = tf.pad(
            inputMatrix, ([[0, 0], [padSize, padSize], [padSize, padSize],
                           [0, 0]]))
        conv = tf.nn.conv2d(paddedInput, weight, strides=[
                            1, stride, stride, 1], padding='VALID',
                            name=str(index) + '_conv')
        conv_bias = tf.add(conv, bias, name=str(index) + '_conv')
        if self.verbose:
            print(' Layer %d Type: Conv Size: %dx%d Stride: %d No.Filters: %d \
                Input Channels : %d' % (index, sizeOfFilter, sizeOfFilter,
                                        stride, numOfFilters, numOfChannels))
        # leaky relu as mentioned in YOLO paper
        return tf.maximum(self.leakyReLUAlpha * conv_bias, conv_bias,
                          name=str(index) + '_leaky_relu')

    # Max pool layer
    def maxpool_layer(self, index, inputMatrix, sizeOfFilter, stride):
        if self.verbose:
            print(' Layer %d Type: Maxpool Size: %dx%d Stride: %d' %
                  (index, sizeOfFilter, sizeOfFilter, stride))
        maxpool = tf.nn.max_pool(inputMatrix,
                                 ksize=[1, sizeOfFilter, sizeOfFilter, 1],
                                 strides=[1, sizeOfFilter, sizeOfFilter, 1],
                                 padding='SAME', name=str(index) + '_maxpool')
        return maxpool

    # FC layer
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
        # W, b
        weight = tf.Variable(tf.truncated_normal(
            [inputDimension, outputNeurons], stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=[outputNeurons]))
        if self.verbose:
            print(' Layer %d Type: FullyConnected InSize: %d OutSize %d \
              Linear: %d' % (index, inputDimension, outputNeurons, int(linear)))
        # linear or leaky relu activation
        if linear:
            return tf.add(tf.matmul(inputMatrixAdjust, weight), bias,
                          name=str(index) + '_fc')
        else:
            answer = tf.add(tf.matmul(inputMatrixAdjust, weight), bias,
                            name=str(index) + '_fc')
            return tf.maximum(self.leakyReLUAlpha * answer, answer,
                              name=str(index) + '_fc')


def main():
    yolo = Yolo()
    # cv2.waitKey(1000)

if __name__ == '__main__':
    main()
