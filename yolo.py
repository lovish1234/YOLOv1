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

class Yolo:
    """Implement YOLO for Classifiacation and Detection"""

    numOfClasses = 20

    classes = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor"
    ]

    # bounding box seeds
    seed = [random.randint(1, 1000) for i in range(3)]

    def __init__(self,
                 mode='testVideo',
                 weightFile='weights/yolo_small.ckpt',
                 showImage=None,
                 saveAnnotatedImage=None,
                 saveAnnotatedXML=None,
                 numOfGridsIn1D=7,
                 numOfBoxesPerGrid=2,
                 batchSize=64,
                 verbose=False,
                 debug=False,
                 minClassProbability=0.2,
                 iouThreshold=0.5,
                 lambdaCoordinate=5.0,
                 lambdaNoObject=0.5,
                 leakyReLUAlpha=0.1,
                 inputFile='test/person.jpg',
                 outputFile='test/output.jpg',
                 textOutputFile=None,
                 inputFolder=None,
                 outputFolder=None,
                 textOutputFolder=None):
        """Init function"""
        # Mode to run the Yolo code in
        # {testLive, testFile, testDB, testVideo, train}
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
        # To display debug logs of the program
        self.debug = debug
        # Used to disregard bounding boxes
        self.minClassProbability = minClassProbability
        # Used for non-maximum supression
        self.iouThreshold = iouThreshold
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
        self.build_graph()
        self.init_other_vars()
        # If YOLO is to be tested live
        if self.mode == 'testLive':
            # By default, show annotated images, but don't save
            # annotated image or details of predicted objects
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
            self.yolo_test_live()
        # Else, if YOLO is to be tested on a file
        elif self.mode == 'testFile':
            # By default, show annotated image, save the annotated
            # image, but don't save details of predicted objects
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
            self.yolo_test_file()
        # Else, if YOLO is to be tested on a database
        elif self.mode == 'testDB':
            # By default, don't show annotated image, but save the
            # annotated image and details of predicted objects
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
            self.yolo_test_db()
        # Else, if YOLO is to be tested on a video
        elif self.mode =='testVideo':
            self.yolo_test_video()

        else:
            # TODO: train mode
            pass

    def build_graph(self):
        """Build the computational graph for the network"""
        # Print
        if self.verbose:
            print('Building Yolo Graph....')
        # Reset default graph
        tf.reset_default_graph()
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
            print('Loading Complete')

    def yolo_test_live(self):
        """Test YOLO live"""
        # To capture video
        cap = cv2.VideoCapture(0)
        # Try capturing video and performing YOLO on frames
        try:
            # Capture video in a loop
            while(True):
                # Capture a frame
                ret, frame = cap.read()
                # Detect objects
                annotatedImage, predictedObjects = self.detect_from_image(
                    frame)
                # Show image
                if self.showImage:
                    cv2.imshow('YOLO Detection', annotatedImage)
                    # Press 'q' to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("YOLO stopped by pressing 'q'.")
                        break
                # Save annotated image
                if self.saveAnnotatedImage:
                    cv2.imwrite('liveImageAnnotations.jpg', annotatedImage)
                # Save the parameters of detected objects in xml format
                if self.saveAnnotatedXML:
                    xmlFileName = 'liveImagePredictions.xml'
                    self.save_xml(fileName, xmlFileName, predictedObjects)
        # Press Ctrl+C to quit
        except KeyboardInterrupt:
            print("YOLO stopped via keyboard interrupt.")
        # If video could not be processed
        except:
            print("Could not capture/process video...!")
        # Roll back
        cap.release()
        cv2.destroyAllWindows()

    def yolo_test_file(self):
        """Test YOLO on a file"""
        # Detect objects
        annotatedImage, predictedObjects = self.detect_from_file(
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
            self.save_xml(xmlFileName, predictedObjects)

    def yolo_test_db(self):
        """Test YOLO on a database"""
        # For each file in database
        for fileName in os.listdir(self.inputFolder):
            # File path
            inputFile = os.path.join(self.inputFolder, fileName)
            # Detect object
            annotatedImage, predictedObjects = self.detect_from_file(
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
                self.save_xml(xmlFileName, predictedObjects)
    
    def yolo_test_video(self):
        """Test YOLO on a video"""
        # Open the input video, blocking call
        inputVideo = cv2.VideoCapture(self.inputFile)
		
        # Get infomration about the input video
        codec = int(inputVideo.get(cv2.CAP_PROP_FOURCC))
        fps = int(inputVideo.get(cv2.CAP_PROP_FPS))
        frameWidth = int(inputVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(inputVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Open the output stream
        outputVideo = cv2.VideoWriter(self.outputFile,
                                      codec,
                                      fps,
                                      (frameWidth,frameHeight))
        frameIndex = inputVideo.get(cv2.CAP_PROP_POS_FRAMES)
        totalFrames = inputVideo.get(cv2.CAP_PROP_FRAME_COUNT)
 	 
	avgGrabTime = 0
	avgYoloTime = 0
	avgWriteTime = 0
        
        # For each frame in the video
        while True:
            
            startTime = time.time()
            
            # Calculate the time it takes to grab a frame
            startGrabTime = time.time()
            grabbed, frame = inputVideo.read()
            endGrabTime = time.time() 
	    avgGrabTime+=(endGrabTime-startGrabTime)
	   

            if grabbed:
		
                # Calculate the time it takes to run YOLO pipeline 
		startYoloTime = time.time()
                annotatedFrame, predictedObjects = self.detect_from_image(frame)
		endYoloTime = time.time()
		avgYoloTime+= ( endYoloTime - startYoloTime)

                frameIndex = inputVideo.get(cv2.CAP_PROP_POS_FRAMES)
 	
		currentTime = time.time()
		elapsedTime = currentTime - startTime
		currentFPS = (1)/elapsedTime    
		        	
                #cv2.rectangle(annotatedFrame, (0, 0), (30, 30), (0,0,0), -1)
                cv2.putText(
                        annotatedFrame, 'FPS' + ': %.2f' % currentFPS,
                        (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2
                        )
		
                # Calculate the time it takes to write an annotated frame to video
		startWriteTime = time.time()
                outputVideo.write(annotatedFrame)
		endWriteTime = time.time()
		avgWriteTime +=(endWriteTime - startWriteTime)
	
            else:
                inputVideo.set(cv2.CAP_PROP_POS_FRAMES, frameIndex-1)
                cv2.waitKey(100)

            if frameIndex==totalFrames:
                break
		
        inputVideo.release()
        outputVideo.release()
        cv2.destroyAllWindows()
        
        avgGrabTime/=totalFrames
        avgYoloTime/=totalFrames
        avgWriteTime/=totalFrames

        if self.verbose:
            print ('Average time for extracting compressed video frame : %.3f'  %avgGrabTime)
            print ('Average time for YOLO object detection : %.3f'  %avgYoloTime )
            print ('Average time for writing frame to video : %.3f'  %avgWriteTime)
	       
    def save_xml(self, fileName, outputTextFileName, predictedObjects):
        """To save XML file with details of predicted object"""
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

    def init_other_vars(self):
        """Initialize other relevant variables"""
        self.endIndexOfClassConditionalProbability = self.numOfGridsIn1D \
                                        * self.numOfGridsIn1D * self.numOfClasses
        self.endIndexOfObjectProbability \
            = self.endIndexOfClassConditionalProbability \
            + self.numOfGridsIn1D*self.numOfGridsIn1D*self.numOfBoxesPerGrid
        # Class Conditional Probability: P(class | object),
        self.classConditionalProbability = np.zeros([
            self.numOfGridsIn1D, self.numOfGridsIn1D, self.numOfClasses
            ])
        # P(object): Object probability, i.e. the probability of an
        self.objectProbability = np.zeros([
            self.numOfGridsIn1D, self.numOfGridsIn1D, self.numOfBoxesPerGrid
            ])
        # Box data (x, y, w, h)
        self.boxData = np.zeros([
            self.numOfGridsIn1D, self.numOfGridsIn1D, self.numOfBoxesPerGrid, 4
            ])
        # Offset to add to x and y values to convert from within-grid
        # coordinates to image coordinates
        self.offsetY = np.tile(
            np.arange(self.numOfGridsIn1D)[:, np.newaxis, np.newaxis],
            (1, self.numOfGridsIn1D, self.numOfBoxesPerGrid)
            )
        self.offsetX = np.transpose(self.offsetY, (1, 0, 2))
        # Most probable classes per grid
        self.maxProbableClasses = np.zeros([
            self.numOfGridsIn1D, self.numOfGridsIn1D, self.numOfBoxesPerGrid
            ])
        # Probabilities of most probable classes per grid
        self.maxProbableClassProbabilities = np.zeros([
            self.numOfGridsIn1D, self.numOfGridsIn1D, self.numOfBoxesPerGrid
            ])
        # The probability of an object present, and it being each class
        self.objectClassProbability = np.zeros([
            self.numOfGridsIn1D, self.numOfGridsIn1D, self.numOfBoxesPerGrid,
            self.numOfClasses
            ])

    def detect_from_file(self, fileName):
        """Detect objects in image file"""
        if self.verbose:
            print('Detecting object from :', fileName)
        # Read image from file
        imageMatrix = cv2.imread(fileName)
        # Detect objects in image
        return self.detect_from_image(imageMatrix)

    def detect_from_image(self, imageMatrix):
        """
        Detect objects in image

        Input
        image: raw image to feed into network

        Output
        Annotated images
        """
        image = imageMatrix #TODO
        self.imageHeight, self.imageWidth, _ = imageMatrix.shape
        # Resize the image as required by network
        # Make image shape 1x448x448x3
        # Normalize the image values between -1 and 1
        imageMatrix = np.expand_dims(
                        np.asarray(
                            cv2.cvtColor(
                                cv2.resize(
                                    imageMatrix, (448, 448)
                                ), cv2.COLOR_BGR2RGB
                            ), dtype='float32')/255.*2. - 1., axis=0)
        # Run image through network and get its output
        netOutput = self.sess.run(self.fc3, feed_dict={self.x: imageMatrix})
        # Figure out the object classes and bounding boxes from the
        # network output
        self.result = self.interpret_output(netOutput)
        # Make an annotated image with the classes and bounding boxes
        return self.annotate_image(image, self.result)

    def interpret_output(self, netOutput):
        """
        Calculate bounding boxes of most probable objects

        Input
        image: raw image to feed into network
        netOutput: output of network after feeding with image

        Output
        Objects: their classes and their bounding boxes
        """
        # Class Conditional Probability: P(class | object),
        # i.e. assuming there is an object in the grid being considered,
        # what is the probability the object belongs to each class
        self.classConditionalProbability = np.reshape(
            netOutput[:, :self.endIndexOfClassConditionalProbability],
            [self.numOfGridsIn1D, self.numOfGridsIn1D, self.numOfClasses]
            )
        if self.debug:
            print("classConditionalProbability.")
        # P(object): Object probability, i.e. the probability of an
        # object in each bounding box in each grid
        self.objectProbability = np.reshape(
            netOutput[
                :,
                self.endIndexOfClassConditionalProbability:\
                    self.endIndexOfObjectProbability
                ],
            [self.numOfGridsIn1D, self.numOfGridsIn1D, self.numOfBoxesPerGrid]
            )
        if self.debug:
            print("objectProbability.")
        # objectClassProbability: P(class | object) * P(object), i.e.
        # the probability of an object present, and it being each class
        # Equivalent to:
        #for i in range(self.numOfBoxesPerGrid):
        #    for j in range(self.numOfClasses):
        #        self.objectClassProbability[:, :, i, j] = np.multiply(
        #            self.objectProbability[:, :, i],
        #            self.classConditionalProbability[:, :, j]
        #            )
        # Also equivalent to:
        # for i in range(self.numOfGridsIn1D):
        #     for j in range(self.numOfGridsIn1D):
        #         self.objectClassProbability[i, j] = np.outer(
        #             self.objectProbability[i, j, :],
        #             self.classConditionalProbability[i, j, :]
        #             )
        # Or:
        self.objectClassProbability = np.einsum(
             '...i, ...j',
             self.objectProbability,
             self.classConditionalProbability,
             out=self.objectClassProbability
             )
        if self.debug:
            print("objectClassProbability.")
        # (x, y, w, h) == (<left> <top> <right> <bottom>) of the
        # numOfBoxesPerGrid bounding boxes predicted by the network
        self.boxData = np.reshape(
          netOutput[:, self.endIndexOfObjectProbability:],
          [self.numOfGridsIn1D, self.numOfGridsIn1D, self.numOfBoxesPerGrid, 4]
          )
        # Changing x <left> and y <top> coordinates from within-grid
        # coordinates to image coordinates
        self.boxData[:, :, :, 0] = 1. * \
            (self.boxData[:, :, :, 0]+self.offsetX) * self.imageWidth \
            / self.numOfGridsIn1D
        self.boxData[:, :, :, 1] = 1. * \
            (self.boxData[:, :, :, 1]+self.offsetY) * self.imageHeight \
            / self.numOfGridsIn1D
        # Changing width and height from model representation to image
        # representation, square root of the width and height is predicted
        # because small error in small boxes matter much more than small errors
        # in large boxes
        self.boxData[:, :, :, 2] = self.imageWidth * np.multiply(
                                                    self.boxData[:, :, :, 2],
                                                    self.boxData[:, :, :, 2]
                                                    )
        self.boxData[:, :, :, 3] = self.imageHeight * np.multiply(
                                                    self.boxData[:, :, :, 3],
                                                    self.boxData[:, :, :, 3]
                                                    )
        if self.debug:
            print("boxData.")
        # Find out the index of the class with maximum probability for
        # every object/bounding box
        self.maxProbableClasses = np.argmax(self.objectClassProbability, axis=3)
        if self.debug:
            print("maxProbableClasses.")
        # Find out the probability value of these max classes
        self.maxProbableClassProbabilities = np.max(
                                            self.objectClassProbability, axis=3
                                            )
        if self.debug:
            print("maxProbableClassProbabilities.")
        # Eliminate those objects whose class probabilities are lesser
        # than minClassProbability
        thresholdedClassesIndex = np.where(
            self.maxProbableClassProbabilities >= self.minClassProbability
            )
        if self.debug:
            print("thresholdedClassesIndex.")
        # The classes
        thresholdedClasses = self.maxProbableClasses[thresholdedClassesIndex]
        if self.debug:
            print("thresholdedClasses.")
        # The class probabilities
        thresholdedClassProbabilities = self.maxProbableClassProbabilities[
                                                        thresholdedClassesIndex
                                                        ]
        if self.debug:
            print("thresholdedClassProbabilities.")
        # Find out the boxes corresponding to the filtered objects
        thresholdedBoxes = self.boxData[
                                thresholdedClassesIndex[0],
                                thresholdedClassesIndex[1],
                                thresholdedClassesIndex[2]
                                ]
        if self.debug:
            print("thresholdedBoxes.")
        # Sort the boxes and classes based on probabilities
        sortOrder = np.argsort(thresholdedClassProbabilities)[::-1]
        if self.debug:
            print("sortOrder.")
        thresholdedClassProbabilities \
            = thresholdedClassProbabilities[sortOrder]
        thresholdedBoxes = thresholdedBoxes[sortOrder]
        thresholdedClasses = thresholdedClasses[sortOrder]
        if self.debug:
            print("thresholdedClasses.")
        # Non-maximum supression
        for box1 in range(len(thresholdedClassProbabilities)):
            if thresholdedClassProbabilities[box1] == 0.:
                continue
            for box2 in range(box1 + 1, len(thresholdedClassProbabilities)):
                if self.iou(thresholdedBoxes[box1], thresholdedBoxes[box2]) \
                        > self.iouThreshold:
                    thresholdedClassProbabilities[box2] = 0.
        # Non-suppressed boxes
        nonSuppressedIndex = np.where(thresholdedClassProbabilities > 0)
        thresholdedClassProbabilities \
            = thresholdedClassProbabilities[nonSuppressedIndex]
        thresholdedBoxes = thresholdedBoxes[nonSuppressedIndex]
        thresholdedClasses = thresholdedClasses[nonSuppressedIndex]
        if self.debug:
            print("thresholdedClasses[nonSuppressedIndex].")
        # Results
        result = []
        for i in range(len(thresholdedClasses)):
            result.append([self.classes[thresholdedClasses[i]],
                                        thresholdedBoxes[i][0],
                                        thresholdedBoxes[i][1],
                                        thresholdedBoxes[i][2],
                                        thresholdedBoxes[i][3],
                                        thresholdedClassProbabilities[i]])
        return result

    def annotate_image(self, image, results):
        """ Annotate image with results from netOutput"""
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
            xmax = imageWidth - 3 if not min(x + w - imageWidth, 0) \
                                    else (x + w)
            ymin = 1 if not max(y - h, 0) else (y - h)
            ymax = imageHeight - 3 if not min(y + h - imageHeight, 0) \
                                    else (y + h)
            if self.verbose:
                print('Class : ' + results[i][0] + ', [x, y, w, h] [' +
                    str(x) + ', ' + str(y) + ', ' + str(w) + ', ' + str(h) +
                    '] Confidence : ' + str(results[i][5]))
            
            # Each class must have a unique color
            color = tuple([(j * (1+self.classes.index(results[i][0])) % 255) \
                    for j in self.seed])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            if ymin <= 20:
                cv2.rectangle(
                    image, (xmin, ymin), (xmax, ymin + 20), color, -1
                    )
                cv2.putText(
                    image, results[i][0] + ': %.2f' % results[i][5],
                    (xmin+5, ymin+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2
                    )
            else:
                cv2.rectangle(image, (xmin, ymin), (xmax, ymin-20), color, -1)
                cv2.putText(
                    image, results[i][0] + ': %.2f' % results[i][5],
                    (xmin+5, ymin-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2
                    )
            objectParameters = [
                results[i][0], xmin, ymin, xmax, ymax, results[i][5]
                ]
            predictedObjects.append(objectParameters)
        return image, predictedObjects
        # if self.outputFile:
        #    cv2.imwrite(self.outputFile,image)

    def iou(self, boxA, boxB):
        """Calculate IoU between boxA and boxB"""
        intersectionX = max(0, min(
                                boxA[0] + boxA[2]*0.5, boxB[0] + boxB[2]*0.5
                                ) - max(
                                        boxA[0] - boxA[2]*0.5,
                                        boxB[0] - boxB[2]*0.5
                                        ))
        intersectionY = max(0, min(
                                boxA[1] + boxA[3]*0.5,
                                boxB[1] + boxB[3]*0.5
                                ) - max(
                                        boxA[1] - boxA[3]*0.5,
                                        boxB[1] - boxB[3]*0.5
                                        ))
        intersection = intersectionX * intersectionY
        union = boxA[2]*boxA[3] + boxB[2]*boxB[3] - intersection
        # print(intersection, union, intersection / union)
        return intersection / union

    def iou_train(self, boxA, boxB, realBox):
        """
        Calculate IoU between boxA and realBox

        Calculate the IoU in training phase, to get the box (out of N
            boxes per grid) responsible for ground truth box
        """
        iou1 = tf.reshape(iou_train_unit(boxA, realBox), [-1, 7, 7, 1])
        iou2 = tf.reshape(iou_train_unit(boxB, realBox), [-1, 7, 7, 1])
        return tf.concat([iou1, iou2], 3)

    def iou_train_unit(self, boxA, realBox):
        """
        Calculate IoU between boxA and realBox
        """
        # make sure that the representation of box matches input
        intersectionX = tf.minimum(
            boxA[:, :, :, 0] + 0.5*boxA[:, :, :, 2],
            realBox[:, :, :, 0] + 0.5*realBox[:, :, :, 2]
            ) - tf.maximum(
                    boxA[:, :, :, 0] - 0.5*boxA[:, :, :, 2],
                    realBox[:, :, :, 0] - 0.5*realBox[:, :, :, 2]
                    )
        intersectionY = tf.minimum(
            boxA[:, :, :, 1] + 0.5*boxA[:, :, :, 3],
            realBox[:, :, :, 1] + 0.5*realBox[:, :, :, 3]
            ) - tf.maximum(
                        boxA[:, :, :, 1] - 0.5*boxA[:, :, :, 3],
                        realBox[:, :, :, 1] - 0.5*realBox[:, :, :, 3]
                        )
        intersection = tf.multiply(
            tf.maximum(0, intersectionX), tf.maximum(0, intersectionY)
            )
        union = tf.subtract(
                    tf.multiply(
                        boxA[:, :, :, 1], boxA[:, :, :, 3]) + tf.multiply(
                            realBox[:, :, :, 1], realBox[:, :, :, 3]
                            ),
                        intersection
                        )
        iou = tf.divide(intersection, union)
        return iou

    # TODO
    def train_network(self):
        """
        Determine which bounding box is responsible for prediction.
        
        Save the weights after each epoch.
        """
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
        """
        Calculate the total loss for gradient descent.
        
        For each ground truth object, loss needs to be calculated.
        It is assumed that each image consists of only one object.

        Predicted
        0-19 CLass prediction
        20-21 Confidence that objects exist in bbox1 or bbox2 of grid
        22-29 Coordinates for bbo1, followed by those of bbox2 

        Real
        0-19 Class prediction (One-Hot Encoded)
        20-23 Ground truth coordinates for that box
        24-72 Cell has an object/no object (Only one can be is 1)
        """
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
        # Calulate loss along the 4th axis, localFirstBoxes -1x7x7x1
        # Think there should be a simpler method to do this
        lossFirstBoxes = tf.reduce_sum(
            tf.square(predictedFirstBoxes - groundTruthBoxes), 3)
        lossSecondBoxes = tf.reduce_sum(
            tf.square(predictedSecondBoxes - groundTruthBoxes), 3)
        # Computing which box (bbox1 or bbox2) is responsible for
        # detection
        IOU = iou_train(predictedFirstBoxes,
                       predictedSecondBoxes, groundTruthBoxes)
        responsbileBox = tf.greater(IOU[:, :, :, 0], IOU[:, :, :, 1])
        # Suppose it is known which iou is greater,
        # coordinate loss (loss due to difference in coordinates of
        # predicted-responsible and real box)
        coordinateLoss = tf.where(
            responsibleBox, lossFirstBoxes, lossSecondBoxes)
        # why do we need to reshape it
        coordinateLoss = tf.reshape(coordinateLoss, [-1, 7, 7, 1])
        # count the loss only if the object is in the groundTruth grid
        # gives a sparse -1x7x7x1 matrix, only one element would be nonzero in
        # each slice
        coorinateLoss = self.lambdaCoordinate * \
            tf.multiply(groundTruthGrid, coordinateLoss)
        # object loss (loss due to difference in object confidence)
        # only take the objectLoss of the predicted grid with higher IoU is
        # responsible for the object
        objectLoss = tf.square(predictedObjectConfidence - groundTruthGrid)
        objectLoss = tf.where(responsibleBox, objectLoss[
                              :, :, :, 0], objectLoss[:, :, :, 1])
        tempObjectLoss = tf.reshape(objectLoss, [-1, 7, 7, 1])
        objectLoss = tf.multiply(groundTruthGrid, tempObjectLoss)
        # class loss (loss due to misjudgement in class of the object
        # detected
        classLoss = tf.square(predictedClasses - groundTruthClasses)
        classLoss = tf.reduce_sum(
            tf.mul(groundTruthGrid, classLoss), reduction_indices=3)
        classLoss = tf.reshape(classLoss, [-1, 7, 7, 1])
        # no-object loss, decrease the confidence where there is no
        # object in the ground truth
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
        """
        Convolve inputMatrix with filters

        Input
        index : index of the layer within the network
        inputMatrix : self-Explainatory, the input
        numberOfFilters : number of channel outputs
        sizeOfFilter : defines the receptive field of a neuron
        stride : self-Exmplainatory, pixels to skip

        Output
        Matrix the size of input[0]xinput[1]xnoOfFilters
        """
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
            print(' Layer %d Type: Conv Size: %dx%d Stride: %d No.Filters: %d '
                'Input Channels : %d' % (index, sizeOfFilter, sizeOfFilter,
                                        stride, numOfFilters, numOfChannels))
        # leaky relu as mentioned in YOLO paper
        return tf.maximum(self.leakyReLUAlpha * conv_bias, conv_bias,
                          name=str(index) + '_leaky_relu')

    def maxpool_layer(self, index, inputMatrix, sizeOfFilter, stride):
        """
        Pool inputMatrix into lesser dimensions

        Input
        index : index of the layer within the network
        inputMatrix : self-Explainatory, the input
        sizeOfFilter : defines the receptive field of a neuron
        stride : self-Exmplainatory, pixels to skip 

        Output
        Matrix the size of (input0/stride)x(input1/stride)xnoOfFilters
        """
        if self.verbose:
            print(' Layer %d Type: Maxpool Size: %dx%d Stride: %d' %
                  (index, sizeOfFilter, sizeOfFilter, stride))
        maxpool = tf.nn.max_pool(inputMatrix,
                                 ksize=[1, sizeOfFilter, sizeOfFilter, 1],
                                 strides=[1, sizeOfFilter, sizeOfFilter, 1],
                                 padding='SAME', name=str(index) + '_maxpool')
        return maxpool

    def fc_layer(self, index, inputMatrix, outputNodes, flatten, linear):
        """
        Fully connected neural layer between inputMatrix and output

        Input
        index : index of the layer within the network
        inputMatrix : self-Explainatory, the input
        sizeOfFilter : defines the receptive field of a neuron
        stride : self-Exmplainatory, pixels to skip 

        Output
        Matrix the size of (input0/stride)x(input1/stride)xnoOfFilters
        """
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
            [inputDimension, outputNodes], stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=[outputNodes]))
        if self.verbose:
            print(' Layer %d Type: FullyConnected InSize: %d OutSize %d '
              'Linear: %d' % (index, inputDimension, outputNodes, int(linear)))
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
