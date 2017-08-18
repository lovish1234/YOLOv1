import numpy as np
import os
import xml.etree.cElementTree as ET

predictedFolder = 'VOC2007/test/outputAnnotations/'
groundTruthFolder = '../VOC2007/test/Annotations/'

numOfClasses = 20


classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
           "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

IoUThreshold = 0.5


# Function to calculate mAP
def calculate_mAP():

    # Init
    fileList = os.listdir(groundTruthFolder)
    truePositives = np.zeros(numOfClasses)
    falsePositives = np.zeros(numOfClasses)
    falseNegatives = np.zeros(numOfClasses)

    # For each image
    for file in fileList:

        # File paths
        predictedFilePath = os.path.join(predictedFolder, file)
        groundTruthFilePath = os.path.join(groundTruthFolder, file)

        # Loading Predictions and Ground Truths
        predictedObject = ET.parse(predictedFilePath).findall('object')
        groundTruthObject = ET.parse(groundTruthFilePath).findall('object')

        # Init dictionaries to save predictions and ground truth per class
        dictConfidences = {}
        dictPredicted = {}
        dictGT = {}
        dictMask = {}
        for classIdx in range(numOfClasses):
            dictPredicted[classIdx] = []
            dictGT[classIdx] = []
            dictMask[classIdx] = []

        # Extract the predicted boxes from xml file
        for item in predictedObject:
            itemClass = item.find('name').text
            classIdx = classes.index(itemClass)
            confidence = float(item.find('confidence').text)
            xmin = int(item.find('bndbox').find('xmin').text)
            xmax = int(item.find('bndbox').find('xmax').text)
            ymin = int(item.find('bndbox').find('ymin').text)
            ymax = int(item.find('bndbox').find('ymax').text)
            # Append Confidences
            dictConfidences[classIdx].append(confidence)
            # Append dictPredicted
            dictPredicted[classIdx].append(
                [xmin, xmax, ymin, ymax])

        # Sort each of the dictPredicted values in decreasing order of
        # confidence values
        for classIdx in range(numOfClasses):
            dictPredicted[classIdx] = [pred for (c, pred) in sorted(
                zip(dictConfidences[classIdx], dictPredicted[classIdx]), reverse=True)]

        # Extract the ground truth boxes from xml file
        for item in groundTruthObject:
            itemClass = item.find('name').text
            classIndex = classes.index(itemClass)
            xmin = int(item.find('bndbox').find('xmin').text)
            xmax = int(item.find('bndbox').find('xmax').text)
            ymin = int(item.find('bndbox').find('ymin').text)
            ymax = int(item.find('bndbox').find('ymax').text)
            # Append dictGT
            dictGT[classIndex].append([xmin, xmax, ymin, ymax])
            # To find out if a ground truth exists for an object of a class,
            # and if a prediction has been made corresponding to that object,
            # we shall append dictMask with 0 if GT exists,
            # and modify it to 1 when a prediction corresponds with it
            dictMask[classIndex].append(0)

        # For each class
        for classIdx in range(numOfClasses):
            # For each predicted bounding box
            for predBndbox in dictPredicted[classIdx]:
                # To find the ground truth bounding box corresponding with the
                # predicted bounding box
                maxIoU = 0
                maxIndex = -1
                # For each ground truth bounding box
                for GTbndboxIdx in range(len(dictGT[classIdx])):
                    # Find IoU
                    areaMetric = IoU(dictPredicted[classIdx][predBndbox],
                                     dictGT[classIdx][GTbndboxIdx])
                    # Record that GT bounding box which has maximum IoU with
                    # the predicted bounding box
                    if areaMetric > maxIoU:
                        # print(areaMetric)
                        maxIoU = areaMetric
                        maxIndex = GTbndboxIdx
                # If the IoU between the predicted and GT bounding boxes is
                # greater than IoUThreshold
                if maxIoU > IoUThreshold:
                    # If the object has not been detected before,
                    if dictMask[classIdx][maxIndex] == 0:
                        # Modify dictMask to indicate object has been detected
                        dictMask[classIdx][maxIndex] = 1
                        # This is a true positive
                        truePositives[classIdx] += 1
                    # Else if object has been detected before, since we know
                    # that the current prediction has lesser confidence than the
                    # previous one, so we will consider this a false positive
                    else:
                        falsePositives[classIdx] += 1
                # If the IoU between the predicted and GT bounding boxes is
                # lesser than IoUThreshold, we will consider it a
                # false positive
                else:
                    falsePositives[classIdx] += 1
            # FALSE NEGATIVES
            # For those classes with GT available but no prediction made,
            # we will consider this a false negative
            falseNegatives[
                classIdx] += sum([1 - j for j in dictMask[classIdx]])

    print('True positives in each class:', truePositives)
    print('False positives in each class:', falsePositives)
    print('False negatives in each class:', falseNegatives)

    # Precision in each class
    precisions = truePositives / (truePositives + falsePositives)
    print('Precisions in each class: ', precisions)

    # Average Precision across all classes
    avgPrecision = np.mean(precisions)
    print('mAP =', avgPrecision)


# A function to calculate Intersection over Union (IoU)
# i.e. fraction of common area between two boxes
def IoU(boxA, boxB):
    intersectionX = max(0, min(boxA[1], boxB[1]) - max(boxA[0], boxB[0]))
    intersectionY = max(0, min(boxA[3], boxB[3]) - max(boxA[2], boxB[2]))
    intersection = intersectionX * intersectionY
    union = (boxA[1] - boxA[0]) * (boxA[3] - boxA[2]) + \
        (boxB[1] - boxB[0]) * (boxB[3] - boxB[2]) - intersection
    # print(intersection, union, intersection * 1.0 / union)
    return intersection * 1. / union

if __name__ == '__main__':
    calculate_mAP()
