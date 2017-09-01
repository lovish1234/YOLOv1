import matplotlib.pyplot as plt
import numpy as np
import os
import xml.etree.cElementTree as ET

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
           "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

currDir = os.path.dirname(os.path.realpath(__file__))

predictedFolder = os.path.join(currDir, 'test/outputAnnotations/')
groundTruthFolder = os.path.join(currDir, 'test/Annotations/')

numOfClasses = 20

IoUThreshold = 0.5


def calculate_mAP(predictedFolder=predictedFolder,
                    groundTruthFolder=groundTruthFolder,
                    IoUThreshold=IoUThreshold,
                    plotPCCurve=False):
    # Init
    fileListPredicted = os.listdir(predictedFolder)
    fileListGT = os.listdir(groundTruthFolder)
    # Each index of values in key:value pair would consist
    # of [confidence, image_name, [x,y,w,h]],key:class
    dictPredicted = {}
    for classId in range(numOfClasses):
        dictPredicted[classId] = []
    # Total numbers of objects predicted can help in
    # calculating recall as it equals TP + FN
    totalPredicted = np.zeros(numOfClasses, dtype=int)
    totalGT = np.zeros(numOfClasses, dtype=int)
    # For all predicted notations
    for file in fileListPredicted:
        # Read the file
        predictedFilePath = os.path.join(predictedFolder, file)
        predictedObjects = ET.parse(predictedFilePath).findall('object')
        # For each item in the file
        for item in predictedObjects:
            classId = classes.index(item.find('name').text)
            confidence = float(item.find('confidence').text)
            xmin = int(item.find('bndbox').find('xmin').text)
            xmax = int(item.find('bndbox').find('xmax').text)
            ymin = int(item.find('bndbox').find('ymin').text)
            ymax = int(item.find('bndbox').find('ymax').text)
            dictPredicted[classId].append([confidence,
                                           file,
                                           [xmin, xmax, ymin, ymax]])
            totalPredicted[classId] += 1
    # For each predicted box, sort according to confidence
    for classId in range(numOfClasses):
        dictPredicted[classId].sort(key=lambda x: x[0], reverse=True)
    # Dictionary of dictionary, key: class, nested key : file
    # eg. { 'car' : {000001.xml: [[x,y,w,h],[a,b,c,d]], '0000002.xml': [] } }
    dictGT = {}
    dictMask = {}
    for classId in range(numOfClasses):
        dictGT[classId] = {}
        dictMask[classId] = {}
        for file in fileListGT:
            dictGT[classId][file] = []
            dictMask[classId][file] = []
    # For all ground truth notations
    for file in fileListGT:
        # Read the file
        GTFilePath = os.path.join(groundTruthFolder, file)
        groundTruthObjects = ET.parse(GTFilePath).findall('object')
        # For each item in the file
        for item in groundTruthObjects:
            classId = classes.index(item.find('name').text)
            xmin = int(item.find('bndbox').find('xmin').text)
            xmax = int(item.find('bndbox').find('xmax').text)
            ymin = int(item.find('bndbox').find('ymin').text)
            ymax = int(item.find('bndbox').find('ymax').text)
            # Append dictGT
            dictGT[classId][file].append([xmin, xmax, ymin, ymax])
            # To find out if a ground truth exists for an object of a class,
            # and if a prediction has been made corresponding to that object,
            # we shall append dictMask with 0 if GT exists,
            # and modify it to 1 when a prediction corresponds with it
            dictMask[classId][file].append(0)
            totalGT[classId] += 1
    # To record true positives and false positives
    truePositives = []
    falsePositives = []
    # FIND TRUE POSITIVES
    # For each class
    for classId in range(numOfClasses):
        # Init
        numberOfPredictedObjectsInClass = totalPredicted[classId]
        truePositives.append(np.zeros(numberOfPredictedObjectsInClass, dtype=int))
        falsePositives.append(np.zeros(numberOfPredictedObjectsInClass, dtype=int))
        # For each predicted object
        for predictedObjectIndex in range(numberOfPredictedObjectsInClass):
            predictedItem = dictPredicted[classId][predictedObjectIndex]
            # If no item of classId predicted is present in the ground truth
            # image
            if len(dictGT[classId][predictedItem[1]]) == 0:
                falsePositives[classId][predictedObjectIndex] = 1
                continue
            # Init
            maxIoU = 0.
            maxIndex = -1
            # Find the ground truth bounding box corresponding with the
            # predicted bounding box
            for GTObjectIndex in range(len(dictGT[classId][predictedItem[1]])):
                # If particular GTbox has already been alloted to a predicted
                # box, move to the next box without considering it
                if dictMask[classId][predictedItem[1]][GTObjectIndex] == 1:
                    continue
                # Otherwise
                areaMetric = IoU(dictGT[classId][predictedItem[1]][GTObjectIndex],
                                 predictedItem[2])
                # Record that GT bounding box which has maximum IoU with
                # the predicted bounding box
                if areaMetric > maxIoU:
                    maxIoU = areaMetric
                    maxIndex = GTObjectIndex
            # If all the GT box in a particular image are already alloted
            # to predicted boxes, add the new predictedBox to fP
            if maxIndex == -1:
                falsePositives[classId][predictedObjectIndex] = 1
                continue
            # Otherwise,
            # If the IoU exceeds a threshold,
            # add it to True Positives
            if maxIoU > IoUThreshold:
                dictMask[classId][predictedItem[1]][maxIndex] = 1
                truePositives[classId][predictedObjectIndex] = 1
            # Else,
            else:
                # FALSE POSITIVE (actually FALSE NEGATIVE)
                # For those classes with GT available but no prediction made,
                # we will consider this a false positive
                falsePositives[classId][predictedObjectIndex] = 1
    # Average precision per class
    cumulativePrecision = []
    cumulativeRecall = []
    averagePrecision = np.zeros(numOfClasses)
    # For each class, calculate Interpolated Average Precision
    # as given in PASCAL VOC handbook
    for classId in range(numOfClasses):
        # Cumulative precision : precision with increasing number of detections considered
        cumulativePrecision.append(np.divide(np.cumsum(truePositives[classId]),
            1 + np.arange(totalPredicted[classId])))
        # Cumulative Recall : recall with increasing number of detections considered
        cumulativeRecall.append(np.cumsum(truePositives[classId]) / totalGT[classId])
        # # Draw PC Curve
        # plt.plot(cumulativeRecall, cumulativePrecision); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.show()
        # Recall values
        recallValues = np.unique(cumulativeRecall[-1])
        if len(recallValues) > 1:
            recallStep = recallValues[1] - recallValues[0]
        else:
            recallStep = recallValues[0]
        # For each recall value
        for recallThreshold in recallValues:
            # Interpolated area under curve for recall value
            averagePrecision[classId] \
                += np.max(cumulativePrecision[-1][cumulativeRecall[-1] >= recallThreshold]) * recallStep
    # Mean Average Precision across classes
    meanAveragePrecision = np.mean(averagePrecision)
    # Print results
    print("\nMean Average Precision : %0.4f\n" % meanAveragePrecision)
    print("{0:>12}".format("Class-Name"),
          "{0:7}".format("TotalGT"),
          "{0:9}".format("TotalPred"),
          "{0:13}".format("TruePositives"),
          "{0:14}".format("FalsePositives"),
          "{0:12}".format("AvgPrecision"))
    for classId in range(numOfClasses):
        print("{0:>12}".format(classes[classId]),
              "{0:>7}".format(totalGT[classId]),
              "{0:>9}".format(len(dictPredicted[classId])),
              "{0:>13}".format(np.sum(truePositives[classId])),
              "{0:>14}".format(np.sum(falsePositives[classId])),
              "{0:8.4f}".format(averagePrecision[classId]))
    # Plot PC curve
    if plotPCCurve:
        for cl, classId in enumerate(classes):
            plt.plot(cumulativeRecall[cl], cumulativePrecision[cl], label=classId, c=np.random.rand(3, 1))
        plt.xlim([0, 1])
        plt.ylim([0.5, 1])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        leg = plt.legend(loc='right', fontsize=11)
        plt.show()
    # Return
    return meanAveragePrecision, averagePrecision


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
