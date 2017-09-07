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



def calculate_mAP():

    # Init
    fileListGT = os.listdir(groundTruthFolder)
    fileListPredicted = os.listdir(predictedFolder)
    
    dictPredicted={}
    
    # Each index of values in key:value pair would consist
    # of [confidence, image_name, [x,y,w,h]],key:class 
    for classId in range(numOfClasses):
        dictPredicted[classId]=[]

    # total numbers of objects predicted can help in calcu
    # - lating recall as it equals TP + FN 
    totalPredicted = np.zeros(numOfClasses, dtype=int)
    totalGT = np.zeros(numOfClasses, dtype=int)

    for file in fileListPredicted:
        
        predictedFilePath = os.path.join(predictedFolder, file)
        predictedObject = ET.parse(predictedFilePath).findall('object')
        
        for item in predictedObject:
            itemClass = item.find('name').text
            classId = classes.index(itemClass)
            confidence = float(item.find('confidence').text)
            xmin = int(item.find('bndbox').find('xmin').text)
            xmax = int(item.find('bndbox').find('xmax').text)
            ymin = int(item.find('bndbox').find('ymin').text)
            ymax = int(item.find('bndbox').find('ymax').text)
             

            dictPredicted[classId].append([confidence,
                file,
                [xmin, xmax, ymin, ymax]])

            totalPredicted[classId]+=1
    
    # for each predicted box, sort according to confidence   
    for classId in range(numOfClasses):
        dictPredicted[classId].sort(key=lambda x: x[0], reverse=True)

    # dictionary of dictionary, key: class, nested key : file
    # eg { 'car' : {000001.xml: [[x,y,w,h],[a,b,c,d]], '0000002.xml': [] } }
    dictGT = {}
    dictMask = {}
    
    for classId in range(numOfClasses):
        dictGT[classId] = {}
        dictMask[classId] = {}
        for file in fileListGT:
            dictGT[classId][file] = []
            dictMask[classId][file] = []

    for file in fileListGT:

        GTFilePath = os.path.join(groundTruthFolder, file)
        groundTruthObject = ET.parse(GTFilePath).findall('object')
 
        for item in groundTruthObject:
            itemClass = item.find('name').text
            classId = classes.index(itemClass)
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
            dictMask[classId][file].append(False)
            totalGT[classId]+=1
 
    truePositives = []
    falsePositives = []
    falseNegatives = np.zeros(numOfClasses)
   
    for classId in range(numOfClasses):
        numberOfPredictedObjectsInClass = totalPredicted[classId]
        truePositives.append(np.zeros(numberOfPredictedObjectsInClass,
            dtype=np.float64))
        falsePositives.append(np.zeros(numberOfPredictedObjectsInClass,
            dtype=np.float64))

        for predictedObjectIndex in range(len(dictPredicted[classId])):
        # To find the ground truth bounding box corresponding with the
        # predicted bounding box

            predictedItem = dictPredicted[classId][predictedObjectIndex]
            maxIoU = 0.0
            maxIndex = -1
            # If no item of classId predicted is present in the ground truth image
            if len(dictGT[classId][predictedItem[1]])==0:
                falsePositives[classId][predictedObjectIndex]=1
                continue
             # For each ground truth box            
            for GTObjectIndex in  range(len(dictGT[classId][predictedItem[1]])):
                # If particular GTbox has already been alloted to predicted box
                # move to the next box without considering it
                if dictMask[classId][predictedItem[1]][GTObjectIndex]==True:
                    continue
                GTItem = dictGT[classId][predictedItem[1]]
                areaMetric = IoU(GTItem[GTObjectIndex],
                                 predictedItem[2])

                # Record that GT bounding box which has maximum IoU with
                # the predicted bounding box
                if areaMetric > maxIoU:

                    maxIoU = areaMetric
                    maxIndex = GTObjectIndex
            
            # If all the GT box in a particular image are already alloted
            # to predictedBox, add the new predictedBox to fP
            if maxIndex==-1:
                falsePositives[classId][predictedObjectIndex]=1
                continue

            if maxIoU > IoUThreshold:

                # If the object has not been detected before
                if dictMask[classId][predictedItem[1]][maxIndex]==False:
                    
                    # Modify dictMask to indicate that object has been 
                    # detected
                    dictMask[classId][predictedItem[1]][maxIndex]=True
                    truePositives[classId][predictedObjectIndex]=1
                    
                    # predictObject has been attributed to GT Object
                    # move to next predictedObject
                else:   
                    
                    # Else if object has been detected before, since we know
                    # that the current prediction has lesser confidence than the
                    # previous one, so we will consider this a false positive
                    falsePositives[classId][predictedObjectIndex]=1
            else:
                falsePositives[classId][predictedObjectIndex]=1
    cumulativePrecision = []
    cumulativeRecall = []
    averagePrecision = np.zeros(numOfClasses)

    # For each class calculate Interpolated Average Precision
    # as given in PASCAL VOC handbook
    for classId in range(numOfClasses):

        for image in dictMask[classId]:
            falseNegatives[classId]+=sum([ not x for x in
                dictMask[classId][image]])
        cumulativePrecision.append(np.divide(np.cumsum(truePositives[classId]),
            np.cumsum(truePositives[classId])+np.cumsum(falsePositives[classId])))
        #cumulativePrecision.append(np.divide(np.cumsum(truePositives[classId]),
            #1 + np.arange(totalPredicted[classId])))

        cumulativeRecall.append(np.cumsum(truePositives[classId])/totalGT[classId])

        classPrecision = np.asarray(cumulativePrecision[-1], dtype=np.float64)
        classRecall = np.asarray(cumulativeRecall[-1], dtype=np.float64)
        for threshold in range(0,11,1):
            threshold = (threshold/10.0)

            # Get the maximum precision above a particular recall value 
            precisionValues = (classPrecision[classRecall>=threshold])

            # If precision is 0 for all the values 
            if precisionValues.shape[0]==0:
                p=0
            # Otherwise store the maximum
            else:
                p = np.amax(precisionValues)

            # Average precision would be mean of the precision values 
            # taken at these 11 points ( according to VOC handbook)
            averagePrecision[classId]+=(p/11)

    meanAveragePrecision = np.mean(averagePrecision)
    print ("Mean Average Precision : %.3f" % meanAveragePrecision)
    print ( "Class-Name",  
            "Total Ground Truth", 
            "Total Predicted", 
            "True Positives", 
            "False Positives",
            "False Negatives",
            "Average Precision") 
 
    for classId in range(numOfClasses):
        print ( classes[classId],
                totalGT[classId],
                len(dictPredicted[classId]),
                np.sum(truePositives[classId]),
                np.sum(falsePositives[classId]),
                falseNegatives[classId],
                averagePrecision[classId])

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





if __name__=='__main__':
    calculate_mAP()
