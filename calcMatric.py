import xml.etree.cElementTree as ET
from operator import itemgetter 
import os
import numpy as np

predictedFolder = 'VOC2007/test/outputAnnotations/'  
groundTruthFolder = '../VOC2007/test/Annotations/'

numOfClasses = 20

 
classes = ["aeroplane","bicycle","bird","boat","bottle","bus","car" \
           ,"cat","chair","cow","diningtable","dog","horse","motorbike" \
           , "person","pottedplant","sheep","sofa","train","tvmonitor"]

IoUThreshold = 0.5

def calculate_mAP():

    fileList = os.listdir(groundTruthFolder)
    truePositives = np.zeros(numOfClasses)
    falsePositives = np.zeros(numOfClasses)
    falseNegatives = np.zeros(numOfClasses)
    
    iteration=0
    for file in fileList:
        
        predictedFilePath = predictedFolder+file
        groundTruthFilePath = groundTruthFolder+file

        xmlParsePredicted = ET.parse(predictedFilePath)
        xmlParseGT = ET.parse(groundTruthFilePath)
        
        predictedObject = xmlParsePredicted.findall('object')
        groundTruthObject = xmlParseGT.findall('object')
        
        dictPredicted = {}
        dictGT = {}
        dictMask = {}
        for i in range(numOfClasses):
            dictPredicted[i]=[]
            dictGT[i] = []
            dictMask[i]=[]
        # extract the predicted boxes from xml file
        for item in predictedObject:
            category = item.find('name').text
            categoryIndex = classes.index(category)
            confidence = float(item.find('confidence').text)                     
            xmin = int(item.find('bndbox').find('xmin').text)
            xmax = int(item.find('bndbox').find('xmax').text)
            ymin = int(item.find('bndbox').find('ymin').text)
            ymax = int(item.find('bndbox').find('ymax').text)
            
            dictPredicted[categoryIndex].append([confidence,xmin,xmax,ymin,ymax])           
        
        # if no object is detected, count it as False Negative
        #if not bool(dictPredicted):
        #    falseNegatives+=1

        # sort each of the entries by decreasing order of confidence values
        for i in range(numOfClasses):
            dictPredicted[i]=sorted(dictPredicted[i],key=itemgetter(0),reverse=True)

        # extract the ground truth boxes from xml file
        for item in groundTruthObject:
            category = item.find('name').text
            categoryIndex = classes.index(category)
  
            xmin = int(item.find('bndbox').find('xmin').text)
            xmax = int(item.find('bndbox').find('xmax').text)
            ymin = int(item.find('bndbox').find('ymin').text)
            ymax = int(item.find('bndbox').find('ymax').text)
            
            dictGT[categoryIndex].append([xmin,xmax,ymin,ymax])

            # 0/1 denotes weather the ground truth has been alloted to a prediction
            dictMask[categoryIndex].append(0)
        
        # for a particular class, say chair
        for i in xrange(numOfClasses):
            # if atleast one of the box is predicted to be positive
            while dictPredicted[i]!=[]:
                maxIoU=0
                maxIndex=-1
                for j in range(len(dictGT[i])):
                    areaMetric = IoU(dictPredicted[i][0][1:],dictGT[i][j])
                    #print areaMetric,dictPredicted[i][0][1:],dictGT[i][j]
                    if areaMetric > maxIoU:
                        #print areaMetric
                        maxIoU = areaMetric
                        maxIndex = j
                if maxIoU > IoUThreshold:
                    # if the object has not been detected before
                    if dictMask[i][maxIndex]==0:
                        dictMask[i][maxIndex]=1
                        truePositives[i]+=1
                    else:
                        falsePositives[i]+=1
                else:
                    falsePositives[i]+=1
                dictPredicted[i].pop(0)
        
        for i in xrange(numOfClasses):
            falseNegatives[i]+=sum([1-j for j in dictMask[i]])
        
        #print 'fileName'+str(file)
        #print 'truePositives'+str(truePositives)
        #print 'falsePositives'+str(falsePositives)
        #print 'falseNegatives'+str(falseNegatives)

    print truePositives
    print falsePositives
    print falseNegatives 

    #print truePositives/(truePositives+falsePositives)
    #print truePositives/(truePositives+falseNegatives)
    
    
    fP = np.sum(falsePositives)
    tP = np.sum(truePositives)
    fN = np.sum(falseNegatives)
    precision = tP/(tP+fP)
    recall = tP/(tP+fN)
    print precision
    print recall

    '''
    avgPrecision =0
    for th in xrange(0.0,1.0,0.1):
        p = np.maximum(precision(recall>=th))
        avgPrecision+=(p/11)
    print avgPrecision
    '''

def IoU(boxA,boxB):

    intersectionX = max(0,min(boxA[1],boxB[1])-max(boxA[0],boxB[0]))
    intersectionY = max(0,min(boxA[3],boxB[3])-max(boxA[2],boxB[2]))
    
    intersection = intersectionX*intersectionY
    union = (boxA[1]-boxA[0])*(boxA[3]-boxA[2])+(boxB[1]-boxB[0])*(boxB[3]-boxB[2])-intersection
    #print intersection,union,intersection*1.0/union
    return intersection*1.0/union
    

if __name__=='__main__':
    calculate_mAP()
