### You Only Look Once ( YOLO v1 )

Implementation of YOLO object detection pipeline using tensorflow. YOLO is a real time object detection method. It treats both object detection and localisation as regression problems. This is in contrast to previous object detection pipelines such as R-CNN, which had seperate entities for detection and localisation and were far more complicated to fine tune/train. More on YOLO [here](https://arxiv.org/pdf/1506.02640.pdf).

As of now, YOLO v2 is out. Check it out [here](https://arxiv.org/pdf/1612.08242.pdf). 

### How to use ?

Make sure that weight file is present in weights directory. Currently there are three modes, all pertaining to test the pre-trained model. 

- 'testDB' - Tests the code on a database. ( PASCAL VOC 2007, 2012, MS-COCO ). Keep in mind the model has been trained on PASCAL VOC 2007+2012. So any other dataset would require training. As this has not been implemented yet, [darkflow](https://github.com/thtrieu/darkflow) may help. 
- 'testLive' - Tests from a live Webcam feed.
- 'testFile' - Tests on a single image. 

By default, it runs on 'testLive' mode.

```
python yolo.py
```
### Results (PACAL VOC 2007)

Class Name | Ground Truth | Predicted | True Positive | False Positive | Avg. Precision
---------- | ------------ | --------- | ------------- | -------------- | --------------
aeroplane| 311| 213| 141| 72| 0.55494075262829001
bicycle| 389| 237| 157| 80| 0.56105639251746608
bird| 576| 359| 184| 175| 0.42433899865958929
boat| 393| 213| 77| 136| 0.24679748475368918
bottle| 657| 128| 33| 95| 0.17272727272727273
bus| 254| 168| 117| 51| 0.54621080695222668
car| 1541| 925| 436| 489| 0.34186953795331443
cat| 370| 322| 250| 72| 0.67658801636799115
chair| 1374| 420| 102| 318| 0.12245608573113981
cow| 329| 204| 66| 138| 0.17318304265255621
diningtable| 299| 160| 114| 46| 0.6494860956834515
dog| 530| 422| 299| 123| 0.63538308205967287
horse| 395| 279| 209| 70| 0.63316214093397671
motorbike| 369| 228| 140| 88| 0.48458388143892261
person| 5227| 3319| 1166| 2153| 0.23070649513423477
pottedplant| 592| 200| 53| 147| 0.17236723672367235
sheep| 311| 172| 44| 128| 0.16292819499341238
sofa| 396| 141| 105| 36| 0.6494177280693908
train| 302| 255| 191| 64| 0.66130285346624584
tvmonitor| 361| 209| 133| 76| 0.54199124564843304

### Requirements 

- Tenseflow 1.0 
- OpenCV 2
- Python 2
- Pre-trained [weights](https://drive.google.com/file/d/0B2JbaJSrWLpza08yS2FSUnV2dlE/view)

#### TODO List

- [x] Complete the mean-Average Precision 
- [ ] Document the code
- [x] Add PASCAL VOC 2007 results to the readme
- [ ] Complete network training function

### References 

- Author's [Website](https://pjreddie.com/darknet/yolo/)
- [This](https://github.com/hizhangp/yolo_tensorflow) implementation 
