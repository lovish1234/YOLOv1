### You Only Look Once ( YOLO v1 )

Implementation of YOLO object detection pipeline using tensorflow. YOLO is a real time object detection method. It treats both object detection and localisation as regression problems. This is in contrast to previous object detection pipelines such as R-CNN, which had seperate entities for detection and localisation and were far more complicated to fine tune/train. More on YOLO [here](https://arxiv.org/pdf/1506.02640.pdf).

As of now, YOLO v2 is out. Check it out [here](https://arxiv.org/pdf/1612.08242.pdf). 

### How to use ?

Make sure that weight file is present in weights directory. Currently there are three modes, all pertaining to test the pre-trained model. 

- 'testDB' - Tests the code on a database. ( PASCAL VOC 2007, 2012, COCO ). Keep in mind the model has been trained on PASCAL VOC 2007+2012. So any other dataset would require training. As this has not been implemented yet, [darkflow](https://github.com/thtrieu/darkflow) may help. 
- 'testLive' - Tests from a live Webcam feed.
- 'testFile' - Tests on a single image. 

By default, it runs on 'testLive' mode.

```
python yolo.py
```

### Requirements 

- Tenseflow 1.0 
- OpenCV 2
- Python 2
- Pre-trained [weights](https://drive.google.com/file/d/0B2JbaJSrWLpza08yS2FSUnV2dlE/view)

#### TODO List

- [ ] Document the code
- [ ] Add PASCAL VOC 2012 results to the readme
- [ ] Complete network training function
- [ ] Complete the mean-Average Precision 

### References 

- Author's [Website](https://pjreddie.com/darknet/yolo/)
- [This](https://github.com/hizhangp/yolo_tensorflow) implementation 
