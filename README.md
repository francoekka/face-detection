# Face Mask Detection
Face mask detection is an object detection task that detects whether people are wearing masks or not in videos. This repo includes a demo for building a face mask detector using YOLOv5 model. 


### Dataset
The model was trained on [Face-Mask](https://www.kaggle.com/andrewmvd/face-mask-detection) dataset which contains 853 images belonging to 3 classes, as well as their bounding boxes in the PASCAL VOC format. The classes are defined as follows:
* `With mask`
* `Without mask`
* `Mask worn incorrectly`

### Setup
cd yolov5
pip install -r requirements.txt
```

#### Training
* Download [Face-Mask](https://www.kaggle.com/andrewmvd/face-mask-detection) dataset from Kaggle and copy it into `datasets` folder. 
* Execute the following command to automatically unzip and convert the data into the YOLO format and split it into train and valid sets. The split ratio was set to 80/20%.
```
cd ..
python prepare.py
```
* Start training:
```
cd yolov5
python train.py --img 640 --batch 16 --epochs 100 --data ../mask_config.yaml --weights yolov5s.pt --workers 0
```
#### Inference
* If you train your own model, use the following command for inference:
```
python detect.py --source ../datasets/input.mp4 --weights runs/train/exp/weights/best.pt --conf 0.2
```
* Or you can use the pretrained model `models/mask_yolov5s.pt` for inference as follows:
```
python detect.py --source ../datasets/input.mp4 --weights ../models/mask_yolov5.pt --conf 0.2
```

### Reference

* [Darknet](https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py)
* [YOLOv5](https://github.com/ultralytics/yolov5)
