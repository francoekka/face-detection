### Dataset
The model was trained on (https://www.kaggle.com/andrewmvd/face-mask-detection) dataset, which contains 853 images belonging to 3 classes, as well as their bounding boxes in the PASCAL VOC format. The classes are defined as follows:
* `If face is detected but covered`
* `Face detected`
* `Face detected but not fully clear`

### Setup
Navigate to the `yolov5` directory and install the required packages:
```bash
cd yolov5
pip install -r requirements.txt
```

### Training
1. Download the [Face-Mask](https://www.kaggle.com/andrewmvd/face-mask-detection) dataset from Kaggle and copy it into the `datasets` folder.
2. Execute the following command to automatically unzip and convert the data into the YOLO format and split it into train and validation sets. The split ratio is set to 80/20:
```bash
cd ..
python prepare.py
```
3. Start training:
```bash
cd yolov5
python train.py --img 640 --batch 16 --epochs 100 --data ../mask_config.yaml --weights yolov5s.pt --workers 0
```

### Inference
* If you train your own model, use the following command for inference:
```bash
python detect.py --source ../datasets/input.mp4 --weights runs/train/exp/weights/best.pt --conf 0.2
```
* Or you can use the pretrained model `models/mask_yolov5s.pt` for inference as follows:
```bash
python detect.py --source ../datasets/input.mp4 --weights ../models/mask_yolov5.pt --conf 0.2
```
* For webcam or live stream:
```bash
python detect_opencv4.py --source ../datasets/input.mp4 --weights ../models/mask_yolov5.pt --conf 0.2
```

### Reference
* [Darknet](https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py)
* [YOLOv5](https://github.com/ultralytics/yolov5)

---

This version ensures that the commands are properly formatted and easy to follow.
