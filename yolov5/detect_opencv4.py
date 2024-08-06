import torch
#import time
import os
import sys
from torchvision.models import resnet50
import torchvision.transforms as transforms
from PIL import Image
import cv2
import csv
import numpy as np
from pathlib import Path
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (non_max_suppression, 
                           scale_boxes, 
                           increment_path, 
                           check_img_size, 
                           check_imshow, 
                           check_file, 
                           Profile, 
                           print_args,
                           xyxy2xywh,
                           apply_classifier,)
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
import argparse


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


# Load the classifier model
def load_classifier_model():
    classifier_model = resnet50(pretrained=True)  # Load a pre-trained ResNet50 model
    classifier_model.eval()  # Set the model to evaluation mode
    return classifier_model

# Transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for classifier input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def custom_apply_classifier(pred, classifier_model, im, im0):
    """Apply a second-stage classifier to YOLO outputs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier_model.to(device)
    
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, det in enumerate(pred):  # per image
        if det is not None and len(det):
            det = det.clone()

            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0[i].shape).round()

            # Process each detection
            ims = []
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                crop_img = im0[i][y1:y2, x1:x2]
                crop_img = Image.fromarray(crop_img)
                crop_img = transform(crop_img).unsqueeze(0).to(device)

                ims.append(crop_img)

            ims = torch.cat(ims)
            
            
            # Classify ROIs
            with torch.no_grad():
                classifier_out = classifier_model(ims)
                class_idx = classifier_out.argmax(dim=1)
                class_conf = torch.nn.functional.softmax(classifier_out, dim=1)
            
            '''
            # Classify ROIs
            with torch.no_grad():
                classifier_out = classifier_model(ims)
                class_conf = torch.nn.functional.softmax(classifier_out, dim=1)
                class_idx = class_conf.argmax(dim=1)
                class_conf = class_conf.max(dim=1)[0]
            '''

            # Convert class_idx to a tensor on the same device as det
            class_idx = class_idx.to(det.device)
            class_conf = class_conf.max(dim=1)[0].to(det.device)
            
            # Update detections
            det[:, 5] = class_idx
            det[:, 4] = class_conf

    return pred




# Load the classifier model
classifier_model = load_classifier_model()

'''
# Function to apply the classifier
def apply_classifier(pred, classifier_model, im, im0s):
    for i, det in enumerate(pred):  # per image
        if len(det):
            for j, (*xyxy, conf, cls) in enumerate(det):
                # Extract the ROI using the bounding box coordinates
                x1, y1, x2, y2 = map(int, xyxy)
                crop_img = im0s[i][y1:y2, x1:x2]

                # Convert the ROI to PIL Image format and apply transformations
                crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                crop_img = transform(crop_img).unsqueeze(0)

                # Pass through the classifier model
                with torch.no_grad():
                    classifier_out = classifier_model(crop_img)
                    class_idx = classifier_out.argmax(dim=1).item()
                    class_conf = torch.nn.functional.softmax(classifier_out, dim=1)[0, class_idx].item()

                # Update the class and confidence in the original prediction
                det[j, 5] = class_idx
                det[j, 4] = class_conf

    return pred
'''

# Main inference function
@torch.no_grad()
def run(
    weights=ROOT / "../models/mask_yolov5.pt",  # model path or triton URL
    source="0",  # use laptop's front camera
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=True,  # show results
    save_txt=False,  # save results to *.txt
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    if isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)
    elif isinstance(imgsz, list) and len(imgsz) == 1:
        imgsz = (imgsz[0], imgsz[0])

    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    vid_writer = None

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)  # use webcam input

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    try:
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                if model.xml and im.shape[0] > 1:
                    ims = torch.chunk(im, im.shape[0], 0)

            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                if model.xml and im.shape[0] > 1:
                    pred = None
                    for image in ims:
                        if pred is None:
                            pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                        else:
                            pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                    pred = [pred, None]
                else:
                    pred = model(im, augment=augment, visualize=visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Second-stage classifier
            #pred = custom_apply_classifier(pred, classifier_model, im, im0s)

            # Define the path for the CSV file
            csv_path = save_dir / "predictions.csv"

            # Create or append to the CSV file
            def write_to_csv(image_name, prediction, confidence):
                """Writes prediction data for an image to a CSV file, appending if the file exists."""
                file_exists = csv_path.exists()
                data = {"Image Name": image_name, "Prediction": prediction, "Confidence": confidence}
                with open(csv_path, mode="a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=data.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(data)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                print(det)  # Add this to check if detections are present
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f"{i}: "
                else:
                    p, im0, frame = path, im0s.copy(), 0

                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in det:
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / im0.shape[1]).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(save_dir / "labels" / (p.stem + ".txt"), "a") as f:
                                f.write(("%g " * len(line)).rstrip() % line + "\n")
                        
                        c = int(cls)  # integer class
                        # Update this line to change the label displayed in the webcam feed
                        display_label = f"Face is covered {conf:.2f}" if names[c] == "with_mask" else f""
                        annotator.box_label(xyxy, display_label, color=colors(c, True))
                
                im0 = annotator.result()
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)

                if view_img:
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_dir / Path(p).name, im0)
                    else:  # video
                        save_path = save_dir / Path(p).name
                
                #time.sleep(0.5)


    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        # Release resources
        if save_img and vid_writer:
            for v in vid_writer:
                if v:
                    v.release()

        if hasattr(dataset, 'cap'):
            dataset.cap.release()
            cv2.destroyAllWindows()


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "../models/mask_yolov5.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default="0", help="file/dir/URL/glob, 0 for webcam")
    #parser.add_argument("--source", type=str, default=ROOT / "data/video.mp4", help="file/dir/URL/glob, 0 for webcam")  # comment out file input
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IOU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --class 0, or --class 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    return parser.parse_args()

                            
if __name__ == "__main__": 
    opt = parse_args()
    print_args(vars(opt))
    try:
        run(**vars(opt))
    except StopIteration:
        print("Iteration has stopped, possibly due to the end of the data source.")