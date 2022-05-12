

#importing libraries
!git clone https://github.com/ultralytics/yolov5  # cloning yolov5 repo
%cd yolov5 #to open yolov5 folder
%pip install -qr requirements.txt  # install
import torch #importing pytorch
from yolov5 import utils
display = utils.notebook_init()  # 


!unzip -q ../train_data.zip -d../ #to unzip the zipped file


#Train YOLOv5s on Custom dataset for 20 epochs
!python train.py --img 640 --batch 30 --epochs 20 --data coco128.yaml --weights yolov5s.pt --cache

#detecting the new image for number plate
!python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source /content/city.jpeg --save-crop
