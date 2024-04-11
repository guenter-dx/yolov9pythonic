import os, sys
import torch
import numpy as np
import cv2

# for plotting boxes 
import matplotlib.pyplot as plt


## MODEL WRAPPER
#  A more pythonic way to use YOLOv9

class Yolo:
    
    def __init__(self, 
                 name = "YoloPythonic",
                 model_yaml='models/detect/yolov9-c.yaml',
                 data_yaml='data/data.yaml',
                 model_path='weights/yolov9-c.pt',
                 batch_size=16,
                 epochs=100
                 ):
        self.name = name
        self.model_yaml = model_yaml
        self.data_yaml = data_yaml
        # either pre-trained weights or path to the model weights
        self.weights = model_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.cuda_available = torch.cuda.is_available()
        self.gpus = 'cpu'
        
        self.just_trained = False
        
        
    def train(self):
        if self.cuda_available:
            num_gpu = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
            self.gpus = ','.join([str(el) for el in range(num_gpu)])
            os.system(f"python train_dual.py --workers 8 --device {self.gpus} --batch {self.batch_size} --data {self.data_yaml} --img 640 --cfg {self.model_yaml} --weights {self.weights} --name yolov9-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs {self.epochs} --close-mosaic 15")
            
            # Train multiple gpus
            # needs adjustments (variables)
            # os.system("python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train_dual.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch 128 --data data/coco.yaml --img 640 --cfg models/detect/yolov9-c.yaml --weights '' --name yolov9-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs 500 --close-mosaic 15")
            
            self.just_trained = True
        
        else:
            os.system(f"python train_dual.py --workers 1 --device cpu --batch 4 --data {self.data_yaml} --img 640 --cfg {self.model_yaml} --weights {self.weights} --name yolov9-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs {self.epochs} --close-mosaic 15")
            
    
    def predict(self, input_path, **kwargs):
        if self.just_trained:
            # if just trained, go for the last run and get best weights
            directory = 'runs/train'
            dirs = os.listdir(directory)
            dirs.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
            self.weights = dirs[0] + '/weights/best.pt'
            
            # make sure we keep these weights stored in the object
            self.just_trained = False            
        
        # inference on cpu
        # os.system(f"python detect_dual.py --source {input_path} --img 640 --device cpu --weights {self.weights} --name yolov9_c_640_detect")
        
        from inference import detect
        
        nosave=kwargs.get('nosave', False)
        conf_thres=kwargs.get('conf_thres', 0.25)
        save_crop=kwargs.get('save_crop', False)
        name=kwargs.get('name', 'exp')
        classes=kwargs.get('classes', None)
        
        res = detect(
            weights=self.weights,
            source=input_path,
            device= self.gpus if self.cuda_available else 'cpu',
            nosave=nosave,
            conf_thres=conf_thres,
            save_crop=save_crop,
            name=name,
            classes=classes
        )
        
        return res
        



