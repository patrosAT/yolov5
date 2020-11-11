import numpy as np
import cv2
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords
from utils.torch_utils import select_device, load_classifier


class YOLOv5Detector:

    def __init__(self, weights='weights/yolov5s.pt',
                       device=False,
                       imgsz=640,
                       conf_thres=0.25,
                       agnostic_nms = False,
                       iou_thres=0.45,
                       classify=False):

        # ---- PARAMETER ----
        self.weights = weights
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.agnostic_nms = agnostic_nms
        self.iou_thres = iou_thres
        self.classify = classify
        
        # ---- GPU ----
        if device:
            self.device = device
        else:
            self.device = select_device(str(torch.cuda.current_device()))
        
        # ---- LOAD MODEL ----
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size

        # ---- SECOND-STAGE CLASSIFIER ----
        if self.classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model'])  # load weights
            modelc.to(self.device).eval()

        # ---- NAMEAS & COLORS ----
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]    


    def detect(self, img):

        with torch.no_grad():

            # Load image
            im0 = img.copy()
            img = letterbox(img, new_shape=self.imgsz)[0]        # Padded resize

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = torch.from_numpy(np.ascontiguousarray(img)).to(self.device)
            img = img.float() / 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = self.model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=self.agnostic_nms)

            # Apply Classifier
            if self.classify:
                pred = apply_classifier(pred, modelc, img, im0)

            # Process detections
            for i, det in enumerate(pred):  # detections per image

                # Rescale boxes from img_size to im0 size
                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            det = det.cpu().numpy()

        return det
