from ultralytics import YOLO
import cv2
import math
import numpy as np
from models.siammask import SiamMask

import argparse
from copy import deepcopy
from typing import Any, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from perception.inferencer import SAMDecoderInferencer, SAMEncoderInferencer
from perception.inference import *
from torchvision.transforms.functional import resize


def click_event(event, x, y, flags, param):
    global selected_box, boxes_info,latest_img,trt_encoder,trt_decoder
    if event == cv2.EVENT_LBUTTONDOWN:
        min_area = float('inf')
        selected_box = None
        for info in boxes_info:
            box = info['box']
            x1, y1, x2, y2 = box
            if x1 < x < x2 and y1 < y < y2:
                area = (x2 - x1) * (y2 - y1)
                if area < min_area:
                    min_area = area
                    selected_box = box
        if selected_box is None:
            origin_image_size = latest_img.shape[:2]
            img = preprocess(cv2.cvtColor(latest_img,cv2.COLOR_BGR2RGB), img_size=1024)

            image_embedding = trt_encoder.infer(img)
            image_embedding = image_embedding[0].reshape(1, 256, 64, 64)

            input_size = get_preprocess_shape(*origin_image_size, long_side_length=1024)

            point = np.array([[[x, y, 1]]], dtype=np.float32)
            point_coords = point[..., :2]
            point_labels = point[..., 2]
            point_coords = apply_coords(point_coords, origin_image_size, input_size).astype(np.float32)

            inputs = (image_embedding, point_coords, point_labels)

            low_res_masks, _ = trt_decoder.infer(inputs)
            low_res_masks = low_res_masks.reshape(1, 1, 256, 256)

            masks = mask_postprocessing(low_res_masks, origin_image_size)
            masks = masks > 0.0
            
            bbox = calculate_bounding_box(masks[0].squeeze().numpy())
            selected_box = bbox

cv2.namedWindow("Webcam")
cv2.setMouseCallback("Webcam", click_event)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_model_type", type=str, required=True, help="v8l/v8s")
    parser.add_argument("--sam_model_type", type=str, required=True, help="l2/xl1")
    parser.add_argument("--enter_to_track", action="store_true", help="是否点完了按回车才track，默认是点了就track")
    parser.add_argument("--class_names", type=str, required=True, help="用逗号分隔的对象类名列表，例如 'person,car,dog'或'red box,green pencil,white box'")

    args = parser.parse_args()

    # 初始化YOLO模型
    detect_model = YOLO(f"/workspace/YOLOv8-TensorRT/yolo{args.yolo_model_type}-world.pt")
    track_model = SiamMask("/workspace/SiamMask/siammask_vot_simp.onnx")
    trt_encoder = SAMEncoderInferencer(f"/workspace/efficientvit/assets/export_models/sam/tensorrt/{args.sam_model_type}_encoder.engine", batch_size=1)
    trt_decoder = SAMDecoderInferencer(f"/workspace/efficientvit/assets/export_models/sam/tensorrt/{args.sam_model_type}_decoder.engine", num=1, batch_size=1)

    # Define custom classes
    classNames = args.class_names.split(',')    
    detect_model.set_classes(classNames)

    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    selected_box = None  # 存储选中框的坐标
    boxes_info = []  # 用于存储框的信息
    latest_img = None

    track_initialized = False

    while True:
        success, img = cap.read()
        if not success:
            break
        latest_img = img
        
        if track_initialized:
            mask = track_model.forward(img)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt_area = [cv2.contourArea(cnt) for cnt in contours]

            if len(contours) != 0 and np.max(cnt_area) > 100:
                contour = contours[np.argmax(cnt_area)]  # use max area polygon
                polygon = contour.reshape(-1, 2)
                img[:, :, 2] = (mask > 0) * 255 + (mask == 0) * img[:, :, 2]
                img = cv2.polylines(img, [polygon], True, (0,0,255), 3)
        else:
            results = detect_model.predict(img)
            boxes = results[0].boxes  # 获取检测结果
            boxes_info = []  # 清空上一帧的信息
            may_use_sam = True

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0]*100))/100
                cls = int(box.cls[0])
                # 存储每个框的信息
                boxes_info.append({'box': (x1, y1, x2, y2), 'conf': conf, 'cls': cls})
                
                # 设置颜色和文本
                if (x1, y1, x2, y2) == selected_box:
                    may_use_sam = False
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                label = f"{classNames[cls]} {conf}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # 打印日志
                print(f"Class name --> {classNames[cls]}, Confidence ---> {conf}")

            if may_use_sam and selected_box:
                color = (0, 0, 255)
                cv2.rectangle(img, (selected_box[0], selected_box[1]), (selected_box[2], selected_box[3]), color, 3)
                cv2.putText(img, "unknown", (selected_box[0], selected_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            if selected_box and not args.enter_to_track:
                # here do something with ROI points values (p1 and p2)
                print("SiamMask Model set initialization")
                x, y, w, h = selected_box[0], selected_box[1], selected_box[2] - selected_box[0], selected_box[3] - selected_box[1]
                print(x, y, w, h)
                track_model.init(img, (x,y,w,h))
                track_initialized = True


        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            selected_box = None  # 重置选中的框
            track_initialized = False
        elif key in [13, 32] and selected_box and args.enter_to_track: # Pressed Enter or Space to use ROI
            # here do something with ROI points values (p1 and p2)
            print("SiamMask Model set initialization")
            x, y, w, h = selected_box[0], selected_box[1], selected_box[2] - selected_box[0], selected_box[3] - selected_box[1]
            print(x, y, w, h)
            track_model.init(img, (x,y,w,h))
            track_initialized = True

        cv2.imshow('Webcam', img)

    cap.release()
    cv2.destroyAllWindows()
