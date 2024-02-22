from ultralytics import YOLO
import cv2
import math
import numpy as np
from models.siammask import SiamMask

# 初始化YOLO模型
detect_model = YOLO("/workspace/YOLOv8-TensorRT/yolov8l-world.pt")
track_model = SiamMask("/workspace/SiamMask/siammask_vot_simp.onnx")

# object classes
classNames = ["red box","green pencil","white box"]

# Define custom classes
detect_model.set_classes(classNames)

# 初始化摄像头
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

selected_box = None  # 存储选中框的坐标
boxes_info = []  # 用于存储框的信息

def click_event(event, x, y, flags, param):
    global selected_box, boxes_info
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

cv2.namedWindow("Webcam")
cv2.setMouseCallback("Webcam", click_event)
track_initialized = False

while True:
    success, img = cap.read()
    if not success:
        break
    
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

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            # 存储每个框的信息
            boxes_info.append({'box': (x1, y1, x2, y2), 'conf': conf, 'cls': cls})
            
            # 设置颜色和文本
            color = (0, 0, 255) if (x1, y1, x2, y2) == selected_box else (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            label = f"{classNames[cls]} {conf}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # 打印日志
            print(f"Class name --> {classNames[cls]}, Confidence ---> {conf}")

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        selected_box = None  # 重置选中的框
        track_initialized = False
    elif key in [13, 32] and selected_box: # Pressed Enter or Space to use ROI
        # here do something with ROI points values (p1 and p2)
        print("SiamMask Model set initialization")
        x, y, w, h = selected_box[0], selected_box[1], selected_box[2] - selected_box[0], selected_box[3] - selected_box[1]
        print(x, y, w, h)
        track_model.init(img, (x,y,w,h))
        track_initialized = True

    cv2.imshow('Webcam', img)

cap.release()
cv2.destroyAllWindows()
