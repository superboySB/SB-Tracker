from ultralytics import YOLO
import cv2
import math
import numpy as np
import argparse
from models.siammask import SiamMask
import torch

# Global variables for drawing
drawing = False  # True if mouse is pressed
ix, iy = -1, -1
selected_box_manual = None  # Manually selected box

def calculate_bounding_box(mask):
    """
    Calculate the bounding box coordinates from the mask.
    """
    pos = np.where(mask)
    if len(pos[0]) == 0 or len(pos[1]) == 0:
        return None
    x_min = np.min(pos[1])
    x_max = np.max(pos[1])
    y_min = np.min(pos[0])
    y_max = np.max(pos[0])
    return x_min, y_min, x_max, y_max

def click_event(event, x, y, flags, param):
    global selected_box, boxes_info, latest_img, track_initialized, drawing, ix, iy, selected_box_manual, frame_display
    detect_model = param['detect_model']
    classNames = param['classNames']
    tracker = param['tracker']
    output_writer = param['output_writer']
    scale_factor = param['scale_factor']

    if event == cv2.EVENT_LBUTTONDOWN and not track_initialized:
        if len(boxes_info) > 0:
            # Check if click is inside any detection box
            min_area = float('inf')
            selected_box_candidate = None
            for info in boxes_info:
                box = info['box']
                x1, y1, x2, y2 = box
                # Scale the box coordinates to the display size
                x1_disp, y1_disp, x2_disp, y2_disp = [int(coord * scale_factor) for coord in box]
                if x1_disp < x < x2_disp and y1_disp < y < y2_disp:
                    area = (x2_disp - x1_disp) * (y2_disp - y1_disp)
                    if area < min_area:
                        min_area = area
                        selected_box_candidate = box
            if selected_box_candidate is not None:
                # Initialize tracker with selected detection box
                x, y, w, h = selected_box_candidate[0], selected_box_candidate[1], selected_box_candidate[2] - selected_box_candidate[0], selected_box_candidate[3] - selected_box_candidate[1]
                print("Tracker Model set initialization with selected box")
                print(x, y, w, h)
                tracker.init(latest_img, (x, y, w, h))
                selected_box = selected_box_candidate
                track_initialized = True
                print("Tracking initialized with selected bounding box.")
                return  # Exit after initializing

        # If click is outside any detection box, start drawing manually
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Update the rectangle being drawn
            frame_display[:] = latest_img_resized.copy()
            cv2.rectangle(frame_display, (ix, iy), (x, y), (255, 0, 0), 2)
            cv2.imshow('Video', frame_display)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            selected_box_manual = (min(ix, x) / scale_factor, min(iy, y) / scale_factor, max(ix, x) / scale_factor, max(iy, y) / scale_factor)
            print(f"Manual selection box: {selected_box_manual}")
            # Initialize tracker with manually drawn box
            x1, y1, x2, y2 = selected_box_manual
            w = x2 - x1
            h = y2 - y1
            tracker.init(latest_img, (x1, y1, w, h))
            selected_box = selected_box_manual
            track_initialized = True
            print("Tracking initialized with manually selected bounding box.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_model_type", type=str, default="v8l", help="v8s (small) / v8l (large)")
    parser.add_argument("--class_names", type=str, default="person", help="用逗号分隔的对象类名列表，例如 'person,car,dog'或'red box,green pencil,white box'")
    args = parser.parse_args()

    # 选择检测模型
    detect_model = YOLO(f"/workspace/YOLOv8-TensorRT/yolo{args.yolo_model_type}-worldv2.pt")
    
    # 选择跟踪模型
    tracker = SiamMask("/workspace/SiamMask/siammask_vot_simp.onnx")

    # 定义自定义类别
    classNames = args.class_names.split(',')    
    detect_model.set_classes(classNames)

    # 打开视频文件
    video_path = "/workspace/SB-Tracker/data/ballon1.mp4"
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        exit(1)
    
    # 获取视频的帧率和尺寸
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 设置输出视频
    output_path = video_path.rsplit('.', 1)[0] + "_results.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 读取第一帧
    ret, first_frame = video_cap.read()
    if not ret:
        print("无法读取视频的第一帧。")
        exit(1)
    
    latest_img = first_frame.copy()
    
    # 进行YOLO检测
    results = detect_model.predict(latest_img)
    boxes = results[0].boxes  # 获取检测结果
    boxes_info = []  # 用于存储框的信息

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = math.ceil((box.conf[0]*100))/100
        cls = int(box.cls[0])
        # 存储每个框的信息
        boxes_info.append({'box': (x1, y1, x2, y2), 'conf': conf, 'cls': cls})
        
        # 设置颜色和文本
        color = (0, 255, 0)
        cv2.rectangle(latest_img, (x1, y1), (x2, y2), color, 3)
        label = f"{classNames[cls]} {conf}"
        cv2.putText(latest_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 打印日志
        print(f"Class name --> {classNames[cls]}, Confidence ---> {conf}")

    # 如果没有检测到指定类别的物体，则不显示任何框
    if len(boxes_info) == 0:
        print("当前帧没有检测到指定类别的物体。您可以手动拖拽选择一个框来初始化跟踪。")
    else:
        print("显示第一帧，请点击选择要跟踪的物体。")
    
    # Resize factor
    scale_factor = 0.5  # Resize to half
    width_resized = int(width * scale_factor)
    height_resized = int(height * scale_factor)
    
    # Resize the image for display
    latest_img_resized = cv2.resize(latest_img, (width_resized, height_resized))
    frame_display = latest_img_resized.copy()

    # 显示第一帧并等待用户交互
    cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)
    
    # 创建包含所需变量的字典
    params = {
        'detect_model': detect_model,
        'classNames': classNames,
        'tracker': tracker,
        'output_writer': output_writer,
        'scale_factor': scale_factor,
    }
    cv2.setMouseCallback("Video", click_event, params)
    
    selected_box = None  # 存储选中框的坐标
    track_initialized = False

    print("显示第一帧，请点击选择要跟踪的物体。")
    
    while True:
        cv2.imshow('Video', frame_display)
        key = cv2.waitKey(1) & 0xFF
        if track_initialized:
            break
        elif key == ord('q'):
            print("用户选择退出。")
            video_cap.release()
            output_writer.release()
            cv2.destroyAllWindows()
            exit(0)
        elif key == ord('r'):
            selected_box = None  # 重置选中的框
            track_initialized = False
            # 重新显示第一帧
            latest_img = first_frame.copy()
            # 重新进行YOLO检测
            results = detect_model.predict(latest_img)
            boxes = results[0].boxes  # 获取检测结果
            boxes_info = []  # 用于存储框的信息

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0]*100))/100
                cls = int(box.cls[0])
                # 存储每个框的信息
                boxes_info.append({'box': (x1, y1, x2, y2), 'conf': conf, 'cls': cls})
                
                # 设置颜色和文本
                color = (0, 255, 0)
                cv2.rectangle(latest_img, (x1, y1), (x2, y2), color, 3)
                label = f"{classNames[cls]} {conf}"
                cv2.putText(latest_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # 打印日志
                print(f"Class name --> {classNames[cls]}, Confidence ---> {conf}")

            # 如果没有检测到指定类别的物体，则不显示任何框
            if len(boxes_info) == 0:
                print("当前帧没有检测到指定类别的物体。您可以手动拖拽选择一个框来初始化跟踪。")
            else:
                print("重新显示第一帧，请点击选择要跟踪的物体。")
            
            # Resize for display
            latest_img_resized = cv2.resize(latest_img, (width_resized, height_resized))
            frame_display = latest_img_resized.copy()
            cv2.imshow('Video', frame_display)
            if len(boxes_info) > 0:
                print("已重置，请再次点击选择要跟踪的物体。")
            else:
                print("已重置，请手动拖拽选择要跟踪的物体。")
    
    # 初始化输出视频写入器
    output_writer.write(first_frame)
    
    # 开始跟踪
    frame_idx = 1  # 已处理第一帧
    print("开始跟踪视频...")
    
    while True:
        ret, frame = video_cap.read()
        if not ret:
            print("视频处理完毕。")
            break
        
        frame_idx += 1
        mask = tracker.forward(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]

        if len(contours) != 0 and np.max(cnt_area) > 100:
            contour = contours[np.argmax(cnt_area)]  # 使用最大面积的轮廓
            polygon = contour.reshape(-1, 2)
            frame[:, :, 2] = (mask > 0) * 255 + (mask == 0) * frame[:, :, 2]
            frame = cv2.polylines(frame, [polygon], True, (0,0,255), 3)
        else:
            print(f"跟踪失败，未能在第{frame_idx}帧中找到目标。")
            cv2.putText(
                frame,
                f"Tracking failed at frame {frame_idx}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        
        # 显示和保存结果帧
        # Resize for display
        frame_resized = cv2.resize(frame, (width_resized, height_resized))
        cv2.imshow('Video', frame_resized)
        output_writer.write(frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户选择退出。")
            break

    # 释放资源
    video_cap.release()
    output_writer.release()
    cv2.destroyAllWindows()
    print(f"跟踪结果已保存到: {output_path}")
