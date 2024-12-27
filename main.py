from ultralytics import YOLO
import cv2
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils import *
import torch
import tensorrt as trt
from torch2trt import TRTModule
from models.siammask import SiamMask

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

def show_bounding_box(ax, bbox, color='red', linewidth=2):
    """
    Display the bounding box on the image.
    """
    if bbox is None:
        return
    x_min, y_min, x_max, y_max = bbox
    ax.add_patch(plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, edgecolor=color, facecolor='none', linewidth=linewidth))

def click_event(event, x, y, flags, param):
    global selected_box, boxes_info, latest_img, track_initialized
    sam_encoder = param['sam_encoder']
    sam_decoder = param['sam_decoder']
    sam_model_type = param['sam_model_type']
    detect_model = param['detect_model']
    classNames = param['classNames']
    tracker = param['tracker']
    output_writer = param['output_writer']
    
    if event == cv2.EVENT_LBUTTONDOWN and not track_initialized:
        min_area = float('inf')
        selected_box_candidate = None
        for info in boxes_info:
            box = info['box']
            x1, y1, x2, y2 = box
            if x1 < x < x2 and y1 < y < y2:
                area = (x2 - x1) * (y2 - y1)
                if area < min_area:
                    min_area = area
                    selected_box_candidate = box
        
        if selected_box_candidate is not None:
            # 初始化跟踪器
            x, y, w, h = selected_box_candidate[0], selected_box_candidate[1], selected_box_candidate[2] - selected_box_candidate[0], selected_box_candidate[3] - selected_box_candidate[1]
            print("Tracker Model set initialization with selected box")
            print(x, y, w, h)
            tracker.init(latest_img, (x, y, w, h))
            selected_box = selected_box_candidate
            track_initialized = True
            print("Tracking initialized with selected bounding box.")
            return  # Exit after initializing
        
        else:
            # 使用SAM进行分割
            origin_image_size = latest_img.shape[:2]
            if sam_model_type == "xl1":
                img_preprocessed = preprocess(cv2.cvtColor(latest_img, cv2.COLOR_BGR2RGB), img_size=1024, device="cuda")
            elif sam_model_type == "l2":
                img_preprocessed = preprocess(cv2.cvtColor(latest_img, cv2.COLOR_BGR2RGB), img_size=512, device="cuda")
            else:
                raise NotImplementedError("Unsupported SAM model type.")
            
            image_embedding = sam_encoder(img_preprocessed)
            image_embedding = image_embedding[0].reshape(1, 256, 64, 64)

            input_size = get_preprocess_shape(*origin_image_size, long_side_length=1024)

            point = np.array([[[x, y, 1]]], dtype=np.float32)
            point_coords = point[..., :2]
            point_labels = point[..., 2]
            point_coords = apply_coords(point_coords, origin_image_size, input_size).astype(np.float32)

            inputs = (image_embedding, torch.from_numpy(point_coords).to("cuda"), torch.from_numpy(point_labels).to("cuda"))
            assert all([x.dtype == torch.float32 for x in inputs])

            low_res_masks, _ = sam_decoder(*inputs)
            low_res_masks = low_res_masks.reshape(1, 1, 256, 256)

            masks = mask_postprocessing(low_res_masks, origin_image_size)
            masks = masks > 0.0

            bbox = calculate_bounding_box(masks[0].squeeze().cpu().numpy())
            if bbox is not None:
                selected_box = bbox
                x, y, w, h = selected_box[0], selected_box[1], selected_box[2] - selected_box[0], selected_box[3] - selected_box[1]
                print("Tracker Model set initialization with SAM-generated box")
                print(x, y, w, h)
                tracker.init(latest_img, (x, y, w, h))
                track_initialized = True
                print("Tracking initialized with SAM-generated bounding box.")
            else:
                print("SAM could not generate a valid mask for the clicked point.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_model_type", type=str, default="v8l", help="v8s (small) / v8l (large)")
    parser.add_argument("--sam_model_type", type=str, default="xl1", help="l0 (small) / l2 (middle) / xl1 (large)")
    parser.add_argument("--class_names", type=str, default="person", help="用逗号分隔的对象类名列表，例如 'person,car,dog'或'red box,green pencil,white box'")
    args = parser.parse_args()

    # 默认使用TensorRT
    use_tensorrt = True

    # 选择检测模型
    detect_model = YOLO(f"/workspace/YOLOv8-TensorRT/yolo{args.yolo_model_type}-worldv2.pt")
    
    # 选择分割模型 (SAM) 使用TensorRT
    if use_tensorrt:
        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            with open(f"/workspace/efficientvit/assets/export_models/efficientvit_sam/tensorrt/efficientvit_sam_{args.sam_model_type}_encoder.engine", "rb") as f:
                engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)
        trt_encoder = TRTModule(engine, input_names=["input_image"], output_names=["image_embeddings"])

        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            with open(f"/workspace/efficientvit/assets/export_models/efficientvit_sam/tensorrt/efficientvit_sam_{args.sam_model_type}_decoder.engine", "rb") as f:
                engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)
        trt_decoder = TRTModule(
            engine,
            input_names=["image_embeddings", "point_coords", "point_labels"],
            output_names=["masks", "iou_predictions"],
        )
    else:
        raise NotImplementedError("Only TensorRT is supported in the current implementation.")

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
        latest_img = first_frame.copy()
        print("当前帧没有检测到指定类别的物体。")
    
    # 显示第一帧并等待用户交互
    cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)
    
    # 创建包含所需变量的字典
    params = {
        'sam_encoder': trt_encoder,
        'sam_decoder': trt_decoder,
        'sam_model_type': args.sam_model_type,
        'detect_model': detect_model,
        'classNames': classNames,
        'tracker': tracker,
        'output_writer': output_writer,
    }
    cv2.setMouseCallback("Video", click_event, params)
    
    selected_box = None  # 存储选中框的坐标
    track_initialized = False

    print("显示第一帧，请点击选择要跟踪的物体。")

    while True:
        cv2.imshow('Video', latest_img)
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
            if len(boxes_info) > 0:
                for info in boxes_info:
                    x1, y1, x2, y2 = info['box']
                    conf = info['conf']
                    cls = info['cls']
                    color = (0, 255, 0)
                    cv2.rectangle(latest_img, (x1, y1), (x2, y2), color, 3)
                    label = f"{classNames[cls]} {conf}"
                    cv2.putText(latest_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow('Video', latest_img)
            print("已重置，请再次点击选择要跟踪的物体。")

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
        cv2.imshow('Video', frame)
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
