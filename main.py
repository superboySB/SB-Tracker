from ultralytics import YOLO
import pyrealsense2 as rs

import cv2
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils import *

def calculate_bounding_box(mask):
    """
    Calculate the bounding box coordinates from the mask.
    """
    pos = np.where(mask)
    x_min = np.min(pos[1])
    x_max = np.max(pos[1])
    y_min = np.min(pos[0])
    y_max = np.max(pos[0])
    return x_min, y_min, x_max, y_max

def show_bounding_box(ax, bbox, color='red', linewidth=2):
    """
    Display the bounding box on the image.
    """
    x_min, y_min, x_max, y_max = bbox
    ax.add_patch(plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, edgecolor=color, facecolor='none', linewidth=linewidth))

def click_event(event, x, y, flags, param):
    global selected_box, boxes_info, latest_img
    sam_encoder = param['sam_encoder']
    sam_decoder = param['sam_decoder']
    sam_model_type = param['sam_model_type']
    
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
            if sam_model_type == "xl1":
                img = preprocess(cv2.cvtColor(latest_img,cv2.COLOR_BGR2RGB), img_size=1024, device="cuda")
            elif sam_model_type == "l2":
                img = preprocess(cv2.cvtColor(latest_img,cv2.COLOR_BGR2RGB), img_size=512, device="cuda")
            else:
                raise NotImplementedError 

            image_embedding = sam_encoder(img)
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
            selected_box = bbox


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_model_type", type=str, default="v8l", help="v8s (small) / v8l (large)")
    parser.add_argument("--sam_model_type", type=str, default="xl1",help="l0 (small) / l2 (middle) / xl1 (large)")
    parser.add_argument("--use_tensorrt", action="store_true", help="为了硬件的通用性，我们默认不使用tensorrt，使用ONNX Runtime;启动该标记视为使用tensorrt")
    parser.add_argument("--class_names", type=str, default="person", help="用逗号分隔的对象类名列表，例如 'person,car,dog'或'red box,green pencil,white box'")

    args = parser.parse_args()

    # 选择检测模型
    detect_model = YOLO(f"/workspace/YOLOv8-TensorRT/yolo{args.yolo_model_type}-worldv2.pt")
    
    # 选择分割模型
    if args.use_tensorrt:
        import tensorrt as trt
        from torch2trt import TRTModule
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
        raise NotImplementedError
        # TODO: 适用更广的技术栈
        # from models.sam.onnx.inference import *
        # sam_encoder = SamEncoder(model_path=f"/workspace/efficientvit/assets/export_models/sam/onnx/{args.sam_model_type}_encoder.onnx")
        # sam_decoder = SamDecoder(model_path=f"/workspace/efficientvit/assets/export_models/sam/onnx/{args.sam_model_type}_decoder.onnx")

    # 选择跟踪模型
    from models.siammask import SiamMask
    tracker = SiamMask("/workspace/SiamMask/siammask_vot_simp.onnx")

    # Define custom classes
    classNames = args.class_names.split(',')    
    detect_model.set_classes(classNames)

    # 初始化 RealSense
    pipeline = rs.pipeline()
    config = rs.config()

    # 配置分辨率为 640x480，帧率为 30fps
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # 启动流并增加等待时间
    try:
        pipeline.start(config)
        print("RealSense camera started.")

        # 获取深度传感器并关闭 Active IR
        depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.emitter_enabled, 0)  # 关闭 IR 投影
        print("Active IR (Emitter) has been disabled. Using Stereo Depth mode.")
    except RuntimeError as e:
        print(f"Error starting RealSense camera: {e}")
        exit(1)
    
    # 初始化摄像头
    cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)

    # 创建包含所需变量的字典
    params = {
        'sam_encoder': trt_encoder,
        'sam_decoder': trt_decoder,
        'sam_model_type': args.sam_model_type,
    }
    cv2.setMouseCallback("RealSense", click_event, params)

    selected_box = None  # 存储选中框的坐标
    boxes_info = []  # 用于存储框的信息
    latest_img = None

    track_initialized = False

    while True:
        # 获取图像帧
        frames = pipeline.wait_for_frames()

        if not frames:
            print("No frames received.")
            continue
        
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue
        
        # 转换为 numpy 数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        img = color_image
        latest_img = img
        
        if track_initialized:
            mask = tracker.forward(img)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt_area = [cv2.contourArea(cnt) for cnt in contours]

            if len(contours) != 0 and np.max(cnt_area) > 100:
                contour = contours[np.argmax(cnt_area)]  # use max area polygon
                polygon = contour.reshape(-1, 2)
                img[:, :, 2] = (mask > 0) * 255 + (mask == 0) * img[:, :, 2]
                img = cv2.polylines(img, [polygon], True, (0,0,255), 3)

                # ------ 新增代码开始：利用mask计算物体的3D坐标 ------
                mask_points = np.where(mask > 0)
                depths = []
                for i in range(len(mask_points[0])):
                    py = mask_points[0][i]  # y坐标
                    px = mask_points[1][i]  # x坐标
                    d = depth_frame.get_distance(px, py)
                    if d > 0 and d < 5.0:  # 筛选合理的深度值
                        depths.append(d)

                if len(depths) > 0:
                    # 使用中值深度，提高对异常值的鲁棒性
                    med_depth = np.median(depths)

                    # 计算mask质心
                    M = cv2.moments(mask.astype(np.uint8))
                    if M['m00'] != 0:
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        
                        # 将像素坐标与深度转换为相机坐标系下的3D点
                        point_camera = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cx, cy], float(med_depth))
                        print(f"Tracked Object at (Camera Frame): X={point_camera[0]:.2f}, Y={point_camera[1]:.2f}, Z={point_camera[2]:.2f}")

                        # 在图像上标注坐标
                        cv2.putText(
                            img,
                            f"X={point_camera[0]:.2f}, Y={point_camera[1]:.2f}, Z={point_camera[2]:.2f}",
                            (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            2
                        )
                # ------ 新增代码结束 ------
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
            
            if selected_box:
                # here do something with ROI points values (p1 and p2)
                x, y, w, h = selected_box[0], selected_box[1], selected_box[2] - selected_box[0], selected_box[3] - selected_box[1]
                print("Tracker Model set initialization")
                print(x, y, w, h)
                track_initialized = True
                tracker.init(img, (x, y, w, h))

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            selected_box = None  # 重置选中的框
            track_initialized = False

        cv2.imshow('RealSense', img)

    pipeline.stop()
    cv2.destroyAllWindows()
