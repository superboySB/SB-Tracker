from ultralytics import YOLO

import cv2
import math
import numpy as np
import argparse

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
    global selected_box, boxes_info,latest_img
    sam_encoder = param['sam_encoder']
    sam_decoder = param['sam_decoder']
    sam_model_type = param['sam_model_type']
    device_type = param['device_type']
    
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
                    
        if selected_box is None and device_type == "server":
            origin_image_size = latest_img.shape[:2]
            if sam_model_type == "xl1":
                img = preprocess(cv2.cvtColor(latest_img,cv2.COLOR_BGR2RGB), img_size=1024)
            elif sam_model_type == "l2":
                img = preprocess(cv2.cvtColor(latest_img,cv2.COLOR_BGR2RGB), img_size=512)
            else:
                raise NotImplementedError 

            image_embedding = sam_encoder.infer(img)
            image_embedding = image_embedding[0].reshape(1, 256, 64, 64)

            input_size = get_preprocess_shape(*origin_image_size, long_side_length=1024)

            point = np.array([[[x, y, 1]]], dtype=np.float32)
            point_coords = point[..., :2]
            point_labels = point[..., 2]
            point_coords = apply_coords(point_coords, origin_image_size, input_size).astype(np.float32)

            inputs = (image_embedding, point_coords, point_labels)

            low_res_masks, _ = sam_decoder.infer(inputs)
            low_res_masks = low_res_masks.reshape(1, 1, 256, 256)

            masks = mask_postprocessing(low_res_masks, origin_image_size)
            masks = masks > 0.0
            
            bbox = calculate_bounding_box(masks[0].squeeze().numpy())
            selected_box = bbox

        if selected_box is None and device_type == "deployment":
            origin_image_size = latest_img.shape[:2]
            if sam_model_type == "xl1":
                img = preprocess(cv2.cvtColor(latest_img,cv2.COLOR_BGR2RGB), img_size=1024)
            elif sam_model_type == "l2":
                img = preprocess(cv2.cvtColor(latest_img,cv2.COLOR_BGR2RGB), img_size=512)
            else:
                raise NotImplementedError
            img_embeddings = sam_encoder(img)

            point = np.array([[[x, y, 1]]], dtype=np.float32)
            point_coords = point[..., :2]
            point_labels = point[..., 2]

            masks, _, _ = sam_decoder.run(
                img_embeddings=img_embeddings,
                origin_image_size=origin_image_size,
                point_coords=point_coords,
                point_labels=point_labels,
            )
            bbox = calculate_bounding_box(masks[0].squeeze().numpy())
            selected_box = bbox


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_type", type=str, default="deployment", help="server/deployment")
    parser.add_argument("--yolo_model_type", type=str, default="v8l", help="v8s (small) / v8l (large)")
    parser.add_argument("--sam_model_type", type=str, default="xl1",help="l0 (small) / l2 (middle) / xl1 (large)")
    parser.add_argument("--enter_to_track", action="store_true", help="是否点完了按回车才track，默认是点了就track")
    parser.add_argument("--use_tensorrt", action="store_true", help="为了硬件的通用性，我们默认不使用tensorrt，使用ONNX Runtime;启动该标记视为使用tensorrt")
    parser.add_argument("--class_names", type=str, default="person", help="用逗号分隔的对象类名列表，例如 'person,car,dog'或'red box,green pencil,white box'")

    args = parser.parse_args()

    # 选择检测模型
    detect_model = YOLO(f"/workspace/YOLOv8-TensorRT/yolo{args.yolo_model_type}-world.pt")
    
    # 选择分割模型
    if args.use_tensorrt:
        from models.sam.tensorrt.inferencer import SAMDecoderInferencer, SAMEncoderInferencer
        from models.sam.tensorrt.inference import *
        sam_encoder = SAMEncoderInferencer(f"/workspace/efficientvit/assets/export_models/sam/tensorrt/{args.sam_model_type}_encoder.engine", batch_size=1)
        sam_decoder = SAMDecoderInferencer(f"/workspace/efficientvit/assets/export_models/sam/tensorrt/{args.sam_model_type}_decoder.engine", num=1, batch_size=1)
    else:
        from models.sam.onnx.inference import *
        sam_encoder = SamEncoder(model_path=f"/workspace/efficientvit/assets/export_models/sam/onnx/{args.sam_model_type}_encoder.onnx")
        sam_decoder = SamDecoder(model_path=f"/workspace/efficientvit/assets/export_models/sam/onnx/{args.sam_model_type}_decoder.onnx")

    # 选择跟踪模型
    if args.device_type == "server":
        from models.siammask import SiamMask
        tracker = SiamMask("/workspace/SiamMask/siammask_vot_simp.onnx")
    elif args.device_type == "deployment":
        from models.nanotrack.core.config import cfg
        from models.nanotrack.models.model_builder import ModelBuilder
        from models.nanotrack.tracker.nano_tracker import NanoTracker
        from models.nanotrack.utils.model_load import load_pretrain
        cfg.merge_from_file("/workspace/SiamTrackers/NanoTrack/models/config/configv3.yaml")
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        track_model = load_pretrain(ModelBuilder(cfg), "/workspace/SiamTrackers/NanoTrack/models/pretrained/nanotrackv3.pth").cuda().eval()
        tracker = NanoTracker(track_model, cfg)
    else:
        raise NotImplementedError

    # Define custom classes
    classNames = args.class_names.split(',')    
    detect_model.set_classes(classNames)

    # 初始化摄像头
    cv2.namedWindow("Webcam")
    # 创建包含所需变量的字典
    params = {
        'sam_encoder': sam_encoder,
        'sam_decoder': sam_decoder,
        'sam_model_type': args.sam_model_type,
        'device_type': args.device_type,
    }
    cv2.setMouseCallback("Webcam", click_event, params)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L)
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
            if args.device_type == "server":
                mask = tracker.forward(img)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cnt_area = [cv2.contourArea(cnt) for cnt in contours]

                if len(contours) != 0 and np.max(cnt_area) > 100:
                    contour = contours[np.argmax(cnt_area)]  # use max area polygon
                    polygon = contour.reshape(-1, 2)
                    img[:, :, 2] = (mask > 0) * 255 + (mask == 0) * img[:, :, 2]
                    img = cv2.polylines(img, [polygon], True, (0,0,255), 3)
            elif args.device_type == "deployment":
                outputs = tracker.track(img)
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(img, (bbox[0], bbox[1]),
                                (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                                (0, 0, 255), 3)
            else:
                raise NotImplementedError
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
