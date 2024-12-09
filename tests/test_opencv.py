import pyrealsense2 as rs
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# 初始化 RealSense
pipeline = rs.pipeline()
config = rs.config()

# 配置低分辨率（424x240），并设置帧率为 90fps
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 启动流
pipeline.start(config)

# 加载目标检测模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval().to(device)

# 图像预处理
transform = T.Compose([
    T.ToTensor()
])

# COCO 数据集中 "杯子" 的类别索引
COCO_CUP_CLASS_ID = 47

def detect_objects(image):
    """检测图像中的物体"""
    img_tensor = transform(image).to(device)
    with torch.no_grad():
        predictions = model([img_tensor])
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()

    # 仅保留置信度高于 0.8 的 "杯子"
    mask = (scores > 0.8) & (labels == COCO_CUP_CLASS_ID)
    return boxes[mask]

def main():
    try:
        while True:
            # 获取图像帧
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # 转换为 numpy 数组
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

            # 检测杯子
            boxes = detect_objects(color_image)

            for box in boxes:
                # 获取检测框的中心点
                u = int((box[0] + box[2]) / 2)
                v = int((box[1] + box[3]) / 2)

                # 从深度图获取深度值
                depth = depth_frame.get_distance(u, v)
                if depth == 0 or depth > 5.0:  # 排除异常深度值
                    continue

                # 将像素坐标转换为相机坐标系
                point_camera = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u, v], depth)

                # 打印相机坐标
                print(f"Detected Cup at (Camera Frame): X={point_camera[0]:.2f}, "
                      f"Y={point_camera[1]:.2f}, Z={point_camera[2]:.2f}")

                # 在图像上标注物体的检测框和相对位置
                cv2.rectangle(color_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(
                    color_image,
                    f"X={point_camera[0]:.2f} Y={point_camera[1]:.2f} Z={point_camera[2]:.2f}",
                    (u - 50, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )

            # 显示结果
            cv2.imshow('RealSense', color_image)
            if cv2.waitKey(1) == 27:  # 按 ESC 退出
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
