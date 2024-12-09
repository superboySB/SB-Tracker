import cv2
import pyrealsense2 as rs
import numpy as np

def main():
    # 配置 RealSense 管道以启用 RGB 流
    pipeline = rs.pipeline()
    config = rs.config()

    # 配置 RGB 流（分辨率、帧率）
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 开始流
    pipeline.start(config)

    try:
        while True:
            # 从管道中获取帧数据
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # 转换为 NumPy 数组
            color_image = np.asanyarray(color_frame.get_data())

            # 显示 RGB 图像
            cv2.imshow("RealSense RGB Feed", color_image)

            # 按 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 停止流并释放资源
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
