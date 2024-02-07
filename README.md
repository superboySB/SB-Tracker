# SB-Tracker
开放环境下基于人机交互的UAV Tracker

## 开箱即用
```sh
docker build -f docker/deploy.dockerfile -t sbt_image:r35.3.1 .

# 如果连接了摄像头硬件就可以加，确保摄像头安装:  ls /dev/video*
docker run -itd --privileged -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=$DISPLAY --runtime=nvidia --device /dev/video0:/dev/video0 --device /dev/snd --device /dev/bus/usb --network=host --ipc host --name=sbtracker sbt_image:r35.3.1 /bin/bash

docker exec -it sbtracker /bin/bash
```
### 开放环境检测功能部署
这一步一定要在台式机上(可以考虑使用`train.dockerfile`在run完的容器直接拿)，我们需要在docker中默认准备好的一个训练好的自带pytorch模型作为范例,然后添加bbox decoder、NMS后转为ONNX模型。
```sh
cd /workspace/YOLOv8-TensorRT
python3 export-det.py --weights yolov8s.pt --sim && \
python3 export-det.py --weights yolov8s-oiv7.pt --sim && \
python3 export-seg.py --weights yolov8s-seg.pt --sim && \
yolo export model=yolov8s-pose.pt format=onnx simplify=True
```
下一步一定要在jetson本机上，使用TensorRT的python api来适配几个yolo模型
```sh
cd /workspace/YOLOv8-TensorRT
/usr/src/tensorrt/bin/trtexec --onnx=yolov8s.onnx --saveEngine=yolov8s.engine --fp16 && \
/usr/src/tensorrt/bin/trtexec --onnx=yolov8s-oiv7.onnx --saveEngine=yolov8s-oiv7.engine --fp16 && \
/usr/src/tensorrt/bin/trtexec --onnx=yolov8s-seg.onnx --saveEngine=yolov8s-seg.engine --fp16 && \
/usr/src/tensorrt/bin/trtexec --onnx=yolov8s-pose.onnx --saveEngine=yolov8s-pose.engine --fp16
```
测试yolo检测图片功能是否正常
```sh
python3 infer-det-camera.py \
    --engine yolov8s.engine \
    --imgs data \
    --out-dir outputs \
    --device cuda:0
```

### 开放环境分割功能部署
先去适配一下nanosam的推理加速
```sh
cd /opt/nanosam
```
将mask decoder部分适配TensorRT
```sh
/usr/src/tensorrt/bin/trtexec \
    --onnx=data/mobile_sam_mask_decoder.onnx \
    --saveEngine=data/mobile_sam_mask_decoder.engine \
    --minShapes=point_coords:1x1x2,point_labels:1x1 \
    --optShapes=point_coords:1x1x2,point_labels:1x1 \
    --maxShapes=point_coords:1x10x2,point_labels:1x10
```
将resnet18 encoder部分适配TensorRT
```sh
/usr/src/tensorrt/bin/trtexec \
        --onnx=data/resnet18_image_encoder.onnx \
        --saveEngine=data/resnet18_image_encoder.engine \
        --fp16
```
检验基本功能
```sh
python3 examples/basic_usage.py \
    --image_encoder="data/resnet18_image_encoder.engine" \
    --mask_decoder="data/mobile_sam_mask_decoder.engine"
```

## 运行代码
检验点击跟踪功能
```sh
cd /workspace && git clone https://github.com/superboySB/SB-Tracker && cd SB-Tracker

python3 det-camera.py

python3 click_and_track.py
```