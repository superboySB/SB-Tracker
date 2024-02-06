# SB-Tracker
开放环境下基于人机交互的UAV Tracker

## 开箱即用
```sh
docker build -f docker/deploy.dockerfile -t sbt_image:r35.3.1 .

# 如果连接了摄像头硬件就可以加，确保摄像头安装:  ls /dev/video*
docker run -itd --privileged -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=$DISPLAY --runtime=nvidia --device /dev/video0:/dev/video0 --device /dev/snd --device /dev/bus/usb --network=host --ipc host --name=sbtracker sbt_image:r35.3.1 /bin/bash

docker exec -it sbtracker /bin/bash

cd /workspace && git clone https://github.com/superboySB/SB-Tracker && cd SB-Tracker
```
### 开放环境检测功能部署
在docker中默认准备好了一个训练好的自带pytorch模型,然后添加bbox decoder、NMS后转为ONNX模型，然后使用TensorRT的python api来适配几个yolo模型
```sh
cd /workspace/YOLOv8-TensorRT
/usr/src/tensorrt/bin/trtexec --onnx=yolov8m.onnx --saveEngine=yolov8m.engine --fp16 && \
/usr/src/tensorrt/bin/trtexec --onnx=yolov8m-oiv7.onnx --saveEngine=yolov8m-oiv7.engine --fp16 && \
/usr/src/tensorrt/bin/trtexec --onnx=yolov8m-seg.onnx --saveEngine=yolov8s-seg.engine --fp16 && \
/usr/src/tensorrt/bin/trtexec --onnx=yolov8m-pose.onnx --saveEngine=yolov8m-pose.engine --fp16
```
测试yolo检测图片功能是否正常
```sh
python3 infer-det.py \
    --engine yolov8m.engine \
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
python3 click_and_track.py
```