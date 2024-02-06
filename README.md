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
先自己准备一个训练好的pytorch模型（这里默认先用自带的）,然后添加bbox decoder、NMS后转为ONNX模型。
```sh
cd /workspace/YOLOv8-TensorRT

python3 export-det.py \
    --weights yolov8s.pt \
    --iou-thres 0.65 \
    --conf-thres 0.25 \
    --topk 100 \
    --opset 11 \
    --sim \
    --input-shape 1 3 640 640 \
    --device cuda:0
```
使用TensorRT的python api来适配yolo模型
```sh
python3 build.py \
    --weights yolov8s.onnx \
    --iou-thres 0.65 \
    --conf-thres 0.25 \
    --topk 100 \
    --fp16  \
    --device cuda:0
```
测试检测功能是否正常
```sh
python3 infer-det.py \
    --engine yolov8s.engine \
    --imgs data \
    --show \
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
python3 examples/demo_click_segment_track.py \
    --image_encoder="data/resnet18_image_encoder.engine" \
    --mask_decoder="data/mobile_sam_mask_decoder.engine"
```