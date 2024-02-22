# SB-Tracker
开放环境下基于人机交互的UAV Tracker，部署在Jetson Orin板载上(以NANO/NX)为例，会先用yolo给用户检测物体类别，然后用户用鼠标双击要跟踪的物体，即可跟踪物体。由于加载了端侧的Segment Anything模型，用户既可以点击指定类别的物体，也可以点击不在检测框的物体，都可以尝试跟踪，`r`为重置，`q`为退出。O(∩_∩)O

## 当前进度
有一些很明显要改进的点
- [ ] 显然可以用yolo的检测框来辅助给SAM画box，会比之前标point要准确很多，通过grounding dino+SAM+diffusion已经证明这样做有效。

## 算法开发（云侧）
```sh
docker build -f docker/train.dockerfile -t sbt_image:train . --progress=plain --no-cache=false

# 如果连接了摄像头硬件就可以加--device /dev/video0:/dev/video0 
docker run -itd --privileged -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=$DISPLAY --runtime=nvidia --network=host --ipc host --name=sbtracker-train sbt_image:train /bin/bash

docker exec -it sbtracker-train /bin/bash
```
**(Optional)** 这一步一定要在台式机上(也可以考虑使用`train.dockerfile`在run完的容器直接拿onnx)，我们需要在docker中默认准备好的一个训练好的自带pytorch模型作为范例,然后添加bbox decoder、NMS后转为ONNX模型。
```sh
cd /workspace/YOLOv8-TensorRT
python3 export-det.py --weights yolov8s.pt --sim && \
python3 export-seg.py --weights yolov8s-seg.pt --sim && \
yolo export model=yolov8s-pose.pt format=onnx simplify=True
```
开始部署云侧优化的YoloV8算法
```sh
cd /workspace/YOLOv8-TensorRT
/usr/src/tensorrt/bin/trtexec --onnx=yolov8s.onnx --saveEngine=yolov8s.engine --fp16 && \
/usr/src/tensorrt/bin/trtexec --onnx=yolov8s-seg.onnx --saveEngine=yolov8s-seg.engine --fp16 && \
/usr/src/tensorrt/bin/trtexec --onnx=yolov8s-pose.onnx --saveEngine=yolov8s-pose.engine --fp16
```
开始部署云侧优化的SiamMask算法(当前仅支持onnx，参考[博客](https://vjraj.dev/blog/siammask_onnx_export/))
```sh
cd /workspace/Siammask/
python export.py && python main.py --model siammask_vot_simp.onnx
```
开始部署云侧优化的ViT算法 (调试需要`--verbose`,xl1和l2模型的性价比详见[韩松团队介绍](https://github.com/mit-han-lab/efficientvit/blob/master/applications/sam.md))
```sh
# Export Encoder
trtexec --onnx=assets/export_models/sam/onnx/xl1_encoder.onnx --minShapes=input_image:1x3x1024x1024 --optShapes=input_image:4x3x1024x1024 --maxShapes=input_image:4x3x1024x1024 --saveEngine=assets/export_models/sam/tensorrt/xl1_encoder.engine
# Export Decoder
trtexec --onnx=assets/export_models/sam/onnx/xl1_decoder.onnx --minShapes=point_coords:1x1x2,point_labels:1x1 --optShapes=point_coords:16x2x2,point_labels:16x2 --maxShapes=point_coords:16x2x2,point_labels:16x2 --fp16 --saveEngine=assets/export_models/sam/tensorrt/xl1_decoder.engine
# TensorRT Inference
python deployment/sam/tensorrt/inference.py --model xl1 --encoder_engine assets/export_models/sam/tensorrt/xl1_encoder.engine --decoder_engine assets/export_models/sam/tensorrt/xl1_decoder.engine --mode point
```

## 算法部署（端侧）
```sh
docker build -f docker/deploy.dockerfile -t sbt_image:deploy . --progress=plain

# 如果连接了摄像头硬件就可以加，确保摄像头安装:  ls /dev/video*
docker run -itd --privileged -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=$DISPLAY --runtime=nvidia --device /dev/video0:/dev/video0 --device /dev/snd --device /dev/bus/usb --network=host --ipc host --name=sbtracker-deploy sbt_image:deploy /bin/bash

docker exec -it sbtracker-deploy /bin/bash
```
### 开放环境检测功能部署
这一步一定要在jetson本机上，使用TensorRT的python api来适配几个yolo模型
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

### 运行DEMO
检验点击跟踪功能
```sh
cd /workspace && git clone https://github.com/superboySB/SB-Tracker && cd SB-Tracker

python3 det-main.py
```
![](assets/demo.gif)

## Acknowledgement
The work was done when the author visited Qiyuan Lab, supervised by [Chao Wang](https://scholar.google.com/citations?user=qmDGt-kAAAAJ&hl=zh-CN).