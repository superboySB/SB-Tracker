# SB-Tracker
开放环境下基于人机交互的UAV Tracker，部署在Jetson Orin板载上(以NANO/NX)为例，会先用yolo给用户检测物体类别，然后用户用鼠标双击要跟踪的物体，即可跟踪物体。由于加载了端侧的Segment Anything模型，用户既可以点击指定类别的物体，也可以点击不在检测框的物体，都可以尝试跟踪，`r`为重置，`q`为退出。O(∩_∩)O

## 当前进度
有一些很明显要改进的点
- [X] 显然可以用yolo的检测框来辅助给SAM画box，会比之前标point要准确很多，通过grounding dino+SAM+diffusion已经证明这样做有效。

## 算法开发/云上服务（服务器侧）
```sh
docker build -f docker/train.dockerfile -t sbt_image:train . --progress=plain --no-cache=false

# 如果连接了摄像头硬件就可以加--device /dev/video0:/dev/video0 
docker run -itd --privileged -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=$DISPLAY --runtime=nvidia --network=host --ipc host --name=sbtracker-train sbt_image:train /bin/bash

docker exec -it sbtracker-train /bin/bash
```
开始部署服务器侧优化的SiamMask算法(当前仅支持转为onnx，参考[博客](https://vjraj.dev/blog/siammask_onnx_export/))
```sh
cd /workspace/SiamMask/ && python export.py
```
开始部署服务器侧优化的ViT算法 (调试需要`--verbose`,xl1和l2模型的性价比详见[韩松团队介绍](https://github.com/mit-han-lab/efficientvit/blob/master/applications/sam.md))
```sh
# Export Encoder and Decoder
trtexec --onnx=assets/export_models/sam/onnx/xl1_encoder.onnx --minShapes=input_image:1x3x1024x1024 --optShapes=input_image:4x3x1024x1024 --maxShapes=input_image:4x3x1024x1024 --saveEngine=assets/export_models/sam/tensorrt/xl1_encoder.engine && \
trtexec --onnx=assets/export_models/sam/onnx/xl1_decoder.onnx --minShapes=point_coords:1x1x2,point_labels:1x1 --optShapes=point_coords:16x2x2,point_labels:16x2 --maxShapes=point_coords:16x2x2,point_labels:16x2 --fp16 --saveEngine=assets/export_models/sam/tensorrt/xl1_decoder.engine
```
尝试运行服务器的指哪儿打哪儿代码
```sh
cd /workspace && git clone https://github.com/superboySB/SB-Tracker && cd SB-Tracker

python main.py --device_type=server --yolo_model_type=v8l --sam_model_type=xl1 --class_names="red box,green pencil,white box"
```
这里包含一个开集检测器，可以自己定义感兴趣的类别`--class_names`


## 算法部署/搞机测试（端侧）
```sh
docker build -f docker/deploy.dockerfile -t sbt_image:deploy . --progress=plain

# 如果连接了摄像头硬件就可以加，确保摄像头安装:  ls /dev/video*
docker run -itd --privileged -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=$DISPLAY --runtime=nvidia --device /dev/video0:/dev/video0 --device /dev/snd --device /dev/bus/usb --network=host --ipc host --name=sbtracker-deploy sbt_image:deploy /bin/bash

docker exec -it sbtracker-deploy /bin/bash
```
开始部署端侧优化的SiamMask算法，原理同服务器侧
```sh
cd /workspace/SiamMask/ && python export.py
```
考虑硬件通用性，ViT算法只转化为ONNX，这一步已经在dockerfile里面完成了，因此可以直接尝试运行jetson的指哪儿打哪儿代码
```sh
cd /workspace && git clone https://github.com/superboySB/SB-Tracker && cd SB-Tracker

python main.py --device_type=deployment --yolo_model_type=v8l --sam_model_type=l2 --class_names="red box,green pencil,white box"
```
![](assets/demo.gif)

## Acknowledgement
The work was done when the author visited Qiyuan Lab, supervised by [Chao Wang](https://scholar.google.com/citations?user=qmDGt-kAAAAJ&hl=zh-CN).