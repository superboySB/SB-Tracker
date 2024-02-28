# SB-Tracker

文档语言: [中文](./README_zh.md) / [英文](./README.md)

开放环境下基于人机交互的UAV Tracker，部署在Jetson Orin板载上(以NX为例)，会先用yolo-world给用户检测物体类别（基于CLIP的开集检测，类别可以自己给），然后用户用鼠标点击要跟踪的物体，可以即刻跟踪物体。由于同时加载了端侧的Segment Anything模型，用户既可以点击指定类别框内的物体，也可以临时点击视野内没有标记检测框的任意物体，都可以尝试进行跟踪，`r`为重置，`q`为退出。

## 当前进度
- [X] 引入CLIP做开集检测
- [X] 引入EfficientViT+SAM给未知物体画框
- [X] 引入Siamese Network中的经典方法（SiamMask/NanoTrack）在不同条件的机器上做点击跟踪
- [ ] 对所有模块引入TensorRT支持
- [ ] 现在端侧tracking的推理没有实际用到生成的两个onnx，可以进一步转为onnx加速

## 算法开发/云上服务（服务器侧）
```sh
# --progress=plain --no-cache=false
docker build -f docker/train.dockerfile -t sbt_image:train .

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
cd /workspace/efficientvit && \
trtexec --onnx=assets/export_models/sam/onnx/xl1_encoder.onnx --minShapes=input_image:1x3x1024x1024 --optShapes=input_image:4x3x1024x1024 --maxShapes=input_image:4x3x1024x1024 --saveEngine=assets/export_models/sam/tensorrt/xl1_encoder.engine && \
trtexec --onnx=assets/export_models/sam/onnx/xl1_decoder.onnx --minShapes=point_coords:1x1x2,point_labels:1x1 --optShapes=point_coords:16x2x2,point_labels:16x2 --maxShapes=point_coords:16x2x2,point_labels:16x2 --fp16 --saveEngine=assets/export_models/sam/tensorrt/xl1_decoder.engine && \
python deployment/sam/tensorrt/inference.py --model xl1 --encoder_engine assets/export_models/sam/tensorrt/xl1_encoder.engine --decoder_engine assets/export_models/sam/tensorrt/xl1_decoder.engine --mode point
```
尝试运行服务器的开放物体检测跟踪代码
```sh
cd /workspace && git clone https://github.com/superboySB/SB-Tracker && cd SB-Tracker

python main.py --use_tensorrt --device_type=server --yolo_model_type=v8l --sam_model_type=xl1 --class_names="red box,green pencil,white box"
```
这里包含一个开集检测器，可以自己定义感兴趣的类别`--class_names`


## 算法部署/实机测试（端侧）
```sh
docker build -f docker/deploy.dockerfile -t sbt_image:deploy .

# 如果连接了摄像头硬件就可以加，确保摄像头安装:  ls /dev/video*
docker run -itd --privileged -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=$DISPLAY --runtime=nvidia --device /dev/video0:/dev/video0 --device /dev/snd --device /dev/bus/usb --network=host --ipc host --name=sbtracker-deploy sbt_image:deploy /bin/bash

docker exec -it sbtracker-deploy /bin/bash
```
考虑硬件通用性（不一定都有GPU），ViT算法的端侧部署建议只使用ONNX Optimizer，所以主要运行下面的命令（已经在docker build阶段提前处理好）
```sh
cd /workspace/efficientvit/ && \
python deployment/sam/onnx/export_encoder.py --model l2 --weight_url assets/checkpoints/sam/l2.pt --output assets/export_models/sam/onnx/l2_encoder.onnx && \ 
python deployment/sam/onnx/export_decoder.py --model l2 --weight_url assets/checkpoints/sam/l2.pt --output assets/export_models/sam/onnx/l2_decoder.onnx --return-single-mask && \
python deployment/sam/onnx/inference.py --model l2 --encoder_model assets/export_models/sam/onnx/l2_encoder.onnx --decoder_model assets/export_models/sam/onnx/l2_decoder.onnx --mode point
```
[正在开发中] 开始部署端侧优化后的NanoTrackV3算法，原理类似服务器侧的Siammask
```sh
cd /workspace/SiamTrackers/NanoTrack/ && mkdir models/onnx && python pytorch2onnx.py
```
[正在开发中] 如果一定需要用到tensorrt，可以运行和服务器侧`trtexec`一样的命令来准备相应的模型 （端侧需要支持最新版本的TensorRT和Jetpack 6）
```sh
/usr/src/tensorrt/bin/trtexec --onnx=assets/export_models/sam/onnx/l2_encoder.onnx --minShapes=input_image:1x3x512x512 --optShapes=input_image:4x3x512x512 --maxShapes=input_image:4x3x512x512 --saveEngine=assets/export_models/sam/tensorrt/l2_encoder.engine && \
/usr/src/tensorrt/bin/trtexec --onnx=assets/export_models/sam/onnx/l2_decoder.onnx --minShapes=point_coords:1x1x2,point_labels:1x1 --optShapes=point_coords:16x2x2,point_labels:16x2 --maxShapes=point_coords:16x2x2,point_labels:16x2 --fp16 --saveEngine=assets/export_models/sam/tensorrt/l2_decoder.engine && \
python deployment/sam/tensorrt/inference.py --model l2 --encoder_engine assets/export_models/sam/tensorrt/l2_encoder.engine --decoder_engine assets/export_models/sam/tensorrt/l2_decoder.engine --mode point
```
现在可以直接尝试运行jetson的开放物体检测跟踪代码
```sh
# git config --global --unset http.proxy

cd /workspace && git clone https://github.com/superboySB/SB-Tracker && cd SB-Tracker

python main.py --device_type=deployment --yolo_model_type=v8l --sam_model_type=l2 --class_names="person,computer case,black box"
```
这里包含一个开集检测器，可以自己定义感兴趣的类别`--class_names`

![](assets/demo.gif)

## 对接HITL仿真与飞控（正在做）
先单独做飞控固件的isaac ros节点
```sh
docker build -f docker/px4.dockerfile -t sbt_image:px4 .

# 如果连接了摄像头硬件就可以加，确保摄像头安装:  ls /dev/video*
docker run -itd --privileged -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=$DISPLAY --runtime=nvidia --device /dev/video0:/dev/video0 --device /dev/snd --device /dev/bus/usb --network=host --ipc host --name=mypx4 sbt_image:px4 /bin/bash

docker exec -it sbtracker-px4 /bin/bash
```

## Acknowledgement
The work was done when the author visited Qiyuan Lab, supervised by [Chao Wang](https://scholar.google.com/citations?user=qmDGt-kAAAAJ&hl=zh-CN).