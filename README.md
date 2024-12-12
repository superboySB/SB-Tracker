# SB-Tracker (Realsense D435i version)

开放环境下基于人机交互的UAV Tracker，部署在Jetson Orin板载上(以NX为例)，会先用yolo-world给用户检测物体类别（基于CLIP的开集检测，类别可以自己给），然后用户用鼠标点击要跟踪的物体，可以即刻跟踪物体。由于同时加载了端侧的Segment Anything模型，用户既可以点击指定类别框内的物体，也可以临时点击视野内没有标记检测框的任意物体，都可以尝试进行跟踪，`r`为重置，`q`为退出。

## 当前进度
- [X] 引入CLIP做开集检测
- [X] 引入EfficientViT+SAM给未知物体画框
- [X] 引入Siamese Network中的经典方法（SiamMask/NanoTrack）在不同条件的机器上做点击跟踪
- [ ] 对所有模块引入TensorRT支持
- [ ] 对接HITL仿真与飞控

## 在笔记本电脑上测试
```sh
# --no-cache=false
docker build -f docker/laptop.dockerfile -t sbt_image:train --network=host --progress=plain .

docker run -itd --privileged --name=sbtracker-train \
--volume /tmp/.X11-unix:/tmp/.X11-unix \
--env DISPLAY=$DISPLAY \
--env QT_X11_NO_MITSHM=1 \
--gpus all \
--network=host \
--device /dev/video0:/dev/video0 \
--device /dev/video1:/dev/video1 \
--device /dev/video2:/dev/video2 \
--device /dev/video3:/dev/video3 \
--device /dev/video4:/dev/video4 \
--device /dev/video5:/dev/video5 \
--device /dev/video6:/dev/video6 \
--device /dev/video7:/dev/video7 \
--device /dev/media0:/dev/media0 \
--device /dev/media1:/dev/media1 \
--device /dev/media2:/dev/media2 \
sbt_image:train /bin/bash

docker exec -it sbtracker-train /bin/bash
```
开始部署服务器侧优化的SiamMask算法(当前仅支持转为onnx，参考[博客](https://vjraj.dev/blog/siammask_onnx_export/))
```sh
cd /workspace/SiamMask/ && python export.py
```
开始部署服务器侧优化的ViT算法 (调试需要`--verbose`,xl1和l2模型的性价比详见[韩松团队介绍](https://github.com/mit-han-lab/efficientvit/tree/master/applications/efficientvit_sam))
```sh
cd /workspace/efficientvit && \
trtexec --onnx=assets/export_models/efficientvit_sam/onnx/efficientvit_sam_xl1_encoder.onnx --minShapes=input_image:1x3x1024x1024 --optShapes=input_image:4x3x1024x1024 --maxShapes=input_image:4x3x1024x1024 --saveEngine=assets/export_models/efficientvit_sam/tensorrt/efficientvit_sam_xl1_encoder.engine && \
trtexec --onnx=assets/export_models/efficientvit_sam/onnx/efficientvit_sam_xl1_decoder.onnx --minShapes=point_coords:1x1x2,point_labels:1x1 --optShapes=point_coords:16x2x2,point_labels:16x2 --maxShapes=point_coords:16x2x2,point_labels:16x2 --fp16 --saveEngine=assets/export_models/efficientvit_sam/tensorrt/efficientvit_sam_xl1_decoder.engine && \
python3 applications/efficientvit_sam/run_efficientvit_sam_trt.py --model efficientvit-sam-xl1 --encoder_engine assets/export_models/efficientvit_sam/tensorrt/efficientvit_sam_xl1_encoder.engine --decoder_engine assets/export_models/efficientvit_sam/tensorrt/efficientvit_sam_xl1_decoder.engine --mode point
```
尝试运行服务器的开放物体检测跟踪代码
```sh
cd /workspace && git clone -b realsense https://github.com/superboySB/SB-Tracker && cd SB-Tracker

python main.py --use_tensorrt --yolo_model_type=v8l --sam_model_type=xl1 --class_names="nlue bottle, white cup"
```
这里包含一个开集检测器，可以自己定义感兴趣的类别`--class_names`


![](assets/demo.gif)