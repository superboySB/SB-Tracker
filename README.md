# SB-Tracker

Document Language: [Chinese](./README_zh.md) / [English](./README.md)

Open environment UAV Tracker based on human-computer interaction, deployed on Jetson Orin board (taking NX as an example), will first use yolo-world to detect object categories for the user (based on CLIP's open set detection, categories can be defined by the user), and then the user can click on the object to be tracked with the mouse, and track the object immediately. Since the Segment Anything model on the edge side is also loaded, users can click on objects within a specified category box, or temporarily click on any object in the field of view without a marked detection box, and try to track, `r` for reset, `q` for exit.

## Current Progress
- [X] Introduced CLIP for open set detection
- [X] Introduced EfficientViT+SAM to draw boxes for unknown objects
- [X] Introduced classic methods from Siamese Network (SiamMask/NanoTrack) for click tracking on machines under different conditions
- [ ] Introduce TensorRT support for all modules
- [ ] The inference of edge side tracking has not actually used the two generated onnx, which can be further converted to onnx for acceleration

## Algorithm Development/Cloud Services (Server or Laptop Side)
```sh
# --progress=plain --no-cache=false
docker build -f docker/train.dockerfile -t sbt_image:train .

# If a camera hardware is connected, you can add --device /dev/video0:/dev/video0
docker run -itd --privileged -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=$DISPLAY --runtime=nvidia --network=host --ipc host --name=sbtracker-train sbt_image:train /bin/bash

docker exec -it sbtracker-train /bin/bash
```
Start deploying server-side optimized SiamMask algorithm (currently only supports conversion to onnx, refer to [blog](https://vjraj.dev/blog/siammask_onnx_export/))
```sh
cd /workspace/SiamMask/ && python export.py
```
Start deploying server-side optimized ViT algorithm (debugging needs `--verbose`, cost-effectiveness of xl1 and l2 models see [Han Song's project introduction]((https://github.com/mit-han-lab/efficientvit/blob/master/applications/sam.md)))
```sh
cd /workspace/efficientvit && \
trtexec --onnx=assets/export_models/sam/onnx/xl1_encoder.onnx --minShapes=input_image:1x3x1024x1024 --optShapes=input_image:4x3x1024x1024 --maxShapes=input_image:4x3x1024x1024 --saveEngine=assets/export_models/sam/tensorrt/xl1_encoder.engine && \
trtexec --onnx=assets/export_models/sam/onnx/xl1_decoder.onnx --minShapes=point_coords:1x1x2,point_labels:1x1 --optShapes=point_coords:16x2x2,point_labels:16x2 --maxShapes=point_coords:16x2x2,point_labels:16x2 --fp16 --saveEngine=assets/export_models/sam/tensorrt/xl1_decoder.engine && \
python deployment/sam/tensorrt/inference.py --model xl1 --encoder_engine assets/export_models/sam/tensorrt/xl1_encoder.engine --decoder_engine assets/export_models/sam/tensorrt/xl1_decoder.engine --mode point
```
Attempt to run the server's open object detection tracking code
```sh
cd /workspace && git clone https://github.com/superboySB/SB-Tracker && cd SB-Tracker

python main.py --use_tensorrt --device_type=server --yolo_model_type=v8l --sam_model_type=xl1 --class_names="red box,green pencil,white box"
```
This includes an open set detector, you can define your own interested categories `--class_names`

## Algorithm Deployment/Field Testing (Edge Side)
```sh
docker build -f docker/deploy.dockerfile -t sbt_image:deploy .

# If a camera hardware is connected, you can add, make sure the camera is installed:  ls /dev/video*
docker run -itd --privileged -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=$DISPLAY --runtime=nvidia --device /dev/video0:/dev/video0 --device /dev/snd --device /dev/bus/usb --network=host --ipc host --name=sbtracker-deploy sbt_image:deploy /bin/bash

docker exec -it sbtracker-deploy /bin/bash
```
Considering the universality of hardware (not all devices have GPU), the deployment of ViT algorithm on the edge side is recommended to use only ONNX Optimizer, so mainly run the following commands (already finished during `docker build` stage)
```sh
cd /workspace/efficientvit/ && \
python deployment/sam/onnx/export_encoder.py --model l2 --weight_url assets/checkpoints/sam/l2.pt --output assets/export_models/sam/onnx/l2_encoder.onnx && \ 
python deployment/sam/onnx/export_decoder.py --model l2 --weight_url assets/checkpoints/sam/l2.pt --output assets/export_models/sam/onnx/l2_decoder.onnx --return-single-mask && \
python deployment/sam/onnx/inference.py --model l2 --encoder_model assets/export_models/sam/onnx/l2_encoder.onnx --decoder_model assets/export_models/sam/onnx/l2_decoder.onnx --mode point
```
[In Development] Start deploying the edge-optimized NanoTrackV3 algorithm, similar in principle to the server-side Siammask
```sh
cd /workspace/SiamTrackers/NanoTrack/ && mkdir models/onnx && python pytorch2onnx.py
```
[In Development] If tensorrt is definitely needed, you can run the same `trtexec` command as on the server side to prepare the corresponding model (the edge side needs to support the latest version of TensorRT and Jetpack 6)
```sh
/usr/src/tensorrt/bin/trtexec --onnx=assets/export_models/sam/onnx/l2_encoder.onnx --minShapes=input_image:1x3x512x512 --optShapes=input_image:4x3x512x512 --maxShapes=input_image:4x3x512x512 --saveEngine=assets/export_models/sam/tensorrt/l2_encoder.engine && \
/usr/src/tensorrt/bin/trtexec --onnx=assets/export_models/sam/onnx/l2_decoder.onnx --minShapes=point_coords:1x1x2,point_labels:1x1 --optShapes=point_coords:16x2x2,point_labels:16x2 --maxShapes=point_coords:16x2x2,point_labels:16x2 --fp16 --saveEngine=assets/export_models/sam/tensorrt/l2_decoder.engine && \
python deployment/sam/tensorrt/inference.py --model l2 --encoder_engine assets/export_models/sam/tensorrt/l2_encoder.engine --decoder_engine assets/export_models/sam/tensorrt/l2_decoder.engine --mode point
```
Now you can directly try to run the jetson's open object detection tracking code
```sh
# git config --global --unset http.proxy

cd /workspace && git clone https://github.com/superboySB/SB-Tracker && cd SB-Tracker

python main.py --device_type=deployment --yolo_model_type=v8l --sam_model_type=l2 --class_names="person,computer case,black box"
```
This includes an open set detector, you can define your own interested categories `--class_names`

![](assets/demo.gif)

## Interfacing with HITL Simulation and Flight Control (In Progress)
First, do the flight control firmware isaac ros node separately
```sh
docker build -f docker/px4.dockerfile -t sbt_image:px4 .

# If a camera hardware is connected, you can add, make sure the camera is installed:  ls /dev/video*
docker run -itd --privileged -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=$DISPLAY --runtime=nvidia --device /dev/video0:/dev/video0 --device /dev/snd --device /dev/bus/usb --network=host --ipc host --name=mypx4 sbt_image:px4 /bin/bash

docker exec -it sbtracker-px4 /bin/bash
```

## Acknowledgement
The work was done when the author visited Qiyuan Lab, supervised by [Chao Wang](https://scholar.google.com/citations?user=qmDGt-kAAAAJ&hl=zh-CN).