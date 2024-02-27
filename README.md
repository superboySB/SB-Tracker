# SB-Tracker

Language: [Chinese](./README_zh.md) / [English](./README.md)

An open-environment UAV Tracker based on human-computer interaction, deployed on the Jetson Orin board (NX as an example), will first use yolo-world to detect object categories for the user (based on CLIP's open-set detection, categories can be specified by the user), and then the user can double-click the object they want to track with the mouse to start tracking immediately. Since the Segment Anything model on the edge is loaded, users can click on objects of a specified category or objects not in the detection box to attempt tracking, r to reset, q to quit. O(∩_∩)O

## Algorithm Development/Cloud Services (Server Side)
```sh
# --progress=plain --no-cache=false
docker build -f docker/train.dockerfile -t sbt_image:train .

# Add --device /dev/video0:/dev/video0 if a camera hardware is connected
docker run -itd --privileged -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=$DISPLAY --runtime=nvidia --network=host --ipc host --name=sbtracker-train sbt_image:train /bin/bash

docker exec -it sbtracker-train /bin/bash
```
Begin deploying the server-side optimized [SiamMask](https://github.com/foolwood/SiamMask) algorithm (currently supports conversion to onnx, refer to the [blog]((https://vjraj.dev/blog/siammask_onnx_export/)))
```sh
cd /workspace/SiamMask/ && python export.py
```
Begin deploying the server-side optimized ViT algorithm (use `--verbose` for debugging, cost-effectiveness of xl1 and l2 models is detailed in [Han Song's team introduction]((https://github.com/mit-han-lab/efficientvit/blob/master/applications/sam.md)))
```sh
# Export Encoder and Decoder
trtexec --onnx=assets/export_models/sam/onnx/xl1_encoder.onnx --minShapes=input_image:1x3x1024x1024 --optShapes=input_image:4x3x1024x1024 --maxShapes=input_image:4x3x1024x1024 --saveEngine=assets/export_models/sam/tensorrt/xl1_encoder.engine && \
trtexec --onnx=assets/export_models/sam/onnx/xl1_decoder.onnx --minShapes=point_coords:1x1x2,point_labels:1x1 --optShapes=point_coords:16x2x2,point_labels:16x2 --maxShapes=point_coords:16x2x2,point_labels:16x2 --fp16 --saveEngine=assets/export_models/sam/tensorrt/xl1_decoder.engine
```
Attempt to run the detect-and_track code on server device (or just a powerful laptop)
```sh
cd /workspace && git clone https://github.com/superboySB/SB-Tracker && cd SB-Tracker

python main.py --device_type=server --yolo_model_type=v8l --sam_model_type=xl1 --class_names="red box,green pencil,white box"
```
This includes an open-set detector, where you can define the categories of interest with `--class_names`

## Algorithm Deployment/Real-Time Testing (Edge Side)
```sh
docker build -f docker/deploy.dockerfile -t sbt_image:deploy .

# Add if a camera hardware is connected, make sure the camera is installed:  ls /dev/video*
docker run -itd --privileged -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=$DISPLAY --runtime=nvidia --device /dev/video0:/dev/video0 --device /dev/snd --device /dev/bus/usb --network=host --ipc host --name=sbtracker-deploy sbt_image:deploy /bin/bash

docker exec -it sbtracker-deploy /bin/bash
```
Begin deploying the edge-side optimized SiamMask algorithm, same principle as the server side
```sh
cd /workspace/SiamMask/ && python export.py
```
Then, considering hardware universality, the ViT algorithm's edge deployment uses only the ONNX Optimizer, similar to the server side, the ONNX step has actually been handled inside the dockerfile and can be skipped
```sh
cd /workspace/efficientvit/ && \
python deployment/sam/onnx/export_encoder.py --model l2 --weight_url assets/checkpoints/sam/l2.pt --output assets/export_models/sam/onnx/l2_encoder.onnx && \ 
python deployment/sam/onnx/export_decoder.py --model l2 --weight_url assets/checkpoints/sam/l2.pt --output assets/export_models/sam/onnx/l2_decoder.onnx --return-single-mask && \
python deployment/sam/onnx/export_encoder.py --model xl1 --weight_url assets/checkpoints/sam/xl1.pt --output assets/export_models/sam/onnx/xl1_encoder.onnx && \ 
python deployment/sam/onnx/export_decoder.py --model xl1 --weight_url assets/checkpoints/sam/xl1.pt --output assets/export_models/sam/onnx/xl1_decoder.onnx --return-single-mask && \
python deployment/sam/onnx/inference.py --model l2 --encoder_model assets/export_models/sam/onnx/l2_encoder.onnx --decoder_model assets/export_models/sam/onnx/l2_decoder.onnx --mode point
```
Attempt to run the detect-and_track code on edge device
```sh
# git config --global --unset http.proxy

cd /workspace && git clone https://github.com/superboySB/SB-Tracker && cd SB-Tracker

python main.py --device_type=deployment --yolo_model_type=v8l --sam_model_type=l2 --class_names="person,computer case,screen"
```
This includes an open-set detector, where you can define the categories of interest with `--class_names`
![](assets/demo.gif)


## Acknowledgement
The work was done when the author visited Qiyuan Lab, supervised by [Chao Wang](https://scholar.google.com/citations?user=qmDGt-kAAAAJ&hl=zh-CN).