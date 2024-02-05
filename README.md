# SB-Tracker
开放环境下基于人机交互的UAV Tracker

## 开箱即用
```sh
docker build -t sbt_image:r35.3.1 .

# 如果连接了摄像头硬件就可以加
docker run -itd --privileged -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=$DISPLAY --runtime=nvidia --device /dev/video0:/dev/video0 --device /dev/snd --device /dev/bus/usb --network=host --ipc host --name=sbtracker sbt_image:r35.3.1 /bin/bash

docker exec -it sbtracker /bin/bash

cd /workspace && git clone https://github.com/superboySB/SB-Tracker && cd SB-Tracker && python3 setup.py develop --user
```

### 检测部署
任务需要nanoowl做开放词汇检测的话，需要用tensorRT优化一下google提供的encoder，默认使用下面的命令就可以了
```sh
python3 -m nanoowl.build_image_encoder_engine \
        data/owl_image_encoder_patch32.engine \
        --onnx_opset=16

# TODO: 可以尝试
python3 -m nanoowl.build_image_encoder_engine \
        data/owl_v2_image_encoder_patch16_ensemble.engine \
        --model_name google/owlv2-base-patch16-ensemble \
        --onnx_opset 16
```
确保摄像头安装
```sh
ls /dev/video*
```
检验效果
```sh
cd examples/tree_demo

python3 tree_demo.py ../../data/owl_image_encoder_patch32.engine
```