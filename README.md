# SB-Tracker (Offline version)
当前这个版本主要是用给定的mp4视频对指定的一个气球做跟踪，笔记本算力负载，出一个能持续tracking的demo

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
开始部署侧优化的SiamMask算法(当前仅支持转为onnx，参考[博客](https://vjraj.dev/blog/siammask_onnx_export/))
```sh
cd /workspace/SiamMask/ && python3 export.py
```
尝试运行服务器的开放物体检测跟踪代码，如果没有显示相应的框、就需要自己拖拽鼠标框出
```sh
cd /workspace && git clone -b offline https://github.com/superboySB/SB-Tracker && cd SB-Tracker

python3 main.py --yolo_model_type=v8l --class_names="red balloon, red ball, balloon"
```
这里包含一个开集检测器，可以自己定义感兴趣的类别`--class_names`


![](assets/demo.gif)