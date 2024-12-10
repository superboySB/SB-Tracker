FROM nvcr.io/nvidia/tensorrt:24.11-py3

# Please contact with me if you have problems
LABEL maintainer="Zipeng Dai <daizipeng@bit.edu.cn>"
# TODO：网络不好的话可以走代理
ENV http_proxy=http://127.0.0.1:8889
ENV https_proxy=http://127.0.0.1:8889

# System Requirements
ARG ROS_PACKAGE=ros_base
ARG ROS_VERSION=humble
ENV ROS_DISTRO=${ROS_VERSION}
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}
ENV ROS_PYTHON_VERSION=3
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash
SHELL ["/bin/bash", "-c"] 
WORKDIR /tmp
RUN apt-get update && \
    apt-get install -y --no-install-recommends locales git tmux gedit vim openmpi-bin openmpi-common libopenmpi-dev libgl1-mesa-glx

# ROS
RUN locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV PYTHONIOENCODING=utf-8
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
COPY docker/ros2_build.sh ros2_build.sh
RUN . ros2_build.sh
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
COPY docker/ros_entrypoint.sh /ros_entrypoint.sh

# YOLOv8
WORKDIR /workspace
RUN git clone https://github.com/superboySB/YOLOv8-TensorRT.git
RUN cd YOLOv8-TensorRT && pip install --upgrade pip && pip install -r requirements.txt && \
    pip install opencv-python==4.8.0.74 opencv-contrib-python==4.8.0.74 timm && \
    wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt && \
    wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-seg.pt && \
    wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-pose.pt && \
    wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-worldv2.pt && \
    wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-worldv2.pt
RUN cd YOLOv8-TensorRT && \
    python export-det.py --weights yolov8s.pt --sim && \
    python export-seg.py --weights yolov8s-seg.pt --sim && \
    yolo export model=yolov8s-pose.pt format=onnx simplify=True
RUN cd YOLOv8-TensorRT && python test_yoloworld.py

# EfficientViT + SAM
WORKDIR /workspace
RUN git clone https://github.com/superboySB/efficientvit.git
RUN cd efficientvit && pip install -r requirements.txt && mkdir -p assets/checkpoints/sam && cd assets/checkpoints/sam && \
    wget https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l2.pt && \
    wget https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_xl1.pt
RUN cd /workspace/efficientvit/ && mkdir -p assets/export_models/sam/tensorrt/ && chmod -R 777 assets/export_models/sam/tensorrt/ && \
    python deployment/sam/onnx/export_encoder.py --model l2 --weight_url assets/checkpoints/sam/efficientvit_sam_l2.pt --output assets/export_models/sam/onnx/l2_encoder.onnx && \ 
    python deployment/sam/onnx/export_decoder.py --model l2 --weight_url assets/checkpoints/sam/efficientvit_sam_l2.pt --output assets/export_models/sam/onnx/l2_decoder.onnx --return-single-mask && \
    python deployment/sam/onnx/export_encoder.py --model xl1 --weight_url assets/checkpoints/sam/efficientvit_sam_xl1.pt --output assets/export_models/sam/onnx/xl1_encoder.onnx && \ 
    python deployment/sam/onnx/export_decoder.py --model xl1 --weight_url assets/checkpoints/sam/efficientvit_sam_xl1.pt --output assets/export_models/sam/onnx/xl1_decoder.onnx --return-single-mask

# Siammask
WORKDIR /workspace
RUN git clone https://github.com/superboySB/SiamMask && cd SiamMask && pip install onnxoptimizer && bash make.sh

# our project
RUN pip install pyrealsense2

WORKDIR /workspace
RUN rm -rf /var/lib/apt/lists/* && apt-get clean
RUN chmod +x /ros_entrypoint.sh
# TODO：如果走了代理、但是想镜像本地化到其它机器，记得清空代理（或者容器内unset）
# ENV http_proxy=
# ENV https_proxy=
# ENV no_proxy=
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["/bin/bash"]