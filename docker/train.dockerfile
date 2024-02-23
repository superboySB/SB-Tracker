FROM nvcr.io/nvidia/tensorrt:24.01-py3

# Please contact with me if you have problems
LABEL maintainer="Zipeng Dai <daizipeng@bit.edu.cn>"

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
    pip install opencv-python==4.8.0.74 opencv-contrib-python==4.8.0.74 && \
    wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt && \
    wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-seg.pt && \
    wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-pose.pt && \
    wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-world.pt && \
    wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-world.pt
RUN cd YOLOv8-TensorRT && \
    python export-det.py --weights yolov8s.pt --sim && \
    python export-seg.py --weights yolov8s-seg.pt --sim && \
    yolo export model=yolov8s-pose.pt format=onnx simplify=True
RUN cd YOLOv8-TensorRT && python test_yoloworld.py

# EfficientViT + SAM
WORKDIR /workspace
RUN git clone https://github.com/superboySB/efficientvit.git
RUN cd efficientvit && pip install -r requirements.txt && mkdir -p assets/checkpoints/sam && cd assets/checkpoints/sam && \
    wget https://huggingface.co/han-cai/efficientvit-sam/resolve/main/l2.pt && \
    wget https://huggingface.co/han-cai/efficientvit-sam/resolve/main/xl1.pt
RUN cd /workspace/efficientvit/ && mkdir -p assets/export_models/sam/tensorrt/ && chmod -R 777 assets/export_models/sam/tensorrt/ && \
    python deployment/sam/onnx/export_encoder.py --model l2 --weight_url assets/checkpoints/sam/l2.pt --output assets/export_models/sam/onnx/l2_encoder.onnx && \ 
    python deployment/sam/onnx/export_decoder.py --model l2 --weight_url assets/checkpoints/sam/l2.pt --output assets/export_models/sam/onnx/l2_decoder.onnx --return-single-mask && \
    python deployment/sam/onnx/export_encoder.py --model xl1 --weight_url assets/checkpoints/sam/xl1.pt --output assets/export_models/sam/onnx/xl1_encoder.onnx && \ 
    python deployment/sam/onnx/export_decoder.py --model xl1 --weight_url assets/checkpoints/sam/xl1.pt --output assets/export_models/sam/onnx/xl1_decoder.onnx --return-single-mask

# Siammask
WORKDIR /workspace
RUN git clone https://github.com/superboySB/SiamMask && cd SiamMask && pip install onnxoptimizer && bash make.sh

WORKDIR /workspace
RUN rm -rf /var/lib/apt/lists/* && apt-get clean
CMD ["/bin/bash"]