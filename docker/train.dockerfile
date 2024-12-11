FROM nvcr.io/nvidia/tensorrt:24.01-py3

# Please contact with me if you have problems
LABEL maintainer="Zipeng Dai <daizipeng@bit.edu.cn>"
# TODO：网络不好的话可以走代理
ENV http_proxy=http://127.0.0.1:8889
ENV https_proxy=http://127.0.0.1:8889

# System Requirements
# ARG ROS_PACKAGE=ros_base
# ARG ROS_VERSION=humble
# ENV ROS_DISTRO=${ROS_VERSION}
# ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}
# ENV ROS_PYTHON_VERSION=3
# ARG PYTHON_VERSION=3.8
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash
SHELL ["/bin/bash", "-c"] 
WORKDIR /tmp
# RUN apt-get update && \
#     apt-get install -q -y --no-install-recommends tzdata software-properties-common && \
#     add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip libprotobuf-dev protobuf-compiler \
    locales git tmux gedit vim openmpi-bin openmpi-common libopenmpi-dev libgl1 libglx-mesa0
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# ONNX
# WORKDIR /workspace
# RUN git clone https://github.com/onnx/onnx.git && cd onnx && git submodule update --init --recursive && \
#     export CMAKE_ARGS=-DONNX_USE_LITE_PROTO=ON && pip3 install --upgrade pip && pip3 install -e . -v

# ROS
# RUN locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
# ENV LANG=en_US.UTF-8
# ENV PYTHONIOENCODING=utf-8
# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
# COPY docker/ros2_build.sh ros2_build.sh
# RUN . ros2_build.sh
# ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# COPY docker/ros_entrypoint.sh /ros_entrypoint.sh

# YOLOv8
WORKDIR /workspace
RUN git clone https://github.com/superboySB/YOLOv8-TensorRT.git
RUN cd YOLOv8-TensorRT && pip3 install -r requirements.txt && \
    pip3 install opencv-python==4.8.0.74 opencv-contrib-python==4.8.0.74 timm loguru && \
    # wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt && \
    # wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-seg.pt && \
    # wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-pose.pt && \
    wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-worldv2.pt && \
    wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-worldv2.pt
# RUN cd YOLOv8-TensorRT && \
#     python3 export-det.py --weights yolov8s.pt --sim && \
#     python3 export-seg.py --weights yolov8s-seg.pt --sim && \
#     yolo export model=yolov8s-pose.pt format=onnx simplify=True
RUN cd YOLOv8-TensorRT && python3 test_yoloworld.py

# EfficientViT + SAM
WORKDIR /workspace
RUN git clone https://github.com/superboySB/efficientvit
RUN cd efficientvit && pip install -r requirements.txt && mkdir -p assets/checkpoints/efficientvit_sam && cd assets/checkpoints/efficientvit_sam && \
    wget https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l2.pt && \
    wget https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_xl1.pt
RUN cd /workspace/efficientvit/ && mkdir -p assets/export_models/efficientvit_sam/tensorrt/ && chmod -R 777 assets/export_models/efficientvit_sam/tensorrt/  && \
    python applications/efficientvit_sam/deployment/onnx/export_encoder.py --model efficientvit-sam-l2 --output assets/export_models/efficientvit_sam/onnx/efficientvit_sam_l2_encoder.onnx && \
    python applications/efficientvit_sam/deployment/onnx/export_decoder.py --model efficientvit-sam-l2 --output assets/export_models/efficientvit_sam/onnx/efficientvit_sam_l2_decoder.onnx --return-single-mask && \
    python applications/efficientvit_sam/deployment/onnx/export_encoder.py --model efficientvit-sam-xl1 --output assets/export_models/efficientvit_sam/onnx/efficientvit_sam_xl1_encoder.onnx && \
    python applications/efficientvit_sam/deployment/onnx/export_decoder.py --model efficientvit-sam-xl1 --output assets/export_models/efficientvit_sam/onnx/efficientvit_sam_xl1_decoder.onnx --return-single-mask

# Siammask
WORKDIR /workspace
RUN git clone https://github.com/superboySB/SiamMask && cd SiamMask && pip3 install onnxoptimizer && \
    bash make.sh

# our project
RUN pip3 install pyrealsense2

WORKDIR /workspace
RUN rm -rf /var/lib/apt/lists/* && apt-get clean
# RUN chmod +x /ros_entrypoint.sh
# TODO：如果走了代理、但是想镜像本地化到其它机器，记得清空代理（或者容器内unset）
# ENV http_proxy=
# ENV https_proxy=
# ENV no_proxy=
# ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["/bin/bash"]