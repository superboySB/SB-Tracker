FROM nvcr.io/nvidia/tensorrt:24.01-py3

# Please contact with me if you have problems
LABEL maintainer="Zipeng Dai <daizipeng@bit.edu.cn>"

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash
ENV LANG=en_US.UTF-8
ENV PYTHONIOENCODING=utf-8
SHELL ["/bin/bash", "-c"] 
WORKDIR /tmp
RUN apt-get update && \
    apt-get install -y --no-install-recommends locales git tmux gedit vim openmpi-bin openmpi-common libopenmpi-dev libgl1-mesa-glx

# YOLOv8
WORKDIR /workspace
RUN git clone https://github.com/superboySB/YOLOv8-TensorRT.git
RUN cd YOLOv8-TensorRT && pip install --upgrade pip && pip install -r requirements.txt && \
    wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt && \
    wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-oiv7.pt && \
    wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-seg.pt && \
    wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-pose.pt
RUN cd YOLOv8-TensorRT && \
    python export-det.py --weights yolov8s.pt --sim && \
    python export-det.py --weights yolov8s-oiv7.pt --sim && \
    python export-seg.py --weights yolov8s-seg.pt --sim && \
    yolo export model=yolov8s-pose.pt format=onnx simplify=True

# EfficientViT + SAM
WORKDIR /workspace
RUN git clone https://github.com/superboySB/efficientvit.git
RUN cd efficientvit && pip install -r requirements.txt && mkdir -p assets/checkpoints/sam && cd assets/checkpoints/sam && \
    wget https://huggingface.co/han-cai/efficientvit-sam/resolve/main/l2.pt && \
    wget https://huggingface.co/han-cai/efficientvit-sam/resolve/main/xl1.pt
RUN cd /workspace/efficientvit/ && \
    python deployment/sam/onnx/export_encoder.py --model l2 --weight_url assets/checkpoints/sam/l2.pt --output assets/export_models/sam/onnx/l2_encoder.onnx && \ 
    python deployment/sam/onnx/export_decoder.py --model l2 --weight_url assets/checkpoints/sam/l2.pt --output assets/export_models/sam/onnx/l2_decoder.onnx --return-single-mask && \
    python deployment/sam/onnx/export_encoder.py --model xl1 --weight_url assets/checkpoints/sam/xl1.pt --output assets/export_models/sam/onnx/xl1_encoder.onnx && \ 
    python deployment/sam/onnx/export_decoder.py --model xl1 --weight_url assets/checkpoints/sam/xl1.pt --output assets/export_models/sam/onnx/xl1_decoder.onnx --return-single-mask

WORKDIR /workspace
RUN rm -rf /var/lib/apt/lists/* && apt-get clean
CMD ["/bin/bash"]