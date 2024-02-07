FROM nvcr.io/nvidia/pytorch:20.03-py3

# Please contact with me if you have problems
LABEL maintainer="Zipeng Dai <daizipeng@bit.edu.cn>"

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash
SHELL ["/bin/bash", "-c"] 
WORKDIR /tmp

# change the locale from POSIX to UTF-8
RUN apt-get update && \
    apt-get install -y --no-install-recommends locales git tmux gedit vim

RUN locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV PYTHONIOENCODING=utf-8

WORKDIR /workspace
RUN git clone https://github.com/triple-Mu/YOLOv8-TensorRT.git && cd YOLOv8-TensorRT && pip install --upgrade pip && \
    pip install -r requirements.txt && pip install ultralytics && pip install opencv-python==4.8.0.74
# Export *.pt to *.onnx
RUN cd /workspace/YOLOv8-TensorRT && \
    wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt && \
    wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-oiv7.pt && \
    wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-seg.pt && \
    wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-pose.pt
RUN python3 export-det.py --weights yolov8s.pt --sim && \
    python3 export-det.py --weights yolov8s-oiv7.pt --sim && \
    python3 export-seg.py --weights yolov8s-seg.pt --sim && \
    yolo export model=yolov8s-pose.pt format=onnx simplify=True

WORKDIR /workspace
RUN rm -rf /var/lib/apt/lists/* && apt-get clean
CMD ["/bin/bash"]