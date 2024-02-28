# TODO： Need to be upgrade to r36.2 to support the latest tensorRT
FROM dustynv/ros:foxy-pytorch-l4t-r35.3.1

# Please contact with me if you have problems
LABEL maintainer="Zipeng Dai <daizipeng@bit.edu.cn>"

# System Requirements
ENV SHELL /bin/bash
WORKDIR /tmp
SHELL ["/bin/bash", "-c"] 
RUN apt-get update && \
    apt-get install -y --no-install-recommends locales git tmux gedit vim openmpi-bin openmpi-common libopenmpi-dev libgl1-mesa-glx \
    gcc zip curl htop libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0 ffmpeg

# PX4 and relevant infra
WORKDIR /workspace
RUN git clone https://github.com/PX4/PX4-Autopilot.git -b v1.14.1 --recursive
COPY docker/requirements.txt /workspace/PX4-Autopilot/requirements.txt
COPY docker/ubuntu.sh /workspace/PX4-Autopilot/ubuntu.sh 
RUN cd PX4-Autopilot/ && bash ubuntu.sh --no-nuttx && make clean
RUN cd PX4-Autopilot/ && DONT_RUN=1 make px4_sitl_default gazebo-classic
WORKDIR /workspace
RUN . /opt/ros/foxy/install/setup.sh && git clone https://github.com/Jaeyoung-Lim/px4-offboard.git && cd px4-offboard && colcon build && cd ..
RUN wget https://d176tv9ibo4jno.cloudfront.net/latest/QGroundControl.AppImage && chmod +x ./QGroundControl.AppImage 
# RUN . /opt/ros/foxy/install/setup.sh && mkdir -p /workspace/px4_ros_com_ws/src && cd /workspace/px4_ros_com_ws/src && \
#     git clone https://github.com/PX4/px4_msgs.git -b release/1.14 && cd .. && colcon build
WORKDIR /workspace
RUN . /opt/ros/foxy/install/setup.sh && mkdir microros_ws && cd microros_ws && \
    git clone -b foxy https://github.com/micro-ROS/micro_ros_setup.git src/micro_ros_setup && \
    rosdep update && rosdep install --from-paths src --ignore-src -y && colcon build && . install/local_setup.sh && \
    ros2 run micro_ros_setup create_agent_ws.sh && ros2 run micro_ros_setup build_agent.sh

# For our projects
# EfficientViT + SAM
WORKDIR /workspace
RUN git clone https://github.com/superboySB/efficientvit.git && pip list
RUN cd efficientvit && \
    pip install --upgrade pip wheel && \
    pip install git+https://github.com/superboySB/pytorch-image-models && \
    pip install --no-cache -r requirements.txt --constraint constraints.txt && mkdir -p assets/checkpoints/sam && cd assets/checkpoints/sam && \
    wget https://huggingface.co/han-cai/efficientvit-sam/resolve/main/l2.pt && \
    wget https://huggingface.co/han-cai/efficientvit-sam/resolve/main/xl1.pt
RUN cd /workspace/efficientvit/ && pip install "opencv-python-headless<4.3" && \
    python deployment/sam/onnx/export_encoder.py --model l2 --weight_url assets/checkpoints/sam/l2.pt --output assets/export_models/sam/onnx/l2_encoder.onnx && \ 
    python deployment/sam/onnx/export_decoder.py --model l2 --weight_url assets/checkpoints/sam/l2.pt --output assets/export_models/sam/onnx/l2_decoder.onnx --return-single-mask && \
    python deployment/sam/onnx/export_encoder.py --model xl1 --weight_url assets/checkpoints/sam/xl1.pt --output assets/export_models/sam/onnx/xl1_encoder.onnx && \ 
    python deployment/sam/onnx/export_decoder.py --model xl1 --weight_url assets/checkpoints/sam/xl1.pt --output assets/export_models/sam/onnx/xl1_decoder.onnx --return-single-mask && \
    python deployment/sam/onnx/inference.py --model l2 --encoder_model assets/export_models/sam/onnx/l2_encoder.onnx --decoder_model assets/export_models/sam/onnx/l2_decoder.onnx --mode point

# Siammask
WORKDIR /workspace
RUN git clone https://github.com/superboySB/SiamMask && cd SiamMask && \
    pip install --no-cache onnxoptimizer loguru && bash make.sh

# YOLO-World
WORKDIR /usr/src/ultralytics
ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.Unicode.ttf \
    /root/.config/Ultralytics/
RUN git clone https://github.com/superboySB/ultralytics /usr/src/ultralytics
RUN grep -v "opencv-python" pyproject.toml > temp.toml && mv temp.toml pyproject.toml
RUN pip install --no-cache tqdm matplotlib pyyaml psutil pandas onnx "numpy==1.23" "opencv-python-headless<4.3" && pip install --no-cache -e .
ENV OMP_NUM_THREADS=1
WORKDIR /workspace
RUN git clone https://github.com/superboySB/YOLOv8-TensorRT.git && cd YOLOv8-TensorRT && \
    wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-world.pt && \
    wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-world.pt && \
    wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt
RUN cd YOLOv8-TensorRT && python test_yoloworld.py

# NanoTrack
WORKDIR /workspace
RUN git clone https://github.com/superboySB/SiamTrackers && cd SiamTrackers/NanoTrack && python setup.py build_ext --inplace
RUN cd SiamTrackers/NanoTrack && pip install yacs


WORKDIR /workspace
RUN rm -rf /var/lib/apt/lists/* && apt-get clean
RUN chmod +x /opt/ros/foxy/install/setup.bash
CMD ["/bin/bash"]
 
