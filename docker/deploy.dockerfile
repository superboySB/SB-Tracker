# TODO： Need to be upgrade to r36.2 to support the latest tensorRT
FROM dustynv/ros:humble-pytorch-l4t-r35.3.1

# Please contact with me if you have problems
LABEL maintainer="Zipeng Dai <daizipeng@bit.edu.cn>"

# System Requirements
ENV SHELL /bin/bash
WORKDIR /tmp
SHELL ["/bin/bash", "-c"] 
RUN apt-get update && \
    apt-get install -y --no-install-recommends locales git tmux gedit vim openmpi-bin openmpi-common libopenmpi-dev libgl1-mesa-glx \
    gcc zip curl htop libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0 ffmpeg

# ROS
# RUN locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
# ENV LANG=en_US.UTF-8
# ENV PYTHONIOENCODING=utf-8
# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
# COPY docker/ros2_build.sh ros2_build.sh
# RUN . ros2_build.sh
# ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# COPY docker/ros_entrypoint.sh /ros_entrypoint.sh

# Install opencv, px4 autopilot for SITL implementation
# We do not need offboard api, qgc, px4_msgs and micro_ros_agent in board.
# WORKDIR /workspace
# RUN git clone https://github.com/opencv/opencv.git && cd opencv && git checkout 4.3.0 && cd .. && \
#     git clone https://github.com/opencv/opencv_contrib.git && cd opencv_contrib && git checkout 4.3.0
# RUN apt-get purge libopencv* -y && apt-get update && pip uninstall opencv-python && \
#     apt-get install -y --no-install-recommends build-essential cmake libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
# RUN cd /workspace/opencv && mkdir build && cd build && cmake -D CMAKE_BUILD_TYPE=RELEASE \
#       -D CMAKE_INSTALL_PREFIX=/usr/local \
#       -D OPENCV_EXTRA_MODULES_PATH=/workspace/opencv_contrib/modules \
#       -D BUILD_EXAMPLES=ON .. && make -j$(nproc) && make install
# WORKDIR /workspace
# RUN git clone https://github.com/PX4/PX4-Autopilot.git -b v1.14.1 --recursive && bash PX4-Autopilot/Tools/setup/ubuntu.sh --no-nuttx
# RUN cd PX4-Autopilot && make clean && DONT_RUN=1 make px4_sitl_default gazebo-classic
# WORKDIR /workspace
# RUN . $ROS_ROOT/install/setup.sh && git clone https://github.com/Jaeyoung-Lim/px4-offboard.git && cd px4-offboard && colcon build && cd ..
# RUN wget https://d176tv9ibo4jno.cloudfront.net/latest/QGroundControl.AppImage && chmod +x ./QGroundControl.AppImage 
# RUN . $ROS_ROOT/install/setup.sh && mkdir -p /workspace/px4_ros_com_ws/src && cd /workspace/px4_ros_com_ws/src && \
#     git clone https://github.com/PX4/px4_msgs.git && cd .. && colcon build
# WORKDIR /workspace
# RUN . $ROS_ROOT/install/setup.sh && mkdir microros_ws && cd microros_ws && \
#     git clone -b $ROS_DISTRO https://github.com/micro-ROS/micro_ros_setup.git src/micro_ros_setup && \
#     rosdep update && rosdep install --from-paths src --ignore-src -y && colcon build && . install/local_setup.sh && \
#     ros2 run micro_ros_setup create_agent_ws.sh && ros2 run micro_ros_setup build_agent.sh


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
# RUN cd /workspace/efficientvit/ && pip install "opencv-python-headless<4.3" && \
#     python deployment/sam/onnx/export_encoder.py --model l2 --weight_url assets/checkpoints/sam/l2.pt --output assets/export_models/sam/onnx/l2_encoder.onnx && \ 
#     python deployment/sam/onnx/export_decoder.py --model l2 --weight_url assets/checkpoints/sam/l2.pt --output assets/export_models/sam/onnx/l2_decoder.onnx --return-single-mask && \
#     python deployment/sam/onnx/export_encoder.py --model xl1 --weight_url assets/checkpoints/sam/xl1.pt --output assets/export_models/sam/onnx/xl1_encoder.onnx && \ 
#     python deployment/sam/onnx/export_decoder.py --model xl1 --weight_url assets/checkpoints/sam/xl1.pt --output assets/export_models/sam/onnx/xl1_decoder.onnx --return-single-mask && \
#     python deployment/sam/onnx/inference.py --model l2 --encoder_model assets/export_models/sam/onnx/l2_encoder.onnx --decoder_model assets/export_models/sam/onnx/l2_decoder.onnx --mode point

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

WORKDIR /workspace
RUN git clone https://github.com/superboySB/SiamTrackers && cd SiamTrackers/NanoTrack && python setup.py build_ext --inplace
RUN cd SiamTrackers/NanoTrack && pip install yacs

WORKDIR /workspace
RUN rm -rf /var/lib/apt/lists/* && apt-get clean
RUN chmod +x /opt/ros/humble/install/setup.bash
# ENTRYPOINT ["/opt/ros/humble/install/setup.bash"]
CMD ["/bin/bash"]
 
