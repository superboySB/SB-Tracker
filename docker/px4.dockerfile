FROM nvcr.io/nvidia/isaac/ros:aarch64-ros2_humble_b7e1ed6c02a6fa3c1c7392479291c035

# Please contact with me if you have problems
LABEL maintainer="Zipeng Dai <daizipeng@bit.edu.cn>"
ARG DEBIAN_FRONTEND=noninteractive

# Random tools
WORKDIR /workspace
RUN git clone https://github.com/opencv/opencv.git && cd opencv && git checkout 4.3.0 && cd .. && \
    git clone https://github.com/opencv/opencv_contrib.git && cd opencv_contrib && git checkout 4.3.0
RUN apt-get purge libopencv* -y && apt-get update && pip uninstall opencv-python && \
    apt-get install -y --no-install-recommends build-essential cmake libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    vim gedit tmux
RUN cd /workspace/opencv && mkdir build && cd build && cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=/workspace/opencv_contrib/modules \
      -D BUILD_EXAMPLES=ON .. && make -j$(nproc) && make install

# PX4 and relevant infra
WORKDIR /workspace
RUN git clone https://github.com/PX4/PX4-Autopilot.git -b v1.14.1 --recursive && bash PX4-Autopilot/Tools/setup/ubuntu.sh --no-nuttx
RUN cd PX4-Autopilot && make clean && DONT_RUN=1 make px4_sitl_default none-iris
WORKDIR /workspace
RUN . /opt/ros/humble/setup.sh && git clone https://github.com/Jaeyoung-Lim/px4-offboard.git && cd px4-offboard && colcon build && cd ..
RUN wget https://d176tv9ibo4jno.cloudfront.net/latest/QGroundControl.AppImage && chmod +x ./QGroundControl.AppImage 
RUN . /opt/ros/humble/setup.sh && mkdir -p /workspace/px4_ros_com_ws/src && cd /workspace/px4_ros_com_ws/src && \
    git clone https://github.com/PX4/px4_msgs.git && cd .. && colcon build
WORKDIR /workspace
RUN . /opt/ros/humble/setup.sh && mkdir microros_ws && cd microros_ws && \
    git clone -b $ROS_DISTRO https://github.com/micro-ROS/micro_ros_setup.git src/micro_ros_setup && \
    rosdep update && rosdep install --from-paths src --ignore-src -y && colcon build && . install/local_setup.sh && \
    ros2 run micro_ros_setup create_agent_ws.sh && ros2 run micro_ros_setup build_agent.sh


USER root
RUN usermod -a -G dialout root

SHELL [ "/bin/bash", "-c" ]