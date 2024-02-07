FROM dustynv/nanosam:r35.3.1

# Please contact with me if you have problems
LABEL maintainer="Zipeng Dai <daizipeng@bit.edu.cn>"

# 0. System Requirements
ARG ROS_PACKAGE=ros_base
ARG ROS_VERSION=humble
ENV ROS_DISTRO=${ROS_VERSION}
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}
ENV ROS_PYTHON_VERSION=3

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

# set Python3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
    
# build ROS from source
COPY docker/ros2_build.sh ros2_build.sh
RUN . ros2_build.sh

# Set the default DDS middleware to cyclonedds
# https://github.com/ros2/rclcpp/issues/1335
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# commands will be appended/run by the entrypoint which sources the ROS environment
COPY docker/ros_entrypoint.sh /ros_entrypoint.sh

# Install opencv, px4 autopilot for SITL implementation
# We do not need offboard api, qgc, px4_msgs and micro_ros_agent in board.
WORKDIR /workspace
RUN git clone https://github.com/opencv/opencv.git && cd opencv && git checkout 4.3.0 && cd .. && \
    git clone https://github.com/opencv/opencv_contrib.git && cd opencv_contrib && git checkout 4.3.0
RUN apt-get purge libopencv* -y && apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
RUN cd /workspace/opencv && mkdir build && cd build && cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=/workspace/opencv_contrib/modules \
      -D BUILD_EXAMPLES=ON .. && make -j$(nproc) && make install
WORKDIR /workspace
RUN git clone https://github.com/PX4/PX4-Autopilot.git -b v1.14.1 --recursive && bash PX4-Autopilot/Tools/setup/ubuntu.sh --no-nuttx
RUN cd PX4-Autopilot && make clean && DONT_RUN=1 make px4_sitl_default gazebo-classic
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
WORKDIR /workspace
RUN git clone https://github.com/triple-Mu/YOLOv8-TensorRT.git && cd YOLOv8-TensorRT && pip install --upgrade pip && \
    pip install -r requirements.txt && pip install ultralytics lapx && pip install opencv-python==4.8.0.74

WORKDIR /workspace
RUN rm -rf /var/lib/apt/lists/* && apt-get clean
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["/bin/bash"]






 
