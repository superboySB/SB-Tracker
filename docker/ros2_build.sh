#!/usr/bin/env bash
# this script builds a ROS2 distribution from source
# ROS_DISTRO, ROS_ROOT, ROS_PACKAGE environment variables should be set

echo "ROS2 builder => ROS_DISTRO=$ROS_DISTRO ROS_PACKAGE=$ROS_PACKAGE ROS_ROOT=$ROS_ROOT"

set -e
#set -x

# add the ROS deb repo to the apt sources list
apt-get update
apt-get install -y --no-install-recommends \
    curl \
    wget \
    gnupg2 \
    lsb-release \
    ca-certificates

curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
apt-get update

# install development packages
apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libbullet-dev \
    libpython3-dev \
    python3-colcon-common-extensions \
    python3-flake8 \
    python3-pip \
    python3-numpy \
    python3-pytest-cov \
    python3-rosdep \
    python3-setuptools \
    python3-vcstool \
    python3-rosinstall-generator \
    libasio-dev \
    libtinyxml2-dev \
    libcunit1-dev

# install some pip packages needed for testing
pip3 install --upgrade --no-cache-dir \
    argcomplete \
    flake8-blind-except \
    flake8-builtins \
    flake8-class-newline \
    flake8-comprehensions \
    flake8-deprecated \
    flake8-docstrings \
    flake8-import-order \
    flake8-quotes \
    pytest-repeat \
    pytest-rerunfailures \
    pytest

# upgrade cmake
python3 -m pip install --upgrade pip
pip3 install --no-cache-dir scikit-build
pip3 install --upgrade --no-cache-dir --verbose cmake
cmake --version
which cmake

# create the ROS_ROOT directory
mkdir -p ${ROS_ROOT}/src
cd ${ROS_ROOT}

# download ROS sources
rosinstall_generator --deps --rosdistro ${ROS_DISTRO} ${ROS_PACKAGE} \
	launch_xml \
	launch_yaml \
	launch_testing \
	launch_testing_ament_cmake \
	demo_nodes_cpp \
	demo_nodes_py \
	example_interfaces \
	camera_calibration_parsers \
	camera_info_manager \
	cv_bridge \
	v4l2_camera \
	vision_opencv \
	vision_msgs \
	image_geometry \
	image_pipeline \
	image_transport \
	compressed_image_transport \
	compressed_depth_image_transport \
> ros2.${ROS_DISTRO}.${ROS_PACKAGE}.rosinstall
cat ros2.${ROS_DISTRO}.${ROS_PACKAGE}.rosinstall
vcs import src < ros2.${ROS_DISTRO}.${ROS_PACKAGE}.rosinstall

# skip installation of RTI Connext DDS due to license agreement requirement
SKIP_KEYS="rti-connext-dds-6.0.1"

echo "--skip-keys $SKIP_KEYS"

# install dependencies using rosdep
rosdep init
rosdep update
rosdep install -y \
    --ignore-src \
    --from-paths src \
    --rosdistro ${ROS_DISTRO} \
    --skip-keys "$SKIP_KEYS"

# build it all
colcon build \
    --merge-install \
    --cmake-args -DCMAKE_BUILD_TYPE=Release 

# cleanup
rm -rf ${ROS_ROOT}/src
rm -rf ${ROS_ROOT}/logs
rm -rf ${ROS_ROOT}/build
rm ${ROS_ROOT}/*.rosinstall

# cleanup apt
rm -rf /var/lib/apt/lists/*
apt-get clean
