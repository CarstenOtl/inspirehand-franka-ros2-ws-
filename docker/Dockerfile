# Dev image for inspire_franka: ROS 2 Jazzy, the MuJoCo simulation stack, and
# the real-arm Franka stack, in one container that also talks to the Inspire
# hand over a USB-RS485 adapter.
#
# Deliberately close to stock. Unlike a Humble build of the same workspace,
# almost everything here comes from apt: mujoco_ros2_control was released for
# Jazzy (0.1.1, with mujoco_vendor 0.1.0) and needs no source build, and
# ros2_control is new enough that franka_ros2 v3.5.x does not need its patched
# fork either. The only thing built from source is libfranka, which has no ROS
# release at all.
ARG ROS_DISTRO=jazzy
FROM ros:${ROS_DISTRO}-ros-base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Optional: corporate root CA(s), e.g. for TLS inspection on a VPN. Drop *.crt
# files into docker/certs/ before building if your network needs it; none are
# committed. The wildcard-plus-mkdir dance is because COPY fails on an empty
# glob but not on an empty directory.
COPY certs/ /usr/local/share/ca-certificates/extra/
RUN update-ca-certificates

# --- Base tooling ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash-completion \
        build-essential \
        cmake \
        curl \
        gdb \
        git \
        iputils-ping \
        nano \
        python3-colcon-argcomplete \
        python3-colcon-common-extensions \
        python3-pip \
        python3-rosdep \
        python3-vcstool \
        vim \
    && rm -rf /var/lib/apt/lists/*

# --- ros2_control, descriptions, visualisation -------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-${ROS_DISTRO}-ros2-control \
        ros-${ROS_DISTRO}-ros2-controllers \
        ros-${ROS_DISTRO}-ros2controlcli \
        ros-${ROS_DISTRO}-controller-manager \
        ros-${ROS_DISTRO}-joint-state-broadcaster \
        ros-${ROS_DISTRO}-joint-trajectory-controller \
        ros-${ROS_DISTRO}-forward-command-controller \
        ros-${ROS_DISTRO}-joint-state-publisher \
        ros-${ROS_DISTRO}-joint-state-publisher-gui \
        ros-${ROS_DISTRO}-robot-state-publisher \
        ros-${ROS_DISTRO}-rviz2 \
        ros-${ROS_DISTRO}-xacro \
        ros-${ROS_DISTRO}-hardware-interface-testing \
        ros-${ROS_DISTRO}-ros2-control-test-assets \
        ros-${ROS_DISTRO}-ament-cmake-clang-format \
        ros-${ROS_DISTRO}-ament-cmake-clang-tidy \
    && rm -rf /var/lib/apt/lists/*

# --- MuJoCo ------------------------------------------------------------------
#
# mujoco_ros2_control is the ros2_control hardware plugin that owns a MuJoCo
# Simulate instance, the way gz_ros2_control owns a Gazebo one. It ships the
# `ros2_control_node` executable the sim launch runs - note that is
# mujoco_ros2_control's, not controller_manager's; using the latter would load
# the plugin with no simulator behind it.
#
# mujoco_vendor 0.1.0 pins MuJoCo 3.12.0. The committed MJCFs in
# inspire_franka_sim were generated with exactly that version.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-${ROS_DISTRO}-mujoco-ros2-control \
        ros-${ROS_DISTRO}-mujoco-vendor \
    && rm -rf /var/lib/apt/lists/*

# --- libfranka ---------------------------------------------------------------
#
# The one thing with no ROS release, so it is built here into its own prefix
# rather than into the workspace. Its nested libfranka-common submodule is
# initialized by the recursive clone below.
#
# 0.20.5 is what franka_ros2 v3.5.3's own dependency.repos pins. Bump this and
# the src/franka_ros2 submodule commit together: franka_hardware does a
# versioned find_package, so a too-old library fails at build time - but a
# too-new one does not, and the real constraint on hardware is the arm's own
# system version. Check Franka's compatibility matrix before deviating.
#
# pinocchio is a libfranka dependency as of 0.15 (its rigid-body model is built
# on it), and franka_example_controllers needs it too.
ARG LIBFRANKA_REF=0.20.5
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpoco-dev \
        libeigen3-dev \
        libtinyxml2-dev \
        libconsole-bridge-dev \
        ros-${ROS_DISTRO}-pinocchio \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --recurse-submodules --shallow-submodules --depth 1 \
        --branch ${LIBFRANKA_REF} https://github.com/frankarobotics/libfranka.git \
        /tmp/libfranka \
    && source /opt/ros/${ROS_DISTRO}/setup.bash \
    && cmake -S /tmp/libfranka -B /tmp/libfranka/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DCMAKE_INSTALL_PREFIX=/opt/libfranka \
    && cmake --build /tmp/libfranka/build -j"$(nproc)" --target install \
    && rm -rf /tmp/libfranka \
    && echo /opt/libfranka/lib > /etc/ld.so.conf.d/libfranka.conf \
    && ldconfig

# --- franka_ros2 build dependencies ------------------------------------------
#
# `rosdep install` under-reports one of these: franka_example_controllers'
# CMakeLists.txt does find_package(moveit_core) and find_package(moveit_msgs)
# while declaring neither in its package.xml. Note that is moveit_core only -
# the full MoveIt stack is NOT installed, and franka_fr3_moveit_config is one of
# the packages docker/entrypoint.sh marks COLCON_IGNORE.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-${ROS_DISTRO}-moveit-core \
        ros-${ROS_DISTRO}-moveit-msgs \
        ros-${ROS_DISTRO}-joy \
        ros-${ROS_DISTRO}-teleop-twist-joy \
        ros-${ROS_DISTRO}-teleop-twist-keyboard \
        ros-${ROS_DISTRO}-control-toolbox \
        ros-${ROS_DISTRO}-backward-ros \
        ros-${ROS_DISTRO}-generate-parameter-library \
        python3-requests \
    && rm -rf /var/lib/apt/lists/*

# --- Inspire hand ------------------------------------------------------------
#
# The driver's only runtime dependency is pyserial. It talks Modbus RTU over a
# USB-RS485 adapter, which appears as /dev/ttyUSB*; docker-compose.yml maps /dev
# in so the container can open it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-serial \
        python3-pytest \
    && rm -rf /var/lib/apt/lists/*

# --- RealSense D415 ----------------------------------------------------------
#
# The ROS wrapper is the src/realsense_d415 submodule.
# Jazzy publishes librealsense through the ROS repository, so the SDK and the
# wrapper's non-core ROS dependencies can stay ordinary apt dependencies here.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-${ROS_DISTRO}-librealsense2 \
        ros-${ROS_DISTRO}-cv-bridge \
        ros-${ROS_DISTRO}-diagnostic-updater \
        ros-${ROS_DISTRO}-image-transport \
        ros-${ROS_DISTRO}-rqt-gui-py \
        ros-${ROS_DISTRO}-rqt-image-view \
        ros-${ROS_DISTRO}-apriltag \
        python3-matplotlib \
        python3-opencv \
        python3-scipy \
        psmisc \
        python3-yaml \
        usbutils \
        v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Python simulation and asset-regeneration tooling. The MuJoCo bindings are
# used directly by the trajectory-replay app and by the MJCF tests/scripts;
# trimesh converts the upstream glTF visual meshes to STL.
#
# mujoco is pinned to mujoco_vendor's version so the MJCF is generated by the
# same library that will load it. --break-system-packages because Jazzy's base
# image is Ubuntu 24.04, whose python is PEP 668 externally-managed; a venv
# would be cleaner but would not see the ROS python packages xacro needs.
ARG MUJOCO_VERSION=3.12.0
RUN pip3 install --no-cache-dir --break-system-packages \
        mujoco==${MUJOCO_VERSION} \
        trimesh \
    && python3 -c "import cv2, mujoco"

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# CMAKE_PREFIX_PATH so find_package(Franka) resolves the out-of-tree install
# prefix; the matching runtime path is handled by /etc/ld.so.conf.d/libfranka.conf
# above rather than LD_LIBRARY_PATH. ROS's setup.bash prepends to
# CMAKE_PREFIX_PATH, so this survives being sourced.
ENV ROS_DOMAIN_ID=42 \
    RCUTILS_COLORIZED_OUTPUT=1 \
    CMAKE_PREFIX_PATH=/opt/libfranka

WORKDIR /root/develop_ws
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["tail", "-F", "/dev/null"]
