cmake_minimum_required(VERSION 3.8)
project(inspire_franka_sim)

find_package(ament_cmake REQUIRED)
find_package(mujoco_ros2_control_plugins REQUIRED)
find_package(mujoco_vendor REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

# The threaded nut in inspire_franka_policy_scene.xml is not part of the robot,
# so no ros2_control interface reaches it. This plugin publishes its state and
# clamps the thread for the release/return transition; see the header.
add_library(${PROJECT_NAME}_thread_pair_plugin SHARED src/thread_pair_plugin.cpp)
target_compile_features(${PROJECT_NAME}_thread_pair_plugin PUBLIC cxx_std_17)
target_include_directories(${PROJECT_NAME}_thread_pair_plugin PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(${PROJECT_NAME}_thread_pair_plugin
  mujoco_ros2_control_plugins
  pluginlib
  rclcpp
  realtime_tools
  sensor_msgs
  std_msgs
  std_srvs
)
# NOT ament_target_dependencies(mujoco_vendor): its exported target carries a
# RELATIVE library path ("opt/mujoco_vendor/lib/libmujoco.so"), which the linker
# reads as -lopt/... and cannot find. mujoco_vendor-extras.cmake defines a
# correct imported target, so link that.
target_link_libraries(${PROJECT_NAME}_thread_pair_plugin mujoco::mujoco)
pluginlib_export_plugin_description_file(
  mujoco_ros2_control_plugins thread_pair_plugin.xml)

install(TARGETS ${PROJECT_NAME}_thread_pair_plugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

install(
  DIRECTORY mjcf config launch rviz
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/make_hand_mjcf.py
    scripts/make_training_hand_mjcf.py
    scripts/inspire_hand_sim_bridge.py
    scripts/mujoco_vis_node.py
    scripts/policy_camera_relay.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # xmllint and lint_cmake are kept - they check the URDF/xacro and CMake files
  # for real mistakes, and they pass. The three style linters are switched off
  # deliberately:
  #
  #   copyright - this repo carries no per-file copyright headers.
  #   flake8    - ament's config sets flake8-quotes to prefer single quotes.
  #               This workspace, like the others on this machine, is written
  #               with double quotes; conforming would be ~500 edits to say
  #               nothing about correctness.
  #   pep257    - its docstring rules (one-line summary, imperative mood) fight
  #               the longer explanatory docstrings used throughout here.
  #
  # Turn any of them back on by deleting its line; the linter is installed and
  # will just run.
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_flake8_FOUND TRUE)
  set(ament_cmake_pep257_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_pytest REQUIRED)
  ament_lint_auto_find_test_dependencies()
  ament_add_pytest_test(mjcf test/test_mjcf.py)
  ament_add_pytest_test(mujoco_vis_node test/test_mujoco_vis_node.py)
endif()

ament_package()
