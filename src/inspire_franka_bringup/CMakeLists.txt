cmake_minimum_required(VERSION 3.8)
project(inspire_franka_bringup)

find_package(ament_cmake REQUIRED)
# Out-of-tree install prefix; the dev image sets CMAKE_PREFIX_PATH=/opt/libfranka.
find_package(Franka REQUIRED)

# No ROS dependencies on purpose: this must stay runnable to answer "is the FCI
# reachable" even when the workspace does not build.
add_executable(fci_check tools/fci_check.cpp)
target_link_libraries(fci_check Franka::Franka)

install(TARGETS fci_check DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY launch rviz
  DESTINATION share/${PROJECT_NAME}
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
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
