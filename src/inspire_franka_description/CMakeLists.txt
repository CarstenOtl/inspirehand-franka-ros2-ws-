cmake_minimum_required(VERSION 3.8)
project(inspire_franka_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf meshes
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
  find_package(ament_cmake_pytest REQUIRED)
  ament_lint_auto_find_test_dependencies()
  ament_add_pytest_test(description test/test_description.py)
endif()

ament_package()
