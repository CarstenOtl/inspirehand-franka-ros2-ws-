cmake_minimum_required(VERSION 3.5)
project(franka_repeatability)

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(control_msgs REQUIRED)
find_package(controller_interface REQUIRED)
find_package(franka_msgs REQUIRED)
find_package(franka_semantic_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME} SHARED
    src/cartesian_target_impedance_ik_controller.cpp
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${EIGEN3_INCLUDE_DIRS}
)

ament_target_dependencies(${PROJECT_NAME}
    control_msgs
    controller_interface
    franka_msgs
    franka_semantic_components
    geometry_msgs
    hardware_interface
    moveit_msgs
    pluginlib
    rclcpp
    rclcpp_lifecycle
    realtime_tools
    sensor_msgs
    std_msgs
    trajectory_msgs
)

pluginlib_export_plugin_description_file(controller_interface franka_repeatability.xml)

install(TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
    scripts/run_repeatability.py
    scripts/analyze_repeatability.py
    scripts/capture_pose.py
    scripts/goto_pose.py
    scripts/preview_poses.py
    scripts/preflight.py
    scripts/impedance_check.py
    scripts/plot_repeatability.py
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch rviz DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
    find_package(ament_cmake_gmock REQUIRED)
    find_package(controller_manager REQUIRED)
    find_package(ros2_control_test_assets REQUIRED)

    ament_add_gmock(${PROJECT_NAME}_test_load test/test_load_controller.cpp)
    target_include_directories(${PROJECT_NAME}_test_load PRIVATE include)
    ament_target_dependencies(${PROJECT_NAME}_test_load
        controller_manager
        ros2_control_test_assets
    )
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(controller_interface franka_semantic_components rclcpp_lifecycle)

ament_package()
