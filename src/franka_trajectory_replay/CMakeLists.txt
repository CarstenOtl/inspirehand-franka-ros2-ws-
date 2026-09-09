cmake_minimum_required(VERSION 3.5)
project(franka_trajectory_replay)

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
find_package(diagnostic_msgs REQUIRED)
find_package(franka_msgs REQUIRED)
find_package(franka_semantic_components REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME} SHARED
    src/trajectory_replay_controller.cpp
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
    diagnostic_msgs
    franka_msgs
    franka_semantic_components
    hardware_interface
    pluginlib
    rclcpp
    rclcpp_lifecycle
    realtime_tools
    sensor_msgs
    std_msgs
    trajectory_msgs
)

pluginlib_export_plugin_description_file(controller_interface franka_trajectory_replay.xml)

install(TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
    scripts/prepare_trajectory.py
    scripts/make_test_trajectory.py
    scripts/simulate_trajectory.py
    scripts/replay_trajectory.py
    scripts/analyze_replay.py
    scripts/plot_replay.py
    scripts/preflight.py
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch trajectories DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
    find_package(ament_cmake_gmock REQUIRED)
    find_package(ament_cmake_pytest REQUIRED)
    find_package(controller_manager REQUIRED)
    find_package(ros2_control_test_assets REQUIRED)

    ament_add_gmock(${PROJECT_NAME}_test_joint_impedance test/test_joint_impedance.cpp)
    target_include_directories(${PROJECT_NAME}_test_joint_impedance PRIVATE
        include ${EIGEN3_INCLUDE_DIRS})

    ament_add_gmock(${PROJECT_NAME}_test_load test/test_load_controller.cpp)
    target_include_directories(${PROJECT_NAME}_test_load PRIVATE include)
    target_link_libraries(${PROJECT_NAME}_test_load ${PROJECT_NAME})
    ament_target_dependencies(${PROJECT_NAME}_test_load
        controller_manager
        ros2_control_test_assets
    )

    ament_add_pytest_test(${PROJECT_NAME}_pytest test/python
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(controller_interface franka_semantic_components rclcpp_lifecycle)

ament_package()
