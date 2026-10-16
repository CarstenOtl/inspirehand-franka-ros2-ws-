cmake_minimum_required(VERSION 3.5)
project(franka_forge_cartesian_impedance)

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(control_msgs REQUIRED)
find_package(controller_interface REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(franka_forge_cartesian_impedance_msgs REQUIRED)
find_package(franka_msgs REQUIRED)
find_package(franka_semantic_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)

add_library(${PROJECT_NAME} SHARED
    src/cartesian_impedance_controller.cpp
    src/arm_mass_model.cpp
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} pinocchio::pinocchio)

ament_target_dependencies(${PROJECT_NAME}
    control_msgs
    controller_interface
    diagnostic_msgs
    franka_forge_cartesian_impedance_msgs
    franka_msgs
    franka_semantic_components
    geometry_msgs
    hardware_interface
    pluginlib
    rclcpp
    rcl_interfaces
    rclcpp_lifecycle
    realtime_tools
    std_msgs
)

pluginlib_export_plugin_description_file(controller_interface franka_forge_cartesian_impedance.xml)

install(TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
    find_package(ament_cmake_gmock REQUIRED)
    find_package(controller_manager REQUIRED)
    find_package(ros2_control_test_assets REQUIRED)

    ament_add_gmock(${PROJECT_NAME}_test_cartesian_impedance test/test_cartesian_impedance.cpp)
    target_include_directories(${PROJECT_NAME}_test_cartesian_impedance PRIVATE
        include ${EIGEN3_INCLUDE_DIRS})

    ament_add_gmock(${PROJECT_NAME}_test_arm_mass_model test/test_arm_mass_model.cpp)
    target_include_directories(${PROJECT_NAME}_test_arm_mass_model PRIVATE
        include ${EIGEN3_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME}_test_arm_mass_model ${PROJECT_NAME})

    ament_add_gmock(${PROJECT_NAME}_test_fr3_kinematics test/test_fr3_kinematics.cpp)
    target_include_directories(${PROJECT_NAME}_test_fr3_kinematics PRIVATE
        include ${EIGEN3_INCLUDE_DIRS})

    ament_add_gmock(${PROJECT_NAME}_test_load test/test_load_controller.cpp)
    target_include_directories(${PROJECT_NAME}_test_load PRIVATE include)
    target_link_libraries(${PROJECT_NAME}_test_load ${PROJECT_NAME})
    ament_target_dependencies(${PROJECT_NAME}_test_load
        controller_manager
        ros2_control_test_assets
    )
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(controller_interface franka_forge_cartesian_impedance_msgs
    franka_semantic_components geometry_msgs rcl_interfaces rclcpp_lifecycle)

ament_package()
