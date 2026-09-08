cmake_minimum_required(VERSION 3.5)
project(franka_mujoco_hardware)

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

# MuJoCo: either a release tarball unpacked at MUJOCO_DIR (include/ + lib/), or the headers and
# shared library that ship inside the `mujoco` Python wheel (pip install mujoco).
set(MUJOCO_DIR "" CACHE PATH "Root of a MuJoCo release (contains include/ and lib/)")
if(NOT MUJOCO_DIR AND DEFINED ENV{MUJOCO_DIR})
    set(MUJOCO_DIR "$ENV{MUJOCO_DIR}")
endif()
if(MUJOCO_DIR)
    set(MUJOCO_INCLUDE_DIR "${MUJOCO_DIR}/include")
    find_library(MUJOCO_LIBRARY NAMES mujoco PATHS "${MUJOCO_DIR}/lib" NO_DEFAULT_PATH)
else()
    execute_process(
        COMMAND python3 -c "import mujoco, os; print(os.path.dirname(mujoco.__file__))"
        OUTPUT_VARIABLE MUJOCO_PY_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE MUJOCO_PY_RESULT ERROR_QUIET)
    if(MUJOCO_PY_RESULT EQUAL 0 AND EXISTS "${MUJOCO_PY_DIR}/include/mujoco/mujoco.h")
        set(MUJOCO_INCLUDE_DIR "${MUJOCO_PY_DIR}/include")
        file(GLOB MUJOCO_LIBRARY "${MUJOCO_PY_DIR}/libmujoco.so*")
        list(SORT MUJOCO_LIBRARY)
        list(GET MUJOCO_LIBRARY -1 MUJOCO_LIBRARY)
    endif()
endif()
if(NOT MUJOCO_LIBRARY OR NOT EXISTS "${MUJOCO_INCLUDE_DIR}/mujoco/mujoco.h")
    message(FATAL_ERROR
        "MuJoCo not found. Either `pip install mujoco` (the wheel carries headers and the shared "
        "library) or unpack a MuJoCo release and pass -DMUJOCO_DIR=/path/to/mujoco-3.x.y")
endif()
message(STATUS "MuJoCo library: ${MUJOCO_LIBRARY}")
get_filename_component(MUJOCO_LIBRARY_DIR "${MUJOCO_LIBRARY}" DIRECTORY)

add_library(${PROJECT_NAME} SHARED src/mujoco_system.cpp)
target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${MUJOCO_INCLUDE_DIR}
)
target_link_libraries(${PROJECT_NAME} ${MUJOCO_LIBRARY})
set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_RPATH "${MUJOCO_LIBRARY_DIR}"
    INSTALL_RPATH "${MUJOCO_LIBRARY_DIR}"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
ament_target_dependencies(${PROJECT_NAME} hardware_interface pluginlib rclcpp rclcpp_lifecycle)

pluginlib_export_plugin_description_file(hardware_interface franka_mujoco_hardware.xml)

install(TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY mujoco urdf DESTINATION share/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(hardware_interface pluginlib rclcpp rclcpp_lifecycle)
ament_package()
