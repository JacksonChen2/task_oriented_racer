cmake_minimum_required(VERSION 3.16)
project(racer_isaac_adapter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(racer_fidelity_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenMP REQUIRED)

set(_ISAAC_SIM_ROOT_DEFAULT "")
if(DEFINED ENV{ISAAC_SIM_ROOT} AND NOT "$ENV{ISAAC_SIM_ROOT}" STREQUAL "")
  set(_ISAAC_SIM_ROOT_DEFAULT "$ENV{ISAAC_SIM_ROOT}")
endif()
set(ISAAC_SIM_ROOT "${_ISAAC_SIM_ROOT_DEFAULT}" CACHE PATH
  "Isaac Sim root used to build the Python 3.11 control extension")
set(ISAAC_PYTHON_INCLUDE "${ISAAC_SIM_ROOT}/kit/python/include/python3.11")
set(ISAAC_PYBIND11_INCLUDE
  "${ISAAC_SIM_ROOT}/exts/omni.isaac.ml_archive/pip_prebundle/torch/include")
if(NOT EXISTS "${ISAAC_PYTHON_INCLUDE}/Python.h")
  message(FATAL_ERROR "Isaac Python headers not found under ${ISAAC_SIM_ROOT}")
endif()
if(NOT EXISTS "${ISAAC_PYBIND11_INCLUDE}/pybind11/pybind11.h")
  message(FATAL_ERROR "Isaac pybind11 headers not found under ${ISAAC_SIM_ROOT}")
endif()
execute_process(
  COMMAND "${ISAAC_SIM_ROOT}/python.sh" -c
    "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
  OUTPUT_VARIABLE ISAAC_PYTHON_EXTENSION_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE ISAAC_PYTHON_QUERY_RESULT)
if(NOT ISAAC_PYTHON_QUERY_RESULT EQUAL 0 OR
   ISAAC_PYTHON_EXTENSION_SUFFIX STREQUAL "")
  message(FATAL_ERROR "Could not query the Isaac Python extension suffix")
endif()

add_executable(position_command_adapter src/position_command_adapter.cpp)
ament_target_dependencies(position_command_adapter
  geometry_msgs nav_msgs racer_fidelity_msgs rclcpp std_msgs)

add_executable(exploration_trigger src/exploration_trigger.cpp)
ament_target_dependencies(exploration_trigger
  geometry_msgs nav_msgs rclcpp sensor_msgs)

add_library(racer_control_batch_cpp MODULE src/control_batch_pybind.cpp)
target_include_directories(racer_control_batch_cpp PRIVATE
  "${ISAAC_PYTHON_INCLUDE}"
  "${ISAAC_PYBIND11_INCLUDE}")
target_link_libraries(racer_control_batch_cpp PRIVATE OpenMP::OpenMP_CXX)
set_target_properties(racer_control_batch_cpp PROPERTIES
  PREFIX ""
  SUFFIX "${ISAAC_PYTHON_EXTENSION_SUFFIX}"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/isaac_sim")

install(TARGETS position_command_adapter exploration_trigger
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(TARGETS racer_control_batch_cpp
  LIBRARY DESTINATION share/${PROJECT_NAME}/isaac_sim)
install(DIRECTORY launch config scripts isaac_sim
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  enable_testing()
  add_test(
    NAME control_batch_correctness
    COMMAND "${ISAAC_SIM_ROOT}/python.sh"
      "${CMAKE_CURRENT_SOURCE_DIR}/test/test_control_batch.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/isaac_sim")
  set_tests_properties(control_batch_correctness PROPERTIES
    ENVIRONMENT
      "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/isaac_sim:${CMAKE_CURRENT_SOURCE_DIR}/isaac_sim")
endif()

ament_package()
