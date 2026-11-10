cmake_minimum_required(VERSION 3.16)
project(hybrid_communication_racer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/HybridDroneState.msg"
  "msg/HybridGlobalAssignment.msg"
)

add_library(hybrid_assignment SHARED src/assignment.cpp)
target_include_directories(hybrid_assignment
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

install(TARGETS hybrid_assignment
  EXPORT export_hybrid_assignment
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_hybrid_assignment test/test_assignment.cpp)
  target_link_libraries(test_hybrid_assignment hybrid_assignment)
endif()

ament_export_targets(export_hybrid_assignment HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
