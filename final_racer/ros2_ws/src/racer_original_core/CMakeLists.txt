cmake_minimum_required(VERSION 3.16)
project(racer_original_core LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
add_compile_options(-O3 -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters kdtree)
find_package(Armadillo REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(hybrid_communication_racer REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(racer_fidelity_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

set(NLOPT_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/install")
find_path(NLOPT_INCLUDE_DIR nlopt.hpp PATHS "${NLOPT_PREFIX}/include" NO_DEFAULT_PATH REQUIRED)
find_library(NLOPT_LIBRARY nlopt PATHS "${NLOPT_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)

set(UPSTREAM "${CMAKE_CURRENT_SOURCE_DIR}/upstream")
set(UPSTREAM_INCLUDE_DIRS
  "${UPSTREAM}/active_perception/include"
  "${UPSTREAM}/bspline/include"
  "${UPSTREAM}/bspline_opt/include"
  "${UPSTREAM}/exploration_manager/include"
  "${UPSTREAM}/path_searching/include"
  "${UPSTREAM}/plan_env/include"
  "${UPSTREAM}/plan_manage/include"
  "${UPSTREAM}/poly_traj/include"
  "${UPSTREAM}/traj_utils/include"
  "${UPSTREAM}/utils/backward"
  "${UPSTREAM}/utils/lkh_mtsp_solver/include"
)

set(RACER_ALGORITHM_SOURCES
  "${UPSTREAM}/plan_env/src/raycast.cpp"
  "${UPSTREAM}/plan_env/src/sdf_map.cpp"
  "${UPSTREAM}/plan_env/src/edt_environment.cpp"
  "${UPSTREAM}/plan_env/src/obj_predictor.cpp"
  "${UPSTREAM}/plan_env/src/multi_map_manager.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/port/map_ros_port.cpp"
  "${UPSTREAM}/path_searching/src/astar2.cpp"
  "${UPSTREAM}/path_searching/src/kinodynamic_astar.cpp"
  "${UPSTREAM}/path_searching/src/topo_prm.cpp"
  "${UPSTREAM}/active_perception/src/frontier_finder.cpp"
  "${UPSTREAM}/active_perception/src/graph_node.cpp"
  "${UPSTREAM}/active_perception/src/heading_planner.cpp"
  "${UPSTREAM}/active_perception/src/hgrid.cpp"
  "${UPSTREAM}/active_perception/src/perception_utils.cpp"
  "${UPSTREAM}/active_perception/src/traj_visibility.cpp"
  "${UPSTREAM}/active_perception/src/uniform_grid.cpp"
  "${UPSTREAM}/bspline/src/non_uniform_bspline.cpp"
  "${UPSTREAM}/bspline_opt/src/bspline_optimizer.cpp"
  "${UPSTREAM}/poly_traj/src/polynomial_traj.cpp"
  "${UPSTREAM}/plan_manage/src/planner_manager.cpp"
  "${UPSTREAM}/traj_utils/src/planning_visualization.cpp"
  "${UPSTREAM}/exploration_manager/src/fast_exploration_manager.cpp"
  "${UPSTREAM}/exploration_manager/src/fast_exploration_fsm.cpp"
)

add_library(racer_original_algorithm SHARED ${RACER_ALGORITHM_SOURCES})
target_include_directories(racer_original_algorithm
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${UPSTREAM_INCLUDE_DIRS}
  SYSTEM PUBLIC
    ${EIGEN3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${ARMADILLO_INCLUDE_DIRS}
    ${NLOPT_INCLUDE_DIR}
)
ament_target_dependencies(racer_original_algorithm
  builtin_interfaces geometry_msgs hybrid_communication_racer nav_msgs pcl_conversions racer_fidelity_msgs
  rclcpp sensor_msgs std_msgs visualization_msgs)
target_link_libraries(racer_original_algorithm
  hybrid_communication_racer::hybrid_assignment
  ${PCL_LIBRARIES} ${ARMADILLO_LIBRARIES} ${NLOPT_LIBRARY})
target_compile_definitions(racer_original_algorithm PRIVATE ${PCL_DEFINITIONS})
set_target_properties(racer_original_algorithm PROPERTIES
  BUILD_RPATH "${NLOPT_PREFIX}/lib"
  INSTALL_RPATH "${NLOPT_PREFIX}/lib")

file(GLOB LKH_MTSP_C_SOURCES "${UPSTREAM}/utils/lkh_mtsp_solver/src/*.c")
add_library(racer_original_lkh3 SHARED
  ${LKH_MTSP_C_SOURCES}
  "${UPSTREAM}/utils/lkh_mtsp_solver/src2/lkh3_interface.cpp")
target_include_directories(racer_original_lkh3 PUBLIC
  "${UPSTREAM}/utils/lkh_mtsp_solver/include")
target_compile_options(racer_original_lkh3 PRIVATE -fcommon)
target_link_libraries(racer_original_lkh3 m)

# The ROS1 server invokes the standalone LKH executable for problem 3
# (pairwise ACVRP).  Keeping a fresh process for every request also preserves
# LKH's original one-process/one-problem global-state lifecycle.
add_executable(racer_original_lkh_cli
  "${UPSTREAM}/utils/lkh_mtsp_solver/LKHmain.c")
target_include_directories(racer_original_lkh_cli PRIVATE
  "${UPSTREAM}/utils/lkh_mtsp_solver/include")
target_compile_options(racer_original_lkh_cli PRIVATE -fcommon)
target_link_libraries(racer_original_lkh_cli racer_original_lkh3 m)

add_executable(racer_original_exploration_node
  "${CMAKE_CURRENT_SOURCE_DIR}/port/exploration_node_port.cpp")
target_include_directories(racer_original_exploration_node PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include" ${UPSTREAM_INCLUDE_DIRS})
ament_target_dependencies(racer_original_exploration_node
  hybrid_communication_racer rclcpp racer_fidelity_msgs)
target_link_libraries(racer_original_exploration_node racer_original_algorithm)

# Separate algorithm variant: the original targets above are compiled without
# RACER_ORACLE_VARIANT and retain their source-faithful behavior and defaults.
add_library(racer_oracle_algorithm SHARED ${RACER_ALGORITHM_SOURCES})
target_include_directories(racer_oracle_algorithm
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${UPSTREAM_INCLUDE_DIRS}
  SYSTEM PUBLIC
    ${EIGEN3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${ARMADILLO_INCLUDE_DIRS}
    ${NLOPT_INCLUDE_DIR}
)
ament_target_dependencies(racer_oracle_algorithm
  builtin_interfaces geometry_msgs hybrid_communication_racer nav_msgs pcl_conversions racer_fidelity_msgs
  rclcpp sensor_msgs std_msgs visualization_msgs)
target_link_libraries(racer_oracle_algorithm
  hybrid_communication_racer::hybrid_assignment
  ${PCL_LIBRARIES} ${ARMADILLO_LIBRARIES} ${NLOPT_LIBRARY})
target_compile_definitions(racer_oracle_algorithm PRIVATE
  RACER_ORACLE_VARIANT ${PCL_DEFINITIONS})
set_target_properties(racer_oracle_algorithm PROPERTIES
  BUILD_RPATH "${NLOPT_PREFIX}/lib"
  INSTALL_RPATH "${NLOPT_PREFIX}/lib")

add_executable(racer_oracle_exploration_node
  "${CMAKE_CURRENT_SOURCE_DIR}/port/exploration_node_port.cpp")
target_include_directories(racer_oracle_exploration_node PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include" ${UPSTREAM_INCLUDE_DIRS})
ament_target_dependencies(racer_oracle_exploration_node
  hybrid_communication_racer rclcpp racer_fidelity_msgs)
target_compile_definitions(racer_oracle_exploration_node PRIVATE RACER_ORACLE_VARIANT)
target_link_libraries(racer_oracle_exploration_node racer_oracle_algorithm)

add_executable(racer_original_traj_server
  "${UPSTREAM}/plan_manage/src/traj_server.cpp")
target_include_directories(racer_original_traj_server PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include" ${UPSTREAM_INCLUDE_DIRS})
ament_target_dependencies(racer_original_traj_server
  geometry_msgs nav_msgs racer_fidelity_msgs rclcpp std_msgs visualization_msgs)
target_link_libraries(racer_original_traj_server racer_original_algorithm)

add_executable(racer_original_lkh_server
  "${CMAKE_CURRENT_SOURCE_DIR}/port/lkh_server_port.cpp")
target_include_directories(racer_original_lkh_server PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${UPSTREAM}/utils/lkh_mtsp_solver/include")
ament_target_dependencies(racer_original_lkh_server rclcpp racer_fidelity_msgs)
target_link_libraries(racer_original_lkh_server racer_original_lkh3)

install(TARGETS racer_original_algorithm racer_oracle_algorithm racer_original_lkh3
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(TARGETS racer_original_exploration_node racer_original_traj_server
  racer_oracle_exploration_node racer_original_lkh_server racer_original_lkh_cli
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY
  "${UPSTREAM}/active_perception/include/"
  "${UPSTREAM}/bspline/include/"
  "${UPSTREAM}/bspline_opt/include/"
  "${UPSTREAM}/exploration_manager/include/"
  "${UPSTREAM}/path_searching/include/"
  "${UPSTREAM}/plan_env/include/"
  "${UPSTREAM}/plan_manage/include/"
  "${UPSTREAM}/poly_traj/include/"
  "${UPSTREAM}/traj_utils/include/"
  DESTINATION include)
# Install the ROS2 transport/Isaac compatibility headers last.  A few have the
# same legacy include path as the upstream ROS1 headers (notably map_ros.h), and
# the compatibility version must be the one exported to downstream ROS2 code.
install(DIRECTORY include/ DESTINATION include)

ament_export_include_directories(include)
ament_export_libraries(racer_original_algorithm)
ament_export_dependencies(hybrid_communication_racer rclcpp racer_fidelity_msgs)
ament_package()
