cmake_minimum_required(VERSION 3.8)
project(racer_sionna_comm)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_gtest REQUIRED)
find_package(ament_cmake_pytest REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(racer_fidelity_msgs REQUIRED)
find_package(racer_recovery_core REQUIRED)
find_package(racer_sionna_interfaces REQUIRED)

add_library(racer_sionna_link_model
  src/link_model.cpp
  src/shared_ofdma_scheduler.cpp)
target_include_directories(racer_sionna_link_model PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

add_executable(racer_sionna_communication_proxy
  src/communication_proxy_node.cpp)
target_link_libraries(racer_sionna_communication_proxy racer_sionna_link_model)
target_include_directories(racer_sionna_communication_proxy PRIVATE include)
ament_target_dependencies(racer_sionna_communication_proxy
  rclcpp nav_msgs racer_fidelity_msgs racer_recovery_core racer_sionna_interfaces)

install(TARGETS
  racer_sionna_link_model
  racer_sionna_communication_proxy
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config assets DESTINATION share/${PROJECT_NAME})
install(PROGRAMS
  scripts/sionna_channel_node.py
  scripts/convert_usd_to_sionna.py
  scripts/generate_hybrid_radio_cache.py
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  ament_add_gtest(racer_sionna_link_model_test test/test_link_model.cpp)
  target_link_libraries(racer_sionna_link_model_test racer_sionna_link_model)
  target_include_directories(racer_sionna_link_model_test PRIVATE include)
  ament_add_gtest(racer_sionna_ofdma_scheduler_test
    test/test_shared_ofdma_scheduler.cpp)
  target_link_libraries(racer_sionna_ofdma_scheduler_test
    racer_sionna_link_model)
  target_include_directories(racer_sionna_ofdma_scheduler_test PRIVATE include)
  ament_add_gtest(racer_sionna_bs_ofdma_scheduler_test
    test/test_bs_ofdma_scheduler.cpp)
  target_link_libraries(racer_sionna_bs_ofdma_scheduler_test racer_sionna_link_model)
  target_include_directories(racer_sionna_bs_ofdma_scheduler_test PRIVATE include)
  ament_add_gtest(racer_sionna_event_driven_one_shot_test
    test/test_event_driven_one_shot.cpp)
  target_include_directories(racer_sionna_event_driven_one_shot_test
    PRIVATE include)
  ament_add_pytest_test(racer_sionna_source_owned_bs_csma_test
    test/test_source_owned_bs_csma.py TIMEOUT 70)
  ament_add_pytest_test(racer_sionna_bs_ofdma_proxy_test
    test/test_bs_ofdma_proxy.py TIMEOUT 70)
  ament_add_pytest_test(racer_sionna_ideal_proxy_test
    test/test_ideal_proxy.py TIMEOUT 70)
  ament_add_pytest_test(racer_sionna_active_links_test
    test/test_sionna_active_links.py TIMEOUT 30)
  ament_add_pytest_test(racer_sionna_synchronous_proxy_test
    test/test_synchronous_proxy.py TIMEOUT 40)
  ament_add_pytest_test(racer_sionna_shared_async_proxy_test
    test/test_shared_async_proxy.py TIMEOUT 45)
endif()

ament_export_include_directories(include)
ament_export_libraries(racer_sionna_link_model)
ament_package()
