cmake_minimum_required(VERSION 3.8)
project(echoflow)

# Default to C++17 for humble
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(marine_sensor_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(grid_map_core REQUIRED)
find_package(grid_map_cv REQUIRED)
find_package(grid_map_ros REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

include_directories(
  include
)

add_executable(radar_grid_map
  src/radar_grid_map_node.cpp
  src/radar_grid_map.cpp
)
ament_target_dependencies(radar_grid_map
  rclcpp
  marine_sensor_msgs
  nav_msgs
  geometry_msgs
  grid_map_core
  grid_map_ros
  grid_map_msgs
  tf2
  tf2_ros
)

add_executable(flow_tracker
  src/radar_grid_map_node.cpp
  src/grid_map_filters.cpp
  src/particle_filter.cpp
  src/particle_filter_node.cpp
  src/statistics.cpp
  src/flow_tracker.cpp
)
ament_target_dependencies(flow_tracker
  rclcpp
  marine_sensor_msgs
  sensor_msgs
  grid_map_core
  grid_map_cv
  grid_map_ros
  grid_map_msgs
  tf2_ros
)

install(TARGETS
  radar_grid_map
  flow_tracker
  DESTINATION lib/${PROJECT_NAME}/
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
