#pragma once
#include <geometry_msgs/Point.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/msg/marker.hpp>
namespace visualization_msgs {
using Marker = msg::Marker;
using MarkerConstPtr = msg::Marker::ConstSharedPtr;
}
