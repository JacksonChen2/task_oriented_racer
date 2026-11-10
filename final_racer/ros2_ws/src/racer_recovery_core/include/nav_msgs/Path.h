#pragma once
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/msg/path.hpp>
namespace nav_msgs {
using Path = msg::Path;
using PathConstPtr = msg::Path::ConstSharedPtr;
}
