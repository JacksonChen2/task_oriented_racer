#pragma once
#include <geometry_msgs/Pose.h>
#include <racer_fidelity_msgs/msg/position_command.hpp>
namespace quadrotor_msgs {
using PositionCommand = racer_fidelity_msgs::msg::PositionCommand;
using PositionCommandConstPtr = PositionCommand::ConstSharedPtr;
}
