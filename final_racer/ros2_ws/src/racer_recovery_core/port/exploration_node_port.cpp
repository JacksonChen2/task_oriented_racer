#include <exploration_manager/fast_exploration_fsm.h>
#include <ros/ros.h>

int main(int argc, char **argv) {
  ros::init(argc, argv, "racer_recovery_exploration_node");
  ros::NodeHandle node("~");
  fast_planner::FastExplorationFSM fsm;
  fsm.init(node);
  ros::spin();
  return 0;
}
