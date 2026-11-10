#include <exploration_manager/fast_exploration_fsm.h>
#include <ros/ros.h>

int main(int argc, char **argv) {
#ifdef RACER_ORACLE_VARIANT
  ros::init(argc, argv, "racer_oracle_exploration_node");
#else
  ros::init(argc, argv, "racer_original_exploration_node");
#endif
  ros::NodeHandle node("~");
  fast_planner::FastExplorationFSM fsm;
  fsm.init(node);
  ros::spin();
  return 0;
}
