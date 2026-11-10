#pragma once
#include <racer_fidelity_msgs/srv/solve_tsp.hpp>
namespace lkh_tsp_solver {
struct SolveTSP {
  using RosService = racer_fidelity_msgs::srv::SolveTSP;
  RosService::Request request;
  RosService::Response response;
};
}
