#pragma once
#include <racer_fidelity_msgs/srv/solve_mtsp.hpp>
namespace lkh_mtsp_solver {
struct SolveMTSP {
  using RosService = racer_fidelity_msgs::srv::SolveMTSP;
  RosService::Request request;
  RosService::Response response;
};
}
