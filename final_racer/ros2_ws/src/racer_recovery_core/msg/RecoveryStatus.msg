uint8 PHASE_NORMAL=0
uint8 PHASE_LOCAL_RESELECT=1
uint8 PHASE_LOCAL_ESCAPE=2
uint8 PHASE_WAITING_FOR_AP=3
uint8 PHASE_HOLD=4

builtin_interfaces/Time stamp
uint32 episode_id
int32 drone_id
uint8 phase
uint32 consecutive_failures
float64 failure_duration_s
float64 progress_m
geometry_msgs/Point position
geometry_msgs/Point goal
int32 failed_grid_id
int32[] grid_ids
bool request_repartition
