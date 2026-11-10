int32 drone_id
int32[] grid_ids
uint64 assignment_epoch
float64 commitment_until
bool requesting_work
float64 recent_attempt_time
float64 stamp
float32[] pos
float32[] vel
float32 yaw

# Startup-only gossip for exploration_assignment_mode=local_component.
# heard_neighbor_ids lists UAVs whose normal Sionna DroneState was received
# recently.  Seeing our own ID in a peer's list proves the link is bidirectional.
int32[] heard_neighbor_ids
int32 local_component_coordinator_id
int32[] local_component_member_ids
# Flattened xyz positions aligned with local_component_member_ids.
float32[] local_component_member_positions
uint32 local_component_assignment_epoch
int32[] local_component_ack_ids
