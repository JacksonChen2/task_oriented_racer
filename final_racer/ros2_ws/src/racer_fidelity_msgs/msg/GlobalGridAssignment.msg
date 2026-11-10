int32 coordinator_id
float64 stamp
uint32 epoch
# Bootstrap is retained only for explicitly enabled legacy experiments.
# Recovery is the normal pairwise mode's exceptional idle-recovery message.
uint8 PURPOSE_BOOTSTRAP=0
uint8 PURPOSE_RECOVERY=1
uint8 purpose
bool global_finish
int32[] recovery_uav_ids
int32[] affected_uav_ids
# Conditional-apply snapshot aligned with fleet UAV order. A recovery cannot
# overwrite a pair commit that changed an affected UAV after this snapshot.
uint64[] base_assignment_epochs
# Empty means the legacy all-UAV assignment.  A non-empty list scopes this
# assignment and its offsets to one startup local connected component.
int32[] member_ids
int32[] offsets
int32[] grid_ids
