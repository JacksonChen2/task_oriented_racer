# RACER Chen BS Recovery

这是一个独立的 ROS 2 外围恢复包。它不链接、复制或修改
`racer_original_core`，也不改变 RACER 的前沿提取、任务分配、视点选择、A*、
B-spline 或轨迹服务器逻辑。

当某个原 RACER exploration node 在 `/rosout` 的滑动窗口中持续报告
`No path to next viewpoint`，且对应 UAV 没有产生足够位移时，BS supervisor：

1. 使用各 UAV 已发布的 `/odom` 和带 hit intensity 的 `/points` 做稀疏射线融合；
2. 汇总 `/planning_vis/frontier`，在 frontier 附近选择已知自由 staging voxel；
3. 只在观测为 FREE 的 voxel 上执行保守的六邻域 A*；
4. 通过控制 mux 临时取得该 UAV 的速度控制权，最多前进 3 m；
5. 到达、超时或状态异常后先制动，再释放 mux；
6. 发布现有 `/tracking_lost`，让原 RACER 从实测 odometry 重新规划。

未知 voxel 永远不会被 A* 当成可通行空间。最终 `/cmd_vel_3d` 后原有 Isaac
障碍物 CBF 和机间 CBF 仍保留最后安全权限。

## 编译与测试

```bash
cd racer_chen/ros2_original_fidelity_sionna_ws
colcon build --packages-select racer_bs_recovery
source install/setup.bash
colcon test --packages-select racer_bs_recovery
colcon test-result --verbose
```

## 必需的控制话题接入

不能让 RACER adapter 和 BS 同时直接发布最终 `/cmd_vel_3d`。在启动文件创建每个
`position_command_adapter` 时，为其增加一条 ROS remap：

```python
remappings=[
    (
        f"/drone_{zero_id}/cmd_vel_3d",
        f"/drone_{zero_id}/cmd_vel_3d/racer",
    )
]
```

这只是 ROS 话题接线，不修改 RACER。接线后的数据流为：

```text
position_command_adapter -> /cmd_vel_3d/racer --+
                                                  +-> cmd_vel_mux -> /cmd_vel_3d
BS supervisor            -> /cmd_vel_3d/bs -------+
```

也可以直接使用本包提供的组合 launch；它通过作用域 remap 包含原 Sionna launch，
无需修改原文件：

```bash
ros2 launch racer_bs_recovery warehouse_sionna_with_bs_recovery.launch.py \
  drone_count:=5 scenario:=warehouse_loaded network_topology:=bs_round_robin
```

如果 RACER 已由其他方式启动并且 adapter 输出已经完成上述 remap，只启动恢复节点用：

```bash
ros2 launch racer_bs_recovery bs_recovery.launch.py drone_count:=5
```

## 参数

默认参数位于 `config/default.yaml`。`map_origin` 只用于把世界坐标稳定地量化为
稀疏 voxel key，不改变 RACER 地图。正式实验建议先保持较低的
`maximum_speed_mps` 和较短的 `maximum_recovery_distance_m`。

## 通信实验边界

当前实现订阅同一 ROS graph 中各 UAV 的点云和 frontier marker，适合先验证恢复控制
本身。若实验要求所有 BS 信息严格经过 Sionna 链路，应把这些状态话题加入通信代理
的 BS uplink policy；恢复算法和控制 mux 无需改动，但届时 BS 只能使用实际成功上行
的数据。
