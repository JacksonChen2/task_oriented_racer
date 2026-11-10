# 源码与资产来源

本仓库包含 RACER 的 ROS 2 移植、ROS 消息、LKH、NLopt 2.7.1 源码，以及训练通信代理、Sionna 信道节点、Isaac 适配器和 Qwen 指导相关代码。

场景相关文件包括仓库 USD 主场景、无人机/AP 模型、Sionna XML/PLY 网格、占据体素真值和初始位置。大型场景依赖 `final_racer/assets/scene_dependencies/` 需另行补齐，其来源和校验清单保存在 `final_racer/assets/scene_dependencies_manifest.json`。训练所需参考输入由运行者在本地提供。

ROS package.xml 和上游源码所附许可声明继续有效。NLopt 的 COPYING 等文件保留在 third_party 中。NVIDIA 场景资产适用其原始资产许可；本仓库不改变这些权利归属。OmniPBR/OmniGlass 等内置材质由目标机器的 Isaac Sim 安装提供。
