# task_oriented_racer

运行所需的仿真环境、模型和依赖版本如下。

| 组件 | 版本 / 要求 |
|---|---|
| 操作系统 | Ubuntu 22.04，x86_64 |
| Isaac Sim | 5.1 |
| ROS 2 | Humble |
| Sionna RT | 2.0.1 |
| Mitsuba | 3.8.0 |
| Dr.Jit | 1.3.1 |
| 大语言模型 | Qwen3-8B-FP8，本地模型权重 |
| GPU 环境 | NVIDIA GPU，兼容的驱动、CUDA 和 Vulkan，以及可同时运行仿真与模型的显存 |

Python 环境需分别配置：

| 用途 | Python 版本 |
|---|---|
| ROS 2 与 Sionna RT | Python 3.10 |
| Isaac Sim | Isaac Sim 自带 Python 3.11 |
| Qwen3 与训练程序 | Python 3.12 |

训练环境依赖版本见 [requirements-training.txt](requirements-training.txt)，Sionna 环境依赖版本见 [requirements-sionna.txt](requirements-sionna.txt)。本机安装路径配置模板见 [env.example.sh](env.example.sh)。

场景环境需要与代码匹配的仓库 USD 场景、无人机/AP 模型、Sionna XML/PLY 网格及纹理依赖。大型依赖目录 `final_racer/assets/scene_dependencies/` 未上传至 GitHub，运行仿真前需从完整场景包补齐。场景依赖清单见 [scene_dependencies_manifest.json](final_racer/assets/scene_dependencies_manifest.json)。

真实训练还需要本地参考输入，通过 `env.local.sh` 中的 `RACER_QWEN8_PERFECT_RESULT` 指定 JSON 文件路径。该输入需与场景、无人机数量及任务时长匹配；仓库不提供参考数据或实验输出。Mock 测试无需该输入。
