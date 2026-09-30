# CLAUDE.md — GR00T-WholeBodyControl-X2

> 本文件是给 Claude Code 的项目上下文。**项目自带的文档非常完整**（`docs/x2/` 下有 13 个 runbook），
> 本文件只做导航、补充本地部署状态、以及记录官方文档没写全的坑。
> **遇到细节问题先去查 `docs/x2/`，不要凭本文件臆断。**

---

## 一、项目意图

**AgiBot X2 Ultra 作为 NVIDIA GEAR-SONIC 的第二个本体（embodiment）。**

上游是 NVIDIA 的 GR00T-WholeBodyControl（人形 G1 的全身控制 + 运动跟踪）。
这个分支把 **AgiBot X2 Ultra（31 DOF，配 OmniHand 灵巧手）** 接了进来，提供：

- **训练/微调**：原生 3-encoder 策略，以及「冻结 G1 核心 + LoRA」的两条线路
- **MuJoCo 仿真栈**：在 docker 里跑**真正的 C++ deploy 二进制**（不是简化版仿真）
- **真机部署**：PC2（机器人上的 Jetson Orin NX）走 colcon 部署
- **三种操作输入**：手柄、Quest 3 VR、**Pico 全身遥操作**
- kplanner ↔ 全身模式切换、动作回放、Pico 录制回放

### 两条模型线路（容易混）

| 线路 | 说明 | 文档 |
|---|---|---|
| **native 3-encoder** | 从零在 X2 上训练，三个编码器（G1 姿态 / teleop / SMPL） | F06 |
| **frozen G1-core + LoRA** | 保留公开的 G1 SONIC 核心，只训 LoRA 适配 X2 | F07 |

仓库自带一套原生双头 SONIC 模型（`gear_sonic_deploy/models/x2_sonic_v16ft8_45000_*`），
**所有 launcher 默认用它**，不需要额外下载就能跑。

### 三个编码器 = 三条控制路径

```
kplanner（手柄/Quest）  → pose encoder       → 遛弯、走位
Pico 全身遥操作          → SMPL encoder       → 全身跟随
动作片段回放             → G1 pose encoder    → 播放录好的动作
```

---

## 二、本机部署状态（2026-09-24）

| 项 | 值 |
|---|---|
| 部署路径 | `/home/nvidia/Documents/czq/GR00T-WholeBodyControl-X2/` |
| Python 环境 | **项目内 `.venv`**（Python 3.10.21，由 `czq/xr-env` 的 python 创建） |
| git-lfs | ✅ 3.8.0（装在 `/home/nvidia/miniconda3/bin/git-lfs`，**未动系统**） |
| Docker | ✅ 28.4.0（⚠️ **当前用户无守护进程权限**，见下方「坑 5」） |
| ROS 2 | ✅ `/opt/ros/jazzy` + `colcon` 已装 |
| 磁盘 | 936 GB 总量，约 620 GB 可用 |
| 动作库 | ✅ 已构建（`x2_demo_bank.pkl` 51 片段 450s + `x2_pad_banks.pkl` + planner primitives + 51 个 x2m2 片段） |
| 验证状态 | ✅ `check_environment.py`(deploy) 通过、14/14 模块导入、`play_x2_motion_mujoco.py` 回放成功、pytest 8/8 通过（2026-09-24） |

### ⚠️ `.venv` 的一个耦合

`.venv` 是用 `czq/xr-env/bin/python` 创建的，因此它的 **stdlib 来自 `xr-env`**
（`pyvenv.cfg` 里 `home = /home/nvidia/Documents/czq/xr-env/bin`）。
site-packages 是独立的（隔离性已验证：看不到 xr-env 的 torch/mujoco/numpy），
**但 venv 依赖 `czq/xr-env` 存在**。删掉 xr-env 会连带弄坏这个 venv。

### 为什么必须独立环境

项目在 `gear_sonic/pyproject.toml` 里**钉死 `numpy==1.26.4`**，而 `xr-env` 是 numpy 2.2.6。
共用环境会把 xr-env 的 numpy 降级，很可能弄坏 XRoboToolkit 那边的 `placo`/`pinocchio`。

---

## 三、环境搭建（复现用）

官方推荐流程（`docs/x2/README.md` 的 10 分钟 quickstart）：

```bash
cd /home/nvidia/Documents/czq/GR00T-WholeBodyControl-X2

# 1. 模型（git-lfs）—— 不做这步，所有 .onnx/.STL 都只是 ~130 字节的 LFS 指针
export PATH=/home/nvidia/miniconda3/bin:$PATH     # 本机 git-lfs 在 conda base 里
git lfs install --local
git lfs pull

# 2. Python 环境
python3.10 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip

# 3. ⚠️ 顺序很重要：先装 CPU 版 torch，否则 gear_sonic 的 torch>=2.4.0
#    会去 PyPI 拉多 GB 的 CUDA 轮子
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. 两个 first-party 包 + 其余依赖
pip install -e "./gear_sonic[sim]" -e ./motionbricks
pip install onnxruntime pygame websockets

# 5. 构建动作库（pad 手势库 / planner primitives / x2m2 片段）
bash tools/build_demo_bank_from_upstream.sh
```

> 也有幂等脚本：`bash install_scripts/setup_x2.sh --skip-models`（加 `--with-docker` 会一并构建镜像）。
>
> **aarch64 注意**：`gear_sonic[teleop]` 里的 `pyvista` 带了 `platform_machine != 'aarch64'` 标记，
> 在 Jetson 上会被自动跳过——项目对 ARM 是有感知的。

---

## 四、文件架构

```
GR00T-WholeBodyControl-X2/
├── gear_sonic/              ★ 主 Python 包：训练 + 仿真 + 全部脚本
│   ├── config/              hydra 配置（exp/ 下是训练实验，kplanner_profiles/ 是运行时档位）
│   ├── envs/                Isaac Lab 训练环境（x2_ultra.py 是训练侧本体）
│   ├── scripts/             ★ 61 个 Python + 12 个 .sh，几乎所有入口都在这
│   ├── data/                ★ 资产（assets/robot_description/{mjcf,urdf,meshes}）+ 动作库
│   ├── train_agent_trl.py   ★ 训练主入口
│   └── tests/
├── gear_sonic_deploy/       ★ 部署侧：C++ 节点 + docker + 模型
│   ├── src/x2/agi_x2_deploy_onnx_ref/   C++ deploy 节点（ROS 2/colcon）
│   ├── deploy_x2.sh         在 docker(sim) 或真机上启动节点
│   ├── docker_x2/           sim 镜像与 compose
│   ├── configs/             调参档位、站姿、仿真初始位姿
│   └── models/              ★ 自带的 ONNX 模型集（git-lfs）
├── decoupled_wbc/           上游的分层 WBC（control / dexmg / sim2mujoco）
├── motionbricks/            动作先验模型（VQ-VAE / pose / root）
├── x2_pc2/                  机器人侧「仪式」脚本（PC2 = 机身上的 Orin NX）
│   ├── PORT_REGISTRY.md     ★ 所有 ZMQ 端口登记表
│   └── push_to_pc2.sh       带 md5 校验的文件推送
├── docs/x2/                 ★★ 13 个 runbook（F01~F12）+ 架构/构建链/训练笔记
├── tools/                   check_docs.py / check_x2_ownership.py / 动作库构建
├── install_scripts/         各子系统的安装脚本
├── external_dependencies/   随仓库附带的 SDK（unitree_sdk2_python、XRoboToolkit-PC-Service-Pybind）
└── X2_FILES.txt             ★ 文件归属清单：哪些是本分支新增/修改的
```

**导航要点**：
- `X2_FILES.txt` 是**这个 port 相对上游改了什么**的权威清单（`A`=新增 `M`=修改 `D`=删除）。
  上游文件不在这个清单里。**做 rebase 时以此为准。**
- `x2_pc2/PORT_REGISTRY.md` 是端口权威来源；**仿真的端口是机器人端口 +100**（5556→5656）。

---

## 五、运行时链路（理解这个就理解了一半）

所有控制器最终都汇聚到一处：**50 Hz 的参考位姿**（或运动 token）送进 SONIC deploy 节点，
节点跑 ONNX 策略驱动物理。

```
手柄/Quest ──planner_cmd:5563──┐
动作片段   ──motion_clip:5568──┤
                               ├→ kplanner ONNX ──pose:5556──→ x2_pose_merger
                               │                                      ↓
Pico 全身 ──pico intent:5573──→ SMPL tokenizer ──tokens:5574──→ pose_watchdog ──pose:5558──→ SONIC deploy
                                                                                              ↓
                                                                    plant（仿真=MuJoCo / 真机=AgiBot SDK）
```

关键点：
- **kplanner 走 pose encoder 路径，Pico 走 SMPL encoder 路径**，
  `x2_pose_merger.py` 负责仲裁两者和手势库。
- **仿真用的是真正的 C++ deploy 二进制**（跑在 docker 里），
  宿主侧 `x2_mujoco_ros_bridge.py` 只当物理引擎。所以「仿真里能跑」≈「真机大概率能跑」。
- 仿真栈 source 的是 `x2_pc2/robot_env.env`——**和真机同一个文件**，保证两边参数一致。

---

## 六、开发注意事项与陷阱

### 坑 1：模型全是 LFS 指针 ⭐

克隆后 `gear_sonic_deploy/models/*.onnx` 只有 ~130 字节：

```
version https://git-lfs.github.com/spec/v1
oid sha256:...
size 57696872
```

**不执行 `git lfs pull` 的话，仿真和部署全都跑不起来。**
先自检：`bash check_environment.py`（它有 Git LFS 检查项）。

### 坑 2：`numpy==1.26.4` 是钉死的

`gear_sonic/pyproject.toml` 硬钉 `numpy==1.26.4` + `scipy==1.15.3`。
**不要把它和用 numpy 2.x 的环境混用。**

### 坑 3：torch 必须**先**装 CPU 版

`requirements-x2.txt` 里专门解释了顺序。如果直接 `pip install -e "gear_sonic[sim]"`，
`torch>=2.4.0` 会从 PyPI 拉 CUDA 轮子（aarch64 上就是 450 MB + 一堆 nvidia_* 依赖，合计数 GB）。

CPU 源是有 aarch64 轮子的（2.10~2.12 都有 `cp310-manylinux_2_28_aarch64`），
所以文档里的 `--index-url https://download.pytorch.org/whl/cpu` 在 Jetson 上**可用**。

### 坑 4：仿真栈需要 docker，且端口是 +100

- `simstack_local.sh` 会在 `x2sim` 容器里构建并运行 C++ deploy 节点，
  **首次构建约 10 分钟**。
- 仿真端口 = 机器人端口 +100（5556→5656, 5563→5663, 5568→5668）。
- **每次结束都要 `./gear_sonic/scripts/simstack_local.sh --stop`**（即使 Ctrl-C 之后），
  否则残留容器会让下次启动行为诡异。检查：`docker ps` 里不该有 `x2sim`。

### 坑 5：本机 docker 权限（**当前未解决**）

Docker 已装（28.4.0），但 `docker info` 报当前用户无权限。要跑仿真栈需要二选一：

```bash
sudo usermod -aG docker $USER     # 然后重新登录（推荐）
# 或每次用 sudo
```

> 这属于系统级改动，**本机部署时刻意没做**，需要时再执行。

### 坑 6：`ALLOW_MISMATCH=1` 的作用域

仿真里跑**任何不是机器人上那套**的模型时**必须**加 `ALLOW_MISMATCH=1`，
否则启动器会因为模型不匹配而拒绝启动。这是防止「以为在测 A 其实跑的是 B」的护栏。

### 坑 7：「仿真通过」不等于「能上机器人」

`docs/x2/F09` 明确写了：**Isaac Lab 的数字不能作为模型放行依据**。
发布门槛是在 MuJoCo 里跑完整个动作库电池：

```bash
python gear_sonic/scripts/eval_x2_mujoco.py --motions gear_sonic/data/motions/x2_demo_bank.pkl
```

### 坑 8：`RTF` 会骗你

`SIM_VIEWER_MODE=container`（容器内渲染）会让实时因子减半（`RTF 0.51`），
**把参考动作以两倍速灌进物理**，从而掩盖不稳定。要盯 `RTF 1.00`。
诊断视角请用宿主侧的 `sim_mirror_viewer.py`（只读，不干扰物理循环）。

### 坑 9：plant 一致性门禁

每次启动都会跑 `check_plant_consistency.py`。报 `REFUSING LAUNCH: plant copies disagree`
说明 MJCF/头文件与 plant yaml 不同步，要按 `docs/x2/BUILD_CHAIN.md` 的 C 节重新生成。

### 约定与检查

```bash
# 代码风格（black line-length=100，isort，ruff line-length=115）
make run-checks      # isort --check + black --check + ruff check

# 项目自带的两个一致性检查（改文档/加文件后必跑）
python tools/check_docs.py           # 文档里引用的路径都存在
python tools/check_x2_ownership.py   # 新增文件都登记到 X2_FILES.txt 了

# 测试
.venv/bin/python -m pytest gear_sonic/tests -q --ignore=gear_sonic/tests/test_input_readers.py
```

> `test_input_readers.py` 是上游未改动的文件，**在上游也是坏的**（引用了已删除的函数），
> 所以被排除。不是你的问题。

---

## 七、常用入口速查

```bash
cd /home/nvidia/Documents/czq/GR00T-WholeBodyControl-X2
source .venv/bin/activate

# 环境自检
python check_environment.py

# ---- 不需要 docker 也能做的 ----
# 运动学检查：在 X2 MJCF 上播放一段动作（无策略）
python gear_sonic/scripts/play_x2_motion_mujoco.py --motion gear_sonic/data/motions/x2_demo_bank.pkl

# 用 ONNX 策略跑 MuJoCo 评估（PD 回路）
python gear_sonic/scripts/eval_x2_mujoco_onnx.py --onnx <name>_g1.onnx --motion <clip.pkl>

# ---- 完整仿真栈（需要 docker）----
./gear_sonic/scripts/sim_onnx_planner.sh --whole-body-teleop   # Pico 全身遥操作
./gear_sonic/scripts/sim_onnx_planner.sh                       # 手柄模式
./gear_sonic/scripts/sim_onnx_planner.sh --dry-run             # 只打印解析出的所有路径后退出
./gear_sonic/scripts/simstack_local.sh --stop                  # ★ 用完必执行
```

**导航到详细文档**：
`docs/x2/README.md`（索引/quickstart）· `docs/x2/ARCHITECTURE.md`（栈图）·
`docs/x2/BUILD_CHAIN.md`（改什么要重建什么）· `docs/x2/TRAINING_NOTES.md`（训练踩坑）·
`MODELS.md`（所有模型变量：`MODEL` / `PLANNER_MODEL` / `X2_MODELS` / `SONIC_HOME` / `CKPT_ROOT`）

---

## 八、与相邻项目的关系

本机 `czq/` 下还有 **XRoboToolkit 遥操作环境**（见 [`../pico/CLAUDE.md`](../pico/CLAUDE.md)）：

| | XRoboToolkit | 本项目 |
|---|---|---|
| 用途 | 通用遥操作 SDK + 示例 | X2 人形全身控制 |
| 环境 | `czq/xr-env`（numpy 2.x） | 本项目 `.venv`（numpy 1.26.4） |
| 控制对象 | UR5e / A1X / G1 上半身 | AgiBot X2 Ultra 全身 31 DOF |
| 渲染 | MuJoCo 原生窗口 / meshcat | docker 内 C++ deploy + MuJoCo 桥接 |

**两者环境不通用**（numpy 版本冲突）。本仓库的 `external_dependencies/` 里
**自带一份 XRoboToolkit-PC-Service-Pybind**，不需要借用 `czq/xr-env` 里那份。
