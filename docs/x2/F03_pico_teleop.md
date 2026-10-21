# F03 — Pico whole-body teleop

Two architectures ship, both driven by the same headset:

1. **Deploy-faithful ("Quest split")** — `run_pico_teleop.sh` on the laptop
   streams human *intent* only (SMPL joints, engage flag, wrist targets) on
   :5573 to the deploy side, where the SMPL tokenizer runs
   (`pc2_pico_token_service.py` on PC2, or the dual-head deploy directly).
   Same command for sim and robot; the target is chosen by `--pc2-host`.
2. **Standalone sim (SMPL path)** — `run_x2_pico_wbc.sh` drives a MuJoCo sim
   in-process through the SMPL encoder. **SIM-ONLY**: no deploy safety stack;
   never point it at the robot. Used for model feel-tests and tape gates.

Controller grammar (both): **A+B+X+Y** held ~0.4 s = OFF <-> LOCOMOTION,
**B** = LOCOMOTION <-> WHOLE_BODY (engages on entry), sticks = kplanner walk /
turn with **LT held as deadman**, **X/Y** = speed setpoint -/+ 0.1 m/s, **A+X**
held + pumping both triggers = e-stop. Grips drive the OmniHand fingers.
`gear_sonic/scripts/pico_button_probe.py` prints raw button state if a
mapping looks wrong.

## Prerequisites

- Quickstart steps 1–5 of [`README.md`](README.md).
- `bash install_scripts/install_pico.sh` — builds `.venv_teleop`
  (`xrobotoolkit_sdk`, `isaacteleop`) and installs the XRoboToolkit PC
  Service (`SKIP_PC_SERVICE=1` to skip the deb). Never start the service
  from its desktop icon; the launchers start the bare daemon themselves
  (`pgrep -x RoboticsService` shows it).
- Headset (once): same Wi-Fi as the PC (`ping <headset-ip>` < 100 ms while
  worn), developer mode on, sideload the XRoboToolkit PICO client APK
  (`python3 -m http.server 8000` on the PC, open the APK URL in the headset
  browser). Motion Tracker app: set **max tracker usage = 3** *before*
  pairing waist + both ankles, select **Full body** mode, then **Calibrate**
  (a headset reboot wipes calibration — recalibrate after every reboot).
- Every session: open XRoboToolkit on the headset, connect to the PC IP,
  **send data ON**, enter the VR scene and keep the app foregrounded. Probe:

```bash
.venv_teleop/bin/python -c "import xrobotoolkit_sdk as xrt, time; xrt.init(); time.sleep(2); print('body:', xrt.is_body_data_available(), 'stamp:', xrt.get_time_stamp_ns(), 'trackers:', xrt.num_motion_data_available())"
# want: body: True, an advancing stamp, trackers: 2 (or 3)
```

- Models: the shipped native dual-head set (`gear_sonic_deploy/models/`, no
  variables needed) **or** your own dual-head / frozen-G1 token set
  ([`MODELS.md`](../../MODELS.md)).

## Sim — deploy-faithful whole-body teleop (acceptance commands)

Terminal 1, the stack. Shipped set + shipped tuning preset (canonical):

```bash
cd <repo> && ./gear_sonic/scripts/sim_onnx_planner.sh --whole-body-teleop
```

Your own native dual-head set (explicit-env variant):

```bash
cd <repo> && TOKEN_SVC_OPERATOR_ROOT_LEVEL=zero ALLOW_MISMATCH=1 SIMSTACK_OMNIHAND=1 \
SIM_TUNING_YAML=gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0_inc_waistmc.yaml \
MODEL=$X2_MODELS/pico_39000/exported/x2_sonic_39000_g1.onnx \
SIMSTACK_DUAL_MODEL=$X2_MODELS/pico_39000/exported/x2_sonic_39000_dual.onnx \
PLANNER_MODEL=$X2_MODELS/kplanner_g1core \
KPLANNER_PROFILE=gear_sonic/config/kplanner_profiles/bigrun_teleop_v1.env \
./gear_sonic/scripts/sim_onnx_planner.sh --whole-body-teleop
```

Frozen-G1 token set (pose graph + token graph + release tokenizer):

```bash
cd <repo> && TOKEN_SVC_OPERATOR_ROOT_LEVEL=zero ALLOW_MISMATCH=1 SIMSTACK_OMNIHAND=1 \
SIM_TUNING_YAML=gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0.yaml \
MODEL=$X2_MODELS/s1p4_35000/exported/x2_sonic_s1p4_35000_g1.onnx \
SIMSTACK_TOKEN=$X2_MODELS/s1p4_35000/exported/x2_sonic_s1p4_35000_g1_token.onnx \
SIMSTACK_TOKENIZER=$X2_MODELS/armA_14900/exported/x2_smpl_tokenizer_v11release.onnx \
PLANNER_MODEL=$X2_MODELS/kplanner_g1core \
KPLANNER_PROFILE=gear_sonic/config/kplanner_profiles/bigrun_teleop_v1.env \
./gear_sonic/scripts/sim_onnx_planner.sh --whole-body-teleop
```

Terminal 2, the headset half (no `--pc2-host` = this sim stack):

```bash
./gear_sonic/scripts/run_pico_teleop.sh
./gear_sonic/scripts/run_pico_teleop.sh --cam sim --headset-ip <HEADSET_IP> --cam-preview   # + sim ego view in the headset
```

Expected output:

- Terminal 1: `[whole-body] NATIVE DUAL-HEAD graph — one SONIC session, both reference heads`
  (or `[whole-body] TOKEN GRAPH — deploy-faithful rehearsal via simstack_local.sh`),
  the `FULL-LOOP SIM` banner, deploy `Loaded ONNX ... kind=native_dual_head`
  (or the token service's `tokenizer ...` line), and the deploy binding :5573.
- Terminal 2: the sender waits for :5573, then prints rate lines with
  `body OK`; every mode flip, engage / `ENGAGE REFUSED ... RECALIBRATE in headset`,
  and deadman edge is logged.
- MuJoCo: A+B+X+Y -> LOCOMOTION (sticks walk), B -> WHOLE_BODY: the robot
  follows your torso, arms and legs; B again releases (hold -> rate-limited
  return to the kplanner).

Stop: Ctrl-C the sender, then `./gear_sonic/scripts/simstack_local.sh --stop`.

Knobs: `KPLANNER_WB_ARM_LATCH_GAIT=keep` (default) keeps the latched arm pose
through a gait; `WRIST_SMPL_MAP=lp:+y,lr:+z,rp:-y,rr:+z` flips a wrist sign;
`--head-source off` disables head-yaw follow; `HEAD_YAW_SIGN=-1` flips it.

## Sim — standalone SMPL path (feel-test, no docker)

```bash
# fused SMPL ONNX (obs 1830) from a native set:
./gear_sonic/scripts/run_x2_pico_wbc.sh --onnx $X2_MODELS/pico_39000/exported/x2_sonic_39000_smpl.onnx
# torch checkpoint (native 3-encoder .pt), or a frozen-G1 composite:
./gear_sonic/scripts/run_x2_pico_wbc.sh --model $X2_MODELS/pico_39000/model_step_039000.pt
./gear_sonic/scripts/run_x2_pico_wbc.sh --model "frozen-core-smpl:$X2_MODELS/armA_14900/armA_14900_merged.pt"
# the public G1 core on the G1 body (pair with --robot g1):
./gear_sonic/scripts/run_x2_pico_wbc.sh --robot g1 --model "smpl-g1:$SONIC_HOME/g1/sonic_v1_1/last.pt"
```

`--model` + `--onnx` together = torch drives, ONNX shadows, per-step diff
CSV. Controls: both grips 0.5 s = engage / disengage, SPACE in the viewer =
same, R = reset. Stop: Ctrl-C.

## 手臂 IK 后端（incremental 上半身链，仅 sim）

`incremental_chain.py` / `incremental_chain_v3.py`（harness:
`gear_sonic/scripts/incremental_upper_teleop_mujoco.py`、
`gear_sonic/scripts/arm_teleop_v3_mujoco.py`）的每条手臂各持一个 IK 求解器，
由 `make_arm_solver()` 选择：

| 取值 | 求解器 | 说明 |
|---|---|---|
| `mink`（默认） | `MinkArmIK` | mink 1.3 的一个 QP；v3 默认走 4-DoF：`solve_position_free4` 只解肩 3 + 肘，腕由 `WristClutch` 直填；`full_pose_ik=True` 时改为位置 + 姿态两个 `RelativeFrameTask`（权重都 1.0），完整 7-DoF |
| `dls` | `ArmIKSolver` | 手写纯 numpy 阻尼最小二乘（v1/v2 的原始实现，留作回退）；同样支持 4-DoF `solve_position_free4` 与 7-DoF `solve` |

选择顺序：构造参数 → 环境变量 `X2_IK_BACKEND` → 默认 `mink`。**缺 mink 时构造期
直接报错**（提示装 mink 或 `X2_IK_BACKEND=dls`），不做静默回退。两个 harness 启动
横幅都会打印解析到的求解器与自由度（v3 默认形如 `arm IK mink` + `4-DoF`；
v3 harness 加 `--full-pose-ik` 时为 `7-DoF`），`arm.backend` /
`arm.full_pose_ik` 是权威字段。

这条链的角色（2026-10 互换）：**握把键 grip 是每臂 clutch**（经 LPF 后过阈值；
`TRIGGER_THRESH` / `--trigger-thresh` 是历史名），**扳机 trigger 驱动 OmniHand
手指**（v2 `ratio = 1 - hand`，v3 `ratio = hand`）。真机 PC2 的平铺副本**尚未同步**，
仍是旧的 trigger clutch / grip 手指；读数前先确认跑的是哪一侧。

```bash
X2_IK_BACKEND=dls .venv/bin/python gear_sonic/scripts/arm_teleop_v3_mujoco.py --wire   # 回退到 DLS
```

### v3 腕部只转动却拖肘/肩：目标补偿与零空间（A/B/C，2026-10）

Pico 的 controller pose 是**控制器原点**，不是人的腕心。手腕原地转动时，控制器
原点会沿一根杠杆画弧；旧的位置目标 `tgt = a_rob + scale*(C - a_op)` 会把这根弧
当成平移，4-DoF IK 只能把位移分配给肩/肘——这就是「肩肘不动、只转腕，机器人却
抬肘」的来源。

三个开关都在 `arm_teleop_v3_mujoco.py` / `IncrementalTeleopTickerV3` 上：

| 开关 | 默认 | 作用 |
|---|---|---|
| `--ctrl-lever X Y Z`（`--ctrl-lever-left/right` 覆盖单侧） | `0 0 0` | 控制器坐标系下的「控制器原点 → 腕心」杠杆（m）。`W = C - R_c @ r_c`，锚点同样扣除；纯腕转时补偿后的目标保持不动。默认 0 = 旧行为，**必须按 [`PICO_LIVE_TELEOP_PITFALLS.md`](PICO_LIVE_TELEOP_PITFALLS.md) §9 标定后再填**；左右手柄握持方向不同，向量也不同，分别用单侧旗标传入 |
| `--position-deadband-mm` | `3.0` | 对补偿后的平移做软径向死区：`d -> d * max(0, 1 - r / |d|)`；半径内目标=锚点，半径外斜率 1（输出 `|d|-r`）。压手抖/重锚漂移，不改变真实位移的增益 |
| `--posture-gain` | `0.01` | 4-DoF 路径的零空间姿态正则：mink 是 `PostureTask` 的 cost，DLS 是 `(I - pinv(J)J)` 精确投影上的增益。目标在 engage 时设为当时的 `q`，只选肩-腕连线上的肘部 swivel，不动腕点 |

后两个默认只在 **4-DoF 默认路径**生效。`--full-pose-ik` 强制关闭它们：full-pose
的 A/B 预算（5 mm）已被冻结守卫在 tick 854 的分叉吃到 4.38 mm，1 mm 的死区就会
把它推过线；7-DoF 的零空间姿态是另一次迁移。`--ctrl-lever` 两条路径都用。

自检：`CHAIN_V3_CHECKS` 末尾新增 `ctrl_lever_pure_wrist_rotation`（合成「腕心不动、
控制器原点随姿态画弧」，断言补偿后目标 < 1e-6 m 且 q[:4] 漂移 < 0.01 rad；同时
断言单侧向量按臂路由）、`position_deadband`、`posture_gain_null_space`（mink/DLS 都跑）。

> **真机（PC2）仍跑 DLS**：mink / mujoco 没有、也不该被推到 Jetson
> （`gear_sonic[teleop]` 里没有 mink，`gear_sonic[sim]` 里才有）。所以
> **sim 里的手臂行为与真机不同是预期内的**，直到另一次单独的迁移把这条链搬上去。

### v4：无 body tracker 的 3-point 上肢遥操（`--tracking 3point`，仅 sim，2026-10）

**不用任何 Pico body tracker**：只戴头显、双手柄。v3 链的 frame 契约只消费
`t / lctrl_pose / rctrl_pose / trigger / grip`（`smpl_joints` 完全被忽略），所以 v4
是**换数据源**，不是改控制律：`ThreePointSource` 用两个 controller 的 pose 填满这 5 个
字段，`--tracking 3point` 打开这条路，IK / 手指极性 / LPF / 阈值全部不变（腕映射
数学不变；默认腕符号的修正见本节末）。

- **一次性固定 operator frame**：标定时取 `lat = p_left - p_right`（去掉世界竖直分量）
  与 `fwd = cross(lat, up)`，origin = 双手柄中点；标定后**冻结**，绝不逐 tick 更新。
  头显不参与帧朝向、不参与任何门控（只做状态打印）。wire 数学与 live full 模式
  （`_torso_M` / `_ctrl_wire_fields`）同构，`--ctrl-lever-*` 继续有效。
- **标定姿势**：双手各握手柄，自然举在胸前、**左右分开 ≥ 20 cm、保持约 1 秒**。
  sender 打印 `[3pt] operator frame calibrated ...` 后进入 `stream live.`。失败时每
  3 s 打印 `hold both controllers apart at chest height (>= 15 cm)` 并自动重试。
- **逐侧失效安全**：某侧 controller 无效或超过 `max_age_s`（0.2 s）时，该侧 wire
  `grip=0`、`trigger=0`，pose 保持最后一帧有效值（不会出现零 pose + live grip）。
- **wire `engaged` 强制 0**：3point 的 `smpl_joints`/`human_quat` 是占位零值，强制
  `engaged=0` 让同 topic 的 SMPL tokenizer 不接管；v3 链不读 `engaged`，**grip 就是离合**。
- 3point 强制 `--no-locomotion`，且与 `--tape-replay` / `--tape-once` 互斥（live-only）。
- **默认腕符号左右相同，逐轴仍在 live A/B**（2026-10-21）：`DEFAULT_WRIST_SIGNS`
  两侧必须一致。旧版左侧是右侧的全局取反（镜像假设），实测左腕 yaw/pitch 与手
  相反；第二轮两侧 `(+1,-1,+1)` 后左手正常、右手 pitch 反，第三轮当前取
  `(+1,+1,+1)` 实测。X2 两侧腕工具系平行而非镜像（零位 roll 轴均为世界 +x）。
  `CHAIN_V3_CHECKS[19]` 钉住“左右相同”；单轴不对时用
  `--wrist-{yaw,pitch,roll}-sign-l/-r` 只改那一轴。
- **X/A 长按回位**：左手柄 X 或右手柄 A 按住 ≥3 s，对应手臂按
  `home_speed_rad_s`（默认 0.5 rad/s，每关节）回到 `ARM_NEUTRAL` 初始姿态；
  回位期间该臂离合释放，完成后仍握 grip 会在新姿态重新 anchor，不跳变。
  3point 专属（full 模式里长按 X 是整段录制开始）。

仿真命令（终端 A 先起 harness，终端 B 起 sender；wrapper 会拉起 PC Service）：

```bash
# A
.venv/bin/python gear_sonic/scripts/arm_teleop_v3_mujoco.py \
  --wire --intent-port 15573 \
  --ctrl-lever-left  0.0410 0.0090 -0.0912 \
  --ctrl-lever-right -0.0393 0.0115 -0.0791 \
  --record /tmp/v4_3pt.npz
# B
./gear_sonic/scripts/run_pico_teleop.sh \
  --pc2-host 127.0.0.1 --intent-port 15573 \
  --tracking 3point --no-locomotion --hand-source off
```

验收要点（完整清单见 `PICO_LIVE_TELEOP_PITFALLS.md` §11）：

1. 平移符号：双手同时朝操作员前方平移，record 列 7:14 的 x 应增大；
2. **头部隔离**：手柄不动、只转头 5 s，列 7:35 应平坦（< 2 mm / < 0.5°），
   手臂指令不随头动；
3. 单侧丢跟踪：失效侧列 4/5 → 0、65/66 → 0，该侧手臂冻结、无跳变；
4. 扣 trigger：列 2/3 上升，viewer 里 OmniHand 闭合（v3 极性）。

> **已知限制**：没有躯干/腿跟踪。固定帧只标定一次，engage 期间转体会给前伸方向带来
> 同样角度的偏航偏差——保持髋/脚不动；转身后松开 grip、恢复朝向、重启 sender 重新
> 标定。控制器的单侧静默冻结无法检测（SDK 无 per-controller 可用性 API），此时手臂
> 只是保持（delta=0），不会跳变。

## Robot

Ritual up ([`F10_real_robot_colcon.md`](F10_real_robot_colcon.md)) with
`X2_WHOLE_BODY_TELEOP=1` in `x2_pc2/robot_env.env` (the default; the overlay
drives only while the laptop stream is fresh and the operator is engaged,
otherwise the session is a plain pad session). Then, on the laptop:

```bash
./gear_sonic/scripts/run_pico_teleop.sh --pc2-host <PC2_IP>
./gear_sonic/scripts/run_pico_teleop.sh --pc2-host <PC2_IP> --cam orbbec --cam-preview   # + robot camera in the headset
```

Health check before engaging (want intent 50 Hz + robot-state 50 Hz, no `[NO ...]` flags):

```bash
ssh run@<PC2_IP> 'tail -2 ${PC2_PREFIX:-/home/run/gear-sonic}/log/pico_token_service.log'
```

`--cam <orbbec|stereo_left|stereo_right|head_front>` needs the PC2 camera
bridge (`gear_sonic_deploy/scripts/x2_pc2_camera_zmq_publisher.py --streams <key>`,
installed as a service by `x2_pc2/install_camera_bridge_service.sh`); on the
headset open Camera -> **Listen** when the script says so. Stop: Ctrl-C the
sender (the robot drops to the kplanner / pad within one tick); the pad
e-stop is always live.

## Recording a session

During a live `run_pico_teleop.sh` session hold **X** >= 0.8 s to start and
**Y** >= 0.8 s to stop; the sender writes `record_<stamp>.npz` into its
`--record-dir`, plus one `session_<stamp>_x2_segNN.npz` per whole-body engage
window. Replay them with [`F05_pico_tape_replay.md`](F05_pico_tape_replay.md).
Record for robustness: stick pushes >= 3 s, a pause between a turn and a
push, a real stop before engaging whole-body.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `--input-source pico needs .venv_teleop` | `bash install_scripts/install_pico.sh` |
| `xrt.init()` connects but everything is zero | headset asleep / app backgrounded / Full-body mode off / other Wi-Fi |
| `body: False` forever with trackers connected | calibration lost (button reads "Calibrate", not "Recalibrate"): recalibrate |
| only 2 of 3 trackers pair | Motion Tracker "max tracker usage" defaults to 2 — set 3 first |
| `ENGAGE REFUSED` | body solve frozen/dead: recalibrate, keep the app foregrounded |
| sender refuses: another sender targets this host | `run_pico_teleop.sh` and `run_pico_replay.sh` are exclusive; Ctrl-C the other one |
| `RoboticsServiceProcess: error while loading shared libraries` | do not start it by hand; the launcher sets `LD_LIBRARY_PATH` |
| `MODEL=` is a g1 planner graph on the standalone path | the SMPL path needs `*smpl*.onnx`, a `.pt`, or an `frozen-core-smpl:` composite |
