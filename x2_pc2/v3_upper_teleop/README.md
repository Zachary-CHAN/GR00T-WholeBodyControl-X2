# X2 v3 增量上身遥操 —— PC2 真机节点

> 目标：把 MuJoCo 里验证过的 **v3 链**（4-DoF 位置 IK + 腕部直接填充 +
> controller-origin lever）原样搬到真机，先验证「只转手腕不再拖肘/肩」。
>
> 本目录是**本机 staging**：节点源码在这里，`build_bundle.sh` 把共享链、
> DLS 求解器、FK 和 HAL 表复制成一个扁平包；用户把包里的 6 个 `.py`
> scp 到 PC2 的 `czq_examples` 包目录即可。**不覆盖 v2 节点**，v2 仍可回退。

---

## 0. v3 与 v2 的操作差异（先读）

| | v2 节点 | **v3 节点** |
|---|---|---|
| 离合 | 扳机（trigger） | **握把（grip）** |
| 手指通道 | 握把 | 扳机（节点**不发布手**，故真机上无效） |
| 手臂 IK | 7-DoF pose IK | **4-DoF 位置 IK（肩 3 + 肘）+ 腕部 WristClutch 直填** |
| 腰 | 跟随操作者躯干扭转 | **固定 `--waist-stand`（默认 0）** |
| SMPL | waist 使用 | **完全忽略** |
| 松开 / 断流 | 手臂保持 | 手臂保持（语义不变） |
| record | 82 列 | **85 列（与 v3 sim harness 完全一致；qpos 70:85 为 NaN）** |

操作员口令：**捏握把 = 该臂咬合；松开 = 保持；扳机不控制手臂。**
第一次真机测试建议同时停掉 CLAW 手桥（或 sender 加 `--hand-source off`），
避免旧的「握把→手」通路和新的「握把→离合」同时动作。

---

## 1. 本机验证（不接机器人）

```bash
cd /home/nvidia/Documents/czq/GR00T-WholeBodyControl-X2

# ① 打包（默认 dist/；会复制 5 个共享文件 + HAL 表 + 本 README）
PYTHON=$PWD/.venv/bin/python SELF_TEST=1 \
  x2_pc2/v3_upper_teleop/build_bundle.sh

# ② 只跑节点 + 扁平依赖（不经过仓库 import），确认 PC2 布局在本机也能跑
.venv/bin/python x2_pc2/v3_upper_teleop/dist/incremental_upper_teleop_v3.py \
  --self-test --backend dls

# ③ 无 sender 的 dry-run：只验证 50 Hz 循环、stale 保持、record 宽度
.venv/bin/python x2_pc2/v3_upper_teleop/dist/incremental_upper_teleop_v3.py \
  --dry-run --intent-port 15573 --record /tmp/v3_dry.npz --duration 2

.venv/bin/python - <<'PY'
import numpy as np
r = np.load("/tmp/v3_dry.npz")["rows"]
print("rows", r.shape, "expected (N, 85)")
assert r.shape[1] == 85
assert np.isnan(r[:, 70:85]).all()
print("fresh", r[:, 1].mean(), "-> stale arms hold is expected")
PY
```

期望自检：`[self-test] all checks: 8/8`（链内 17 项汇总为 1 项，外加 7 项
HAL / record / 接线检查）。`--backend dls` 必须显式给：PC2 没有 mink，
链的纯检查默认会用 mink，不设会 `RuntimeError`。

### 1.1 可选的 tape 端到端（本机）

`--profile wrist` 的 tape 是纯四元数扫掠、位置恒定，适合先验证**腕轴方向**。
**该 tape 用零 lever**（它的控制器位置是常数，加 lever 反而会制造平移目标）。

```bash
# 生成 tape（当前版本：grip 是离合，trigger 是 hand）
.venv/bin/python tools/gen_v4_intent_tape.py /tmp/v4_tape_wrist.npz --profile wrist

# T1：节点 dry-run（dls）。duration 要覆盖 sender 启动（约 2 s）+ tape 24 s
.venv/bin/python x2_pc2/v3_upper_teleop/dist/incremental_upper_teleop_v3.py \
  --dry-run --intent-port 15573 --record /tmp/v3_node_tape.npz --duration 45

# T2：sender 回放。--tape-once 不循环 + --tape-exit-s 2 让 sender 自己退出；
#     缺了 --tape-once 会一直循环，只能 timeout 杀掉（exit=124 不是故障）
.venv_teleop/bin/python gear_sonic/scripts/pico_intent_sender.py \
  --pc2-host 127.0.0.1 --intent-port 15573 \
  --tape-replay /tmp/v4_tape_wrist.npz --tape-once --tape-exit-s 2 \
  --hand-source off --no-locomotion
```

看节点日志（sender 启动约 2 s，窗口整体后移）：约 5–14 s 左臂 `eng=E`、
约 17–23 s 右臂 `eng=E`；腕 `wristL/R` 的 yaw/pitch/roll 应分别在对应窗口
变化，且**肩 3 + 肘基本不动**。本机实测左臂窗口肩肘 p2p ≤ 0.0045 rad。
（tape 的合成姿态只用于离线方向检查；真机 live 的手感以实测为准。）

---

## 2. scp 到 PC2

PC2 包目录（v2 文档实测路径）：

```
~/Workspaces/czq/ros2_ws/src/czq_examples/czq_examples/
```

laptop 上执行（host 别名按你的 ssh config；也可用 `run@10.0.1.41`）：

```bash
cd /home/nvidia/Documents/czq/GR00T-WholeBodyControl-X2
B=x2_pc2/v3_upper_teleop/dist

# 先备份会被覆盖的三个既有文件（x2_arm_ik.py 也要备份，v2 也用它）
ssh Agibot-X2-PC2-Orin-NX \
  'D=~/Workspaces/czq/ros2_ws/src/czq_examples/czq_examples; \
   for f in upper_body_teleop.py x2_arm_fk.py x2_arm_ik.py; do \
     cp -a $D/$f $D/$f.bak.$(date +%s); done'

scp $B/incremental_upper_teleop_v3.py \
    $B/incremental_chain_v3base.py \
    $B/incremental_chain_v3.py \
    $B/x2_arm_ik.py \
    $B/x2_arm_fk.py \
    $B/upper_body_teleop.py \
    Agibot-X2-PC2-Orin-NX:~/Workspaces/czq/ros2_ws/src/czq_examples/czq_examples/

# 校验
ssh Agibot-X2-PC2-Orin-NX \
  'cd ~/Workspaces/czq/ros2_ws/src/czq_examples/czq_examples && sha256sum incremental_upper_teleop_v3.py incremental_chain_v3base.py incremental_chain_v3.py x2_arm_ik.py x2_arm_fk.py upper_body_teleop.py'
cat $B/SHA256SUMS
```

> **不要覆盖 PC2 上已有的 `incremental_chain.py`。** 那是 v2 节点的基座链，
> 仍是旧的 trigger→离合 / grip→手指语义；仓库里的同名文件已经做过角色互换。
> v3 的基座链以 `incremental_chain_v3base.py` 独立部署，节点和 v3 链都优先
> 导入这个名字，因此 v2 节点可以原样回退。
>
> **`x2_arm_fk.py` 这次必须一起换。** 它包含右臂腕部几何修正（旧扁平副本
> 用的是左臂数值，右腕偏 1.22 mm）。v3 的 4-DoF 解和 ZYX 腕映射都以这份 FK
> 为准；新旧混用会让 sim/robot 的 FK 不一致。该修正也会影响 v2 节点，属于
> 预期行为；要回到纯旧 v2，用备份还原。`x2_arm_ik.py` 的新版只新增了
> `solve_position_free4` / `posture_gain`，旧 API 全部保留，v2 兼容。

---

## 3. PC2 安装 / 自检（一次性）

每个 PC2 窗口先 source 三连：

```bash
source /opt/ros/humble/setup.bash
source /home/agi/aimdk/install/setup.bash
source ~/Workspaces/czq/ros2_ws/install/setup.bash
```

### 3.1 先免 colcon 冒烟（推荐先做）

```bash
cd ~/Workspaces/czq/ros2_ws/src/czq_examples/czq_examples
python3 incremental_upper_teleop_v3.py --self-test --backend dls
# 期望 all checks: 8/8（不需要 rclpy/ROS）
```

### 3.2 加 `ros2 run` entry point

```bash
cd ~/Workspaces/czq/ros2_ws/src/czq_examples
grep -q incremental_upper_teleop_v3 setup.py || python3 - <<'EOF'
import pathlib
p = pathlib.Path("setup.py"); s = p.read_text()
anchor = '"incremental_upper_teleop = czq_examples.incremental_upper_teleop:main",'
assert anchor in s, "v2 entry not found -- edit setup.py by hand"
entry = '\n            "incremental_upper_teleop_v3 = czq_examples.incremental_upper_teleop_v3:main",'
if "incremental_upper_teleop_v3" not in s:
    p.write_text(s.replace(anchor, anchor + entry))
print("v3 entry added")
EOF

cd ~/Workspaces/czq/ros2_ws          # ★ 绝不在 ~/Workspaces/czq 顶层跑 colcon
colcon build --packages-select czq_examples

source /opt/ros/humble/setup.bash
source /home/agi/aimdk/install/setup.bash
source install/setup.bash
ros2 pkg executables czq_examples | grep incremental
ros2 run czq_examples incremental_upper_teleop_v3 --self-test --backend dls
```

> 包名/entry 以 PC2 实际的 `czq_examples/setup.py` 为准。如果那里仍是
> `py_examples`，把上面的 `czq_examples` 全部替换成实际包名即可；节点本身
> 不依赖包名。

---

## 4. 实验当天命令

### 4.1 PC2 节点（W-A）

```bash
source /opt/ros/humble/setup.bash
source /home/agi/aimdk/install/setup.bash
source ~/Workspaces/czq/ros2_ws/install/setup.bash

# 先 dry 20 s 看命令流（不发布 hal）
ros2 run czq_examples incremental_upper_teleop_v3 \
  --dry-run --record /tmp/v3_dry.npz --duration 20

# 正式。lever 用 2026-10 两次标定的本次活动段（若握法/人/隔天重戴，先重标）
ros2 run czq_examples incremental_upper_teleop_v3 \
  --ctrl-lever-left  0.0410 0.0090 -0.0912 \
  --ctrl-lever-right -0.0393 0.0115 -0.0791 \
  --record /tmp/v3_live.npz
```

可选旗标：

- `--full-pose-ik`：回到 7-DoF pose IK（v3 默认不是它；只有需要对照时才用）。
- `--position-deadband-mm 3` / `--posture-gain 0.01`：默认值，先别改。
- `--no-legs`：不发布 head/leg 保持命令（只发 arm + waist）。
- `--backend dls`：默认；PC2 只能 dls。
- 腕轴方向反了：`--wrist-yaw-sign-l/-r`、`--wrist-pitch-sign-l/-r`、
  `--wrist-roll-sign-l/-r` 各翻 ±1（先跑 §1.1 的 wrist tape 确认方向）。

### 4.2 laptop sender（W-C）

```bash
cd /home/nvidia/Documents/czq/GR00T-WholeBodyControl-X2
./gear_sonic/scripts/run_pico_teleop.sh --pc2-host 10.0.1.41 --hand-source off
```

- **不要加 `--no-locomotion`**：它会关掉 sender 的控制器层，握把/扳机字段
  恒 0 → v3 永远无法咬合（v2 文档踩过同一个坑）。
- `--hand-source off`：不建 sender 的手部通道，避免 CLAW 桥用旧的
  「握把→手」语义和新的「握把→离合」打架。本节点本来也不发布手。
- 同一目标 host 只允许一个 sender；重复起会被脚本拒绝。

### 4.3 当天顺序（与 v2 相同）

1. 吊架吊住、机器人悬空、周围净空；两人到位。
2. PC2 查状态 = `Ready`；`ss -tlnp | grep :5573` 起节点前应为空。
3. `ros2 run py_examples migrate_system_state Develop_MC` 瘫软（确认吊架吊住）。
4. W-A：先 `--dry-run` 20 s，再正式起节点。
5. W-B：`ss -tlnp | grep :5573`；`ros2 topic echo /aima/hal/joint/arm/command --once`。
6. W-C：起 sender；**捏握把**咬合左/右臂。
7. 先做「肩肘不动、只转腕」的圆锥/圆圈 20–30 s，看机器人上游关节是否稳定。
8. Ctrl-C sender → 0.5 s stale → 双臂保持；Ctrl-C 节点 → 停发 hal；迁回 `Ready`。

---

## 5. record 离线判读（85 列）

scp 回 laptop：

```bash
scp Agibot-X2-PC2-Orin-NX:/tmp/v3_live.npz /tmp/
.venv/bin/python - <<'PY'
import numpy as np
rows = np.load("/tmp/v3_live.npz")["rows"]
print("rows", rows.shape)          # (N, 85)
for side, eng, qs, ws in (("L", 65, (35, 39), (49, 52)),
                          ("R", 66, (42, 46), (52, 55))):
    m = rows[:, eng] > 0.5          # ★ 65/66 = engaged；63/64 是冻结保护
    if not m.any():
        print(side, "never engaged"); continue
    seg = rows[m]
    # 跳过咬合后 0.4 s 的瞬态
    seg = seg[min(int(0.4 / 0.02), len(seg) - 1):]
    print(side, "engaged", int(m.sum()), "ticks",
          "q[sh3+elbow] p2p rad", np.round(np.ptp(seg[:, qs[0]:qs[1]], axis=0), 4),
          "wrist cmd p2p deg", np.round(np.degrees(np.ptp(seg[:, ws[0]:ws[1]], axis=0)), 1))
    t = seg[:, 0] - seg[0, 0]
    for j, name in enumerate(("sh_pitch", "sh_roll", "sh_yaw", "elbow")):
        q = seg[:, qs[0] + j]
        print(f"   {name}: start {q[0]:+.4f} end {q[-1]:+.4f} "
              f"drift {np.polyfit(t, q, 1)[0] * t[-1]:+.4f} rad")
PY
```

- **列 65/66 才是 `engaged`**。列 63/64 是冻结保护（solver cap / NaN 时锁存），
  正常运行时恒 0，不要拿它当离合窗口。
- `wrist_tgt`（列 49:52 / 52:55）是**三个腕关节角的命令（rad）**，不是末端
  位置；4-DoF 路径等于 WristClutch 直填，full-pose 路径是 IK 解出的腕角。
  纯腕旋转时它本来就应跟着扫（例如 120° yaw），不是抖动指标。
- `q` 前 4 个是肩 3 + 肘；纯腕动作时 p2p 最好 **< 0.02 rad**；
  0.02–0.05 rad 可来自 posture/deadband 的慢漂，不应是随腕旋转的周期性联动。
  本机 wrist tape（120° yaw + 60° roll）实测 ≤ 0.0045 rad，可作基线。
- 后 3 个 q 是腕关节，不参与「上游是否被拖」的判据。
- 列 2/3 = 原始 trigger（hand 输入）；列 4/5 = LPF 后 grip（离合）；
  列 55:61 = 控制器腕 delta（yaw/pitch/roll，rad）。
- 若肩肘仍随腕旋转周期性动：优先怀疑 lever（重新标定），
  其次检查 sender 的控制器姿态是否在纯腕旋转时带了平移。

---

## 6. 已知差异 / 风险

1. **后端不同**：MuJoCo v3 harness 默认 `mink`，PC2 只有 `dls`。两者跑的是
   同一条 v3 tick 链、同一组守卫，但求解器实现不同（QP vs DLS）。PC2 的
   `--self-test --backend dls` 必须全绿；首跑仍按新模型上吊架。
2. **腕轴符号未在真机验证**：`DEFAULT_WRIST_SIGNS` 来自 v1 + 镜像推导。
   先用 §1.1 的 wrist tape（零 lever）看方向，再自由操作。
3. **节点不发布手指**：v3 的 `hand`（trigger）只在 record 里可诊断；真机手
   仍走 PC2 的 CLAW 桥，与节点无关。本测试建议 `--hand-source off`。
4. **`x2_arm_fk.py` 右臂修正**：见 §2；会影响 v2。备份已做。
5. **lever 取决于握持**：换握法、换人、隔天重戴都要重标
   （`tools/fit_pico_ctrl_lever.py`，方法见
   `docs/x2/PICO_LIVE_TELEOP_PITFALLS.md` §9）。上面的 0.0410/… 是
   2026-10 本次活动段的增量估计；不要跨握法复用。
6. **stale 语义**：v3 默认 stale 时内部 hand → 0（张开），但节点不发手；
   手臂始终 HOLD。`--stale-grip-close` 只影响内部 hand 通道，对真机手臂
   无影响。
7. **腕关节行程不对称**（`X2_ARM_JOINT_LIMITS_RAD`）：yaw ±146°，pitch
   只有 ±32°，roll 左 [-90°, +41.5°] / 右 [-41.5°, +90°]。大幅腕动作会先
   顶到 pitch/roll 限位并被裁剪，这是关节限位而不是 IK 失效；测试时先小
   幅度确认方向，再逐步加大。本机 wrist tape 的 51.6° pitch 扫掠就被右腕
   ±32° 限位截断（记录里 wrist cmd p2p 只有 32°）。

---

## 7. 回退

- v2 节点文件未被修改；v2 的基座链 `incremental_chain.py` 也**未被覆盖**，
  所以 `ros2 run czq_examples incremental_upper_teleop` 仍按 v2 语义
  （扳机离合、握把手）运行。v3 新增的 `incremental_chain_v3base.py` /
  `incremental_chain_v3.py` 不会被 v2 节点导入。
- 若要把 `x2_arm_fk.py` / `x2_arm_ik.py` / `upper_body_teleop.py` 还原到
  scp 前：
  ```bash
  cd ~/Workspaces/czq/ros2_ws/src/czq_examples/czq_examples
  mv x2_arm_fk.py.bak.<timestamp>      x2_arm_fk.py
  mv x2_arm_ik.py.bak.<timestamp>      x2_arm_ik.py
  mv upper_body_teleop.py.bak.<timestamp> upper_body_teleop.py
  ```
  然后 `colcon build --packages-select czq_examples`（或直接 python 运行时
  无需 build）。v3 的 3 个文件可以直接留着，不影响 v2。
