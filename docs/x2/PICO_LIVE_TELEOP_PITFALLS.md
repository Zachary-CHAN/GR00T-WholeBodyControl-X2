# Pico live 遥操踩坑记录（2026-09-30）

> 这不是 runbook，是一次 live 会话的排查记录。操作步骤看
> [`F03_pico_teleop.md`](F03_pico_teleop.md)，导航看 [`../../CLAUDE.md`](../../CLAUDE.md)，
> 已验收条目在 [`../../x2_repro_checklist.html`](../../x2_repro_checklist.html)。
>
> 这里留下的是：**判据、被证伪的假设、以及下次能直接用的命令**。
> 当天根因是**头显没电**——但排查过程中否掉的三个假设同样值钱，所以都留着。

---

## 1 机器人不动：先看三道门，别猜

`run_pico_teleop.sh` 起不来动作，是三个**独立**的门，逐个排：

| # | 门 | 在哪 | 判据 |
|---|---|---|---|
| 1 | 模式机 | 你按的键 | 必须在 `WHOLE_BODY`；进入即 engage |
| 2 | 体感新鲜度（sender） | `pico_intent_sender.py:1207+` | `body_fresh()`：**1.2 s 内有新帧** 且 `_same_streak < 50` |
| 3 | 体感新鲜度 + engaged（deploy） | deploy 启动横幅 | `< 0.50 s` **且** engaged（warm-up 1.0 s，crossfade 0.40 s） |

最省事的观测点是 sender 那一行状态（每 2 秒刷一次，照抄 `pico_intent_sender.py:1433-1438`）：

| 状态 | 打印的那一行 |
|---|---|
| 健康 | [intent] 50.0 Hz  disengaged  [body OK (last new frame 0.0s ago, identical-streak 0)] |
| 冻结 | [intent] 50.0 Hz  disengaged  [body FROZEN/DEAD — engage will refuse; recalibrate in headset (last new frame 7.0s ago, identical-streak 0)] |

**`identical-streak` 和 `age` 是两种不同的冻结，要分开读：**

- `streak` 大 ⇒ Pico 在**反复喂同一帧**（cached pose re-served）。
- `streak 0` + `age` 大 ⇒ **stamp 根本没在变**，新帧压根没来。

deploy 侧一条 grep 就够（sender 和 deploy 都会打，含这个子串）：

```bash
grep 'WHOLE-BODY ENGAGED' /home/nvidia/.cache/sonic/simstack_pc/logs/deploy.log
# 0 次 = sender 从来没让它 engage 过，问题在 sender 之前，不在 deploy
```

---

## 2 ★ 头显没电的形状（当天根因）

**认出这个形状，能省掉一整轮软件排查：**

```
device found PA94...
  → 数据断断续续，停顿 越拖越长   7.15s → 22.66s → 16.88s
  → device missing
  → 又打一次 device found
  → 最后彻底 device missing
```

**停顿越来越长 + `device missing`，第一件事查电量。** 当天我先怀疑了三个软件原因（见第 3 节），全被数据打掉。

顺带：**电量耗尽期间的 `ping` 结果毫无意义**。当天对着已关机的头显 ping 到 100% 丢包，差点被当成路由故障的证据。

---

## 3 被证伪的假设（都别再查了）

| 假设 | 怎么否掉的 |
|---|---|
| **桌面遮挡**（坐着时踝部 tracker 被桌子挡住） | 探针显示 `root tilt` 坐直时只有 **2.8~6.8°**，骨架连贯、head→ankle ≈ 0.85~1.18 m，不是退化解算。遮挡顶多掉一个 tracker，掉不到整个解算停 |
| **标定基准偏了** | 同上——基准偏了的话 `root tilt` 会稳定停在 25° 以上。实际是 2.8~6.8°，基准是准的 |
| **PC Service 被 CPU 饿死** | 停顿期间实测：load **5.30/14 核**、`RoboticsService` 只 **1.7% CPU**、内存 82 GB 可用。它是**在等**，不是在算 |

**同一条教训出现三次**：我用代理量（CPU%、计数器、几何推断）去推结论，每次都被直接量打掉。**能直接量的就别推。**

---

## 4 控制语法：和弦极易按反

```
A+B+X+Y            主开关，唯一作用 = OFF <-> LOCOMOTION
按住右扳机(>=0.6) + B    LOCOMOTION <-> WHOLE_BODY（进入即 engage）
```

两个坑：

1. **`A+B+X+Y` 只管 `OFF <-> LOCOMOTION`。** 已经在 LOCOMOTION 时再按它，会把你打回 OFF
   （日志上是 `MODE LOCOMOTION -> OFF`，方向反了）。当天第一次就是这么错的。
2. **裸按 B 无效。** 默认 `--wb-toggle-gate right-trigger`（`pico_intent_sender.py:1108`），
   不按右扳机的话 B 被丢弃并打印 `B ignored — hold the RIGHT TRIGGER...`。
   和弦会把 B 一起吃掉（和弦是独立事件，不受这个门控管），所以按和弦时看不到这条提示。

---

## 5 环境事实（会用到的）

### 5.1 音频提示是好的 —— 别摘头显读终端

`gear_sonic/data/audio/` 下 19 个语音片段齐全，`/usr/bin/aplay` 也在。Pico 是开放式耳塞，
提示从笔记本喇叭出来能听见：

| 事件 | 片段 |
|---|---|
| 进入 WHOLE_BODY 且 engage 成功 | `cue_mode_whole_body.wav` |
| engage 被拒 / 断流 | `teleop_disengaged.wav` |
| 裸按 B 忘了扳机 | `cue_b_needs_trigger.wav` |

**这很重要**：确认有没有 engage 成功要摘头显去看终端，而**摘头显本身就让头显脱离追踪、制造断流**——
一个自己给自己造失败的循环。用耳朵听，循环就断了。

### 5.2 `trackers: 0` 是红的，与 X2 无关

F03 的探针写「want: trackers: 2 (or 3)」，那是给 **XRoboToolkit 的 G1 下肢 IK demo** 写的。
X2 全程只用 SDK 五个函数：

```
is_body_data_available / get_time_stamp_ns / get_body_joints_pose / get_left_grip / get_right_grip
```

全仓库（`gear_sonic/` + `gear_sonic_deploy/`）**没有一处**用 `num_motion_data_available`
或 `get_motion_tracker_*`。所以 `trackers: 0` 不影响 X2，别去追。

### 5.3 `stamp: 0` 会永久卡死 append 门

append 的门是（`live_pico_smpl_teleop.py:154`）：

```python
if stamp and stamp != self._last_stamp and self.xrt.is_body_data_available():
```

`get_time_stamp_ns()` 返回 `0`（falsy）时这行**永远不成立** → 一次都不 append →
`_last_append_t` 停在 0 → `body_fresh()` 第一行 `if self._last_append_t <= 0.0: return False` 直接判死。
**症状和「帧冻住」一模一样，机制完全不同。**

### 5.4 坐姿不是故障，但会被大量钳制

坐着时 `[tiltcap]` 的 torso 列会 **100% 超限**（每 5000 帧涨满 5000），`ramp` 恒为 1.00：

```
[tiltcap] > 25 deg: root capped on 7755/30000 frames (25.85%), torso on 29980 (99.93%), ramp 0.6s (now 1.00)
```

这是**守卫在正常干活**（骨盆直立、脊柱链持续前倾 = 坐着看屏幕），不是标定坏了。
但要清楚：**这种状态下 engage，机器人拿到的每一帧都是被钳到 25° 的命令**，
别拿它判断策略跟随质量——那个要站着测。

### 5.5 双网卡同子网会打掉 `--cam`

```
<laptop-wlan0-ip>  wlan0   metric 600
<laptop-eth8-ip>   eth8    metric 103   ← 去同一子网的流量走这块
```

两块网卡同在一个子网，内核挑 metric 小的 **eth8**，而头显挂在 **wlan0** 那侧
（ARP 在 wlan0 上解析得出，eth8 上是 `INCOMPLETE`）。

- **不影响 body 流**：那些连接是头显**主动**连进来的，本地端锁在 wlan0 的地址上，回包自然走 wlan0。
- **会打掉 `--cam`**：视频发送端是 PC **主动**拨头显 `:12345`，会发到 eth8 上打空。

---

## 6 复现用的探针

镜照 append 门本身（50 Hz 轮询），打印 **append 间隔**和 **root tilt**。
`root tilt` 用的是守卫同一个量（`root q` 取 `[6,3,4,5]` → `apply([0,1,0])[:,1]` → `arccos`，设备帧 up = +Y），
可直接和 25° 上限对比。

> **前提：先把 sender 停掉。** 并发 `xrt.init()` 会互相干扰，读数不可信。

```python
# .venv_teleop/bin/python probe.py [seconds]
import time, numpy as np, xrobotoolkit_sdk as xrt
from scipy.spatial.transform import Rotation as sRot

xrt.init(); time.sleep(1.0)

prev_stamp, prev_body, last = 0, None, time.monotonic()
t0, t_next = time.monotonic(), time.monotonic() + 5.0
n = 0
while time.monotonic() - t0 < 60:
    now = time.monotonic()
    stamp, avail = xrt.get_time_stamp_ns(), xrt.is_body_data_available()
    if stamp and stamp != prev_stamp and avail:
        body = np.asarray(xrt.get_body_joints_pose(), np.float64)
        if body.shape == (24, 7):
            gap = now - last
            if n and gap > 0.3:
                print(f"[{time.strftime('%H:%M:%S')}] GAP {gap:.2f}s")
            last, n, prev_body, prev_stamp = now, n + 1, body, stamp
        prev_stamp = stamp
    if now >= t_next and prev_body is not None:
        t_next = now + 5.0
        up = sRot.from_quat(prev_body[0][[6, 3, 4, 5]], scalar_first=True).apply([0., 1., 0.])
        tilt = np.degrees(np.arccos(np.clip(up[1], -1., 1.)))
        print(f"[{time.strftime('%H:%M:%S')}] appends {n:5d}  age {(now-last)*1000:6.0f} ms  "
              f"root tilt {tilt:5.1f} deg  head-ankle {prev_body[15][1]-prev_body[7][1]:.2f} m")
    time.sleep(0.02)
```

读法：

- `age` 一直几十 ms、无 `GAP` ⇒ 健康，可以起 sender。
- `GAP` 越来越长、伴随 `device missing` ⇒ 查第 2 节（电量）。
- `root tilt` 坐直时稳定 > 25° ⇒ 才是标定问题，重标定。

---

## 7 当天其它操作坑（不限于 Pico）

- **抓日志别用 `pkill -f`**：本项目已两次匹配到自己的命令行把自己杀掉。用 `kill <PID>` 或方括号技巧。
- **grep 摔倒别用宽模式**（`FATAL|SAFE_HOLD|tilt` 会撞上启动横幅里的 `--tilt-cos \`）。
  用 `grep -cE '^\[FATAL\] \[17'`。
- **`.venv` 会被 `PYTHONPATH` 污染**：shell 里的 `/opt/ros/jazzy/lib/python3.12/site-packages`
  会让 python3.10 的 venv pytest 报 `No module named 'lark'`。用
  `env -u PYTHONPATH -u AMENT_PREFIX_PATH -u ROS_DISTRO .venv/bin/python -m pytest ...`。
- **抓 deploy 日志看 `deploy.log` / `watchdog.log`**。PC Service 的 stdout 是废的
  （`/home/nvidia/Documents/czq/logs/service.log`，只有 13 字节，内容就两个字 `release mode`）；
  sender 的 stdout 直接进 pty，读不了。
- **「改了 IK 之后 sim 和真机不一样」→ 先看后端，再看行为**。增量上半身链的手臂求解器
  由 `X2_IK_BACKEND`（或 `arm.backend`）决定：sim 默认 `mink`，真机（PC2）仍是 `dls`，
  两侧本来就不同（[`F03_pico_teleop.md`](F03_pico_teleop.md) 的「手臂 IK 后端」）。
  harness 横幅会打印求解器与自由度（默认形如 `arm IK mink` + `4-DoF`；加
  `--full-pose-ik` 才是 `7-DoF`）；拿它对齐口径，别拿 sim 的数字推断真机行为。
- **「改了 trigger/grip 之后 sim 和真机反了」→ 先看运行侧**。2026-10 起 sim 的增量链
  把 **grip 当 clutch、trigger 当手指**；PC2 平铺副本未同步，仍是旧的 trigger clutch /
  grip 手指。读 record 列 2/3（原始 trigger）与 4/5（滤波 grip），或看节点横幅（仍写
  `trigger>... clutches`）时，先确认是 sim 还是真机。
- **live `--no-locomotion` 下 trigger 恒 0（2026-10-20 已修）**：sender 以前只在
  `dual_mode` 块里调用 `_poll_controllers`，非 dual 路径把 `lt=rt=0` 直接写进 wire；
  sim record 列 2/3 全 0，手永远不闭合。现在非 dual 的 **live** 源会在主循环里直接读
  `get_left_trigger()/get_right_trigger()`（`_read_trigger_pair`），tape 源仍用
  `replay_ctrl`。注意这不等于真机手能动：`--no-locomotion` 下 sender 仍不创建
  `hand_sock`，真机手指必须走 §4.2 的 `--hand-source trigger --hand-close-on-grip`
  （不加 `--no-locomotion`）。

---

## 8 同日的前半程：根朝向守卫（已回写，仅索引）

当天早些时候的另一条线是 tape 回放的摔倒与命令侧守卫，结论已进代码注释和
[`../../x2_repro_checklist.html`](../../x2_repro_checklist.html)。两条最反直觉的留在备忘：

1. **加了保护可能更糟。** memoryless 软钳制（cap 25°）把摔倒从 tape_t 219.96 提前到 **78.52**，
   提前了 **141 秒**——它把一段 0.30 s 的无害毛刺（根峰 34.25°）全量钳掉、绕 ROLL 转了 6.49°。
   **任何单一阈值都救不了**：放过毛刺要 cap > 34.25°，兜住真事件要 cap < 30°，**区间为空**。
   区分因子是**持续时长**，所以改用软启动 ramp（默认 `--root-tilt-ramp-s 0.6`）。
   **加保护必须严格 A/B，不能假定它只会变好。**
2. **默认值可能和验证过的值不一致。** `ROOT_TILT_RAMP_S` 曾发布为 `0.5`，而设计、A/B 验证
   （`/tmp/x2stack/stamp_run.sh` 显式传 0.6）和清单写的都是 `0.6`。已改为 0.6。
   理由写进了代码注释：**缩短 ramp 是朝 memoryless 那端走，而摔的就是那一端**——更狠的钳制才是危险方向。

---

## 9 只转手腕却拖肘/肩：Pico 控制器原点 ≠ 人腕心

**症状**：人的肩、肘不动，只转手腕（控制器），机器人却抬肘、甚至动肩；把柄
平移时反而正常。

**根因**：Pico 的 controller pose 是**控制器跟踪原点** `C`，不是握持点/腕心 `W`。
两者差一根固定在控制器坐标系里的杠杆 `r`：`C = W + R @ r`。腕心不动、只转
控制器时，`C` 会沿半径 `|r|` 的弧线扫动；旧目标
`tgt = a_rob + scale*(C - a_op)` 把这根弧当成平移，4-DoF IK 只能把位移摊给
肩/肘。

**修复**：v3 链的 `--ctrl-lever-left/right X Y Z`（米，控制器坐标系）做
`W = C - R @ r`，锚点也扣同一项；纯腕转时补偿后的位置目标保持不变。
`--ctrl-lever` 仍是两臂共用的回退值，默认 `0 0 0` 是旧行为——**必须标定**。

**标定步骤**（v3 sim / live wire 模式）：

1. **两个终端都要起**。只跑 harness 时 `--record` 里控制器列全是 0
   （`fresh=0`），拟合会报「no window with >= 20 deg」。先起 harness：
   ```bash
   .venv/bin/python gear_sonic/scripts/arm_teleop_v3_mujoco.py --wire \
     --record /tmp/lever_fit.npz --no-viewer --duration 30
   ```
   再起 sender（头显已连接、数据 ON），然后做「**腕心尽量不动、控制器画圆锥/圆圈**」
   的动作：同时绕 yaw+pitch+roll 转，累计 ≥ 60°。**不要只沿一个轴来回转**——
   绕单轴的旋转云是秩 2 的，沿该轴的杠杆分量永远不动，拟合只能拿腕部平移去填；
   工具用 `--min-observability 0.3`（旋转云最小与最大奇异值之比）把这种
   窗口和 lag 直接拒掉。最有效的是**把前臂/肘支在桌上，只让手腕画圈**。

   **sender 仍以 body 流为发布门**（`LiveSmplSource` 要
   `is_body_data_available()` + 新 stamp），所以 v3 链不用 SMPL **不等于**
   可以只开手柄：头显需 Full body 模式、追踪器已配对并完成校准，否则会一直
   停在 `Waiting for body stream ...`。`--hand-source off` 不会绕过这个门。
   ```bash
   .venv_teleop/bin/python gear_sonic/scripts/pico_intent_sender.py \
     --pc2-host 127.0.0.1 --intent-port 15573 --hand-source off
   ```
2. 拟合（短滞后增量估计 + 滑窗一致性检查）：
   ```bash
   .venv/bin/python tools/fit_pico_ctrl_lever.py --record /tmp/lever_fit.npz --side left
   ```
   输出三组估计：`best window`（单窗常数 `W`）、`consensus`（多窗共享 `r`）、
   `increment estimate`（短滞后差分 `dC = dR @ r`，默认 lags 2/5/10/20）。
   **用 `re-run with:` 打印的向量**——它取的是 increment 估计。原因：单窗/共识
   都假设整个窗口内腕心不动；live 标定几乎总会以几 cm/s 漂移，漂移会被吸进
   `r`，即使 RMS 只有 1–2 mm、窗口宽度也稳定，向量仍可能错 2–3 cm。短滞后差分
   把漂移差掉，几个 lag 的中位数 + `lag spread` 才是诚实度检查。

   读法：
   - `lag spread` 几 mm、`median pair residual` ≤ 2–3 mm ⇒ 可信；
   - `median pair residual > 5 mm` ⇒ 标定时腕子在动，increment 比窗口法稳，
     但仍建议重录；
   - `increment` 与 `consensus` 相差 > 15 mm ⇒ 正是腕部漂移污染窗口法的特征，
     以 increment 为准；
   - `|r|` 正常约 90–110 mm；> 150 mm 是平移污染，不是握持杠杆。

   左右手柄各做一次。**左右向量应近似镜像**（x 反号，y/z 接近，例如
   左 `0.0449 0.0070 -0.0834`、右 `-0.0451 0.0042 -0.0859`）；若右手的 x 与
   左手同号，或 `|r|` 差很多，说明其中一侧拟合坏了，不要用。
   若报「no ... window」，先用
   `.venv/bin/python -c "import numpy as np; r=np.load('/tmp/lever_fit.npz')['rows']; print(r[:,1].mean(), np.abs(r[:,24:28]).max())"`
   确认 `fresh` 非 0、四元数非 0——否则是 sender/头显没起来，不是动作幅度不够。
3. 用打印出的 `--ctrl-lever-left` / `--ctrl-lever-right X Y Z` 重跑同一动作
   （两侧都传；只传 `--ctrl-lever` 会把同一个向量给两臂，实测左右 x 分量
   可差 ~60 mm），确认 TUI 里 `q_left_cmd` 的肩 3 + 肘（record 列 35:39）基本不动，
   再上真机。

**注意**：

- 杠杆是**控制器坐标系**的常向量，取决于**握持方式**：换握法、换人、或隔天
  重新戴上控制器，都应重标；不要把一次标定的向量跨握法复用。
- 左右手柄的握持方向通常不同，分别标定、分别用 `--ctrl-lever-left/right` 传入。
  `--ctrl-lever` 只是两臂共用的回退。
- 旧 record / v4 tape 里的 controller pose 仍是原始控制器原点语义，没有变；
  只有补偿后的目标变了，record 列 0:35 不需要重解释。
- 补偿后残差（几 mm 的手抖/杠杆误差）再用 `--position-deadband-mm 3` 压掉；
  零空间 `--posture-gain` 只负责在 4-DoF 的 1-D swivel 上把肘部姿态拉回
  engage 时的构型，不改变腕点。

---

## 10 腕部只转一个 DOF，机器人却动另一个/多个关节（2026-10-20）

**症状**：操作者只做一个解剖方向的腕动作（例如屈伸），机器人响应的却是另一个
腕关节，或者 yaw/pitch/roll 同时动；松手重新咬合后方向又变。

### 10.1 先认清机构：yaw 在前，pitch/roll 是十字轴对

X2 腕链（URDF `x2_ultra.urdf`、MJCF `x2_ultra.xml`、厂商 RL deploy 的
x2.xml 三处一致，HAL 限位表也按这组角定义）：

```
elbow -> wrist_yaw (axis z, 近端) -> wrist_pitch (axis y) -> wrist_roll (axis x, 远端)
```

pitch 关节原点距 yaw 关节 82 mm，**roll 关节与 pitch 关节原点重合**，即
pitch/roll 两轴交于一点——这是十字轴（Cardan/万向节）对，电机在 forearm 里
并联驱动，HAL 下发/反馈的是这组 **Cardan 输出角**（厂商 RL deploy 用同一
MJCF 做 sim2real，厂商 teleop 示例也把 `wrist_pr`/`wrist_yaw` 直接写进
这三个关节槽）。

由此得到两条硬事实，决定了「一个输入 DOF 应该动几个关节」：

- **tool roll（绕工具自身 x 轴）**：`W_a·Rx(θ) = Rz Ry Rx(r+θ)`，任何 anchor
  下都只动 roll 关节。
- **tool pitch/yaw（绕工具自身 y/z）**：近端/中段轴与工具当前轴不重合，
  anchor 非零时**必须** pitch+roll（或三个关节）协同，否则工具朝向就不对。
  这不是映射 bug，是机构自由度排布决定的。

因此验收判据是**手（工具）的朝向**是否跟手柄一致，而不是「是否只有一个
关节在动」。

### 10.2 两版错误映射（历史）

控制器差分 `delta = R_a⁻¹ R_now` 是**控制器 anchor body 系**里的旋转
（控制器 body 轴与手刚性固连）。第一次实现用 `W_a⁻¹ delta W_a` 共轭
（`W_a = Rz(q_a)Ry(p_a)Rx(r_a)` 是腕关节角），只有 `q_a ≈ 0` 时才等于
恒等映射，非零 anchor 下单个控制器轴被拆到三个关节：

| 控制器输入 | 共轭输出 dq（anchor `[0.5, 0.2, -0.3]` rad） |
|---|---|
| 绕 x +30° | `[-2.7, -14.9, +26.3]°` |
| 绕 y +30° | `[+14.2, +20.5, +17.2]°` |
| 绕 z +30° | `[+28.6, -6.9, -7.9]°` |

第二次实现改成「关节镜像」`q_tgt = q_wa + scale*signs*euler_ZYX(delta)`，
去掉了共轭，但它在非零 anchor 下仍然不是腕链的逆：只有零 anchor 和
tool-roll 精确。20° 单轴输入的工具姿态误差（数值实验）：

| anchor `(yaw,pitch,roll)` | 工具绕 x | 绕 y | 绕 z |
|---|---|---|---|
| `(17°, 14°, -34°)` | 0.00° | 11.8° | 12.6° |
| `(0°, 30°, -30°)` | 0.00° | 10.3° | 14.1° |

共轭版同一组 anchor 下是 18–26°。

### 10.3 正确映射（现行）

精确求腕链的逆：

```
Δ        = R_a⁻¹ R_now                   # 控制器 anchor body 系
Δ_robot  = C · Δ · Cᵀ                   # C: 每侧 signs + axis_order
W(q_tgt) = W(q_wa) · Δ_robot            # W = Rz(yaw)Ry(pitch)Rx(roll)
q_tgt    = euler_ZYX(W(q_wa) · Δ_robot)
```

`scale` 作用在旋转角（rotvec）上。`C` 是控制器 body 轴到腕关节轴的常量
有符号置换（默认 `0,1,2` + 每侧默认符号 = 控制器 z→yaw、y→pitch、
x→roll）。自检 `wrist_nonzero_anchor_tracks_tool_rotation`（v3 链 + 真机
节点 `--self-test`）在非零腕/控制器 anchor 下覆盖三轴、两种轴序和默认
符号，要求工具姿态误差 < 1e-6 rad；另钉住「tool roll 只动 roll 关节」。

**默认符号左右相同（2026-10-21，逐轴符号仍在 live A/B）**：`DEFAULT_WRIST_SIGNS`
两侧始终用同一组符号。旧值左侧 `(-1,+1,-1)` 是右侧的**全局取反**（当时假定左右
工具系镜像），v4 3-point sim 第一轮表现为**左手 yaw 与 pitch 与真人转向相反**；
改为两侧 `(+1,-1,+1)` 后第二轮左手正常、但**右手 pitch 反**，因此当前把两侧都
置为 `(+1,+1,+1)` 做第三轮实测（控制器 +y→+pitch）。X2 的腕工具系并不镜像：
零位时两侧 `wrist_roll` 铰轴都是世界 +x、yaw/pitch 轴为世界 +z/+y
（`x2_ultra.xml`），所以同一 controller body 对应关系必须适用两侧；但某个轴的
符号仍可能因握法/SDK 约定需要逐侧不同。`CHAIN_V3_CHECKS[19]`
`wrist_default_signs_side_parallel` 钉住“左右相同”这一契约；逐轴用
`WRIST3_*_SIGN*` / `--wrist-{yaw,pitch,roll}-sign-l/-r` 覆盖并记录原因，
**不要恢复整套“左右镜像”**。

### 10.4 真机标定/排查

1. 先分别做 tool yaw / pitch / roll 各 20–30°（不要同时转），**看手的
   朝向**。列 55:61 是控制器原始差分 `(yaw,pitch,roll)`，列 49:55 是腕
   cmd，可用于离线对照。
2. **tool roll 却动了 pitch/yaw，或工具朝向整体偏到别的轴**：先查
   `--wrist-axis-order-l/-r`（默认 `0,1,2`，交换 x/z 用 `2,1,0`，另两对
   用 `0,2,1` / `1,0,2`），再查 `--wrist-{yaw,pitch,roll}-sign-l/-r`。
   换握法/换人后要复查。
3. **tool pitch/yaw 动了 2–3 个关节是正常的**：只要工具朝向跟手一致就
   不要“修”。若工具朝向斜向另一个轴，才回到第 2 步。
4. 腕 pitch 只有 ±32°、roll 左右不对称（左 `[-90°, +41.5°]`、右
   `[-41.5°, +90°]`），大幅动作会先顶到限位；先小幅度确认轴和方向，再加大。
5. 若怀疑真机的 pitch/roll 关节语义不是 Cardan 输出角（而是差动电机角），
   做一次静态检查：在腕零位分别只发 `wrist_pitch=+0.3` 和
   `wrist_roll=+0.3`，pitch 应绕 pitch 铰轴转、hand 不滚转，roll 应绕工具
   x 轴转。若两个命令都产生耦合运动，停止使用当前映射，需要向厂商确认
   关节角语义后再改（本仓库的 URDF/厂商 RL 模型都按输出角建模）。
6. 若上游肩/肘仍随纯腕动作周期性动，那是 §9 的 lever 问题，不是本节。

### 10.5 live 记录复盘：腕映射本身是准的，误差来自限位和 sim 力矩

2026-10-20 live 记录（`/tmp/v3_live.npz`，230 s，11372 tick）的离线结论：

| 指标 | 左臂 | 右臂 |
|---|---|---|
| engaged tick | 999 | 2289 |
| pitch 命令钉在 ±32° | **39.6%** | **35.3%** |
| roll 命令钉在限位 | 1.6% | 4.7% |
| yaw 命令钉在 ±146° | 0.0% | 0.9% |
| 未截断 tick 的工具姿态误差 | max **0.50°** | max **0.00°** |
| 全部 engaged tick 的工具误差 | mean 14.0°，max 92.2° | mean 4.0–11.7°，max 66.5° |

- **命令侧**：`exact-vs-record` 在所有窗口都是 `0.0000°`——record 里的腕命令
  逐位等于精确腕链解；未截断 tick 的工具误差 ≤0.5°，说明映射/符号/轴序没有
  系统性错误。误差全部出现在目标超出关节限位的 tick。
- **物理范围**：pitch 只有 ±32°，而操作者在这段会话里要求的 pitch 达
  `-90°…+69°`；roll 左 `[-90°, +41.5°]`、右 `[-41.5°, +90°]`，也远小于
  要求的扫幅。yaw 有 ±146°，所以显得「只有 yaw 跟得好」。这是硬件行程，
  软件无法把工具朝向变到不可达姿态；需要大幅腕姿时用「离合棘轮」——转
  一段、松开 grip、把手柄转回中性、再咬合继续转。
- **sim 执行侧**：MJCF 里 wrist_pitch/roll 的 `ctrlrange` 是 ±2.2 N·m，
  wrist_yaw 是 ±24 N·m；而 `upper_body_teleop.ARM_KP` 三个腕关节都是
  kp=20、kd=1。命令以 6 rad/s 斜率变化时小电机力矩先饱和，record 的
  qpos-vs-command 实测：yaw max 4.2–4.6°、pitch max 8.3–8.7°、
  roll max **15.2–15.3°**，且误差随单 tick 命令步长单调增大
  （6–20°/tick 档最大 15°）。所以即使命令完全正确，viewer 里看到的
  实际 roll/pitch 也会明显滞后于手柄。真机 vendor 配置的腕 pitch/roll
  力矩上限是 6 N·m（plant yaml），仍只有 yaw 的 1/4，需要按真机实测
  再判断滞后量；sim 的 2.2 N·m 是更保守的下界。

## 11 v4 无 tracker 3-point（`--tracking 3point`，2026-10）

先读 F03 的 v4 小节。这里只列会咬人的点：

1. **不用 body tracker，但仍要戴头显**。controller pose 与头显来自同一个 SDK；
   头显位姿只做状态诊断，**不参与固定帧、不参与任何门控**——掉头显不应冻结手臂。
2. **固定 operator frame 只标定一次**。engage 期间保持髋/脚不动；转体 / 转身后
   前伸方向会带同样角度的偏航偏差（无躯干跟踪的已知限制，不是回归）。要恢复：
   松开 grip → 转回中性朝向 → **重启 sender 重新标定**，再 engage。
3. **标定姿势**：双手柄在胸前左右分开 ≥ 20 cm，保持约 1 s；`|d|` 会打印出来。
   双手前后明显不对称会带一点帧偏航，重启重标。
4. **单侧丢跟踪**：该侧 `grip=0`、`trigger=0`，pose 保持最后一帧（绝不发零 pose
   + live grip）。不要用 `--stale-grip-close` 去“修”；失效侧手臂会保持，恢复后
   重新握 grip 才重新 engage（anchor 在新位置，不跳变）。
5. **单侧静默冻结无法检测**：SDK 没有 per-controller 可用性 API。若 PC Service 在
   丢跟踪时继续发旧 pose，sender 无法区分“静止”与“冻结”；后果只是手臂保持
   （delta=0），不跳变。这是已知限制，用第 0 步探针记录 PC Service 的实际行为。
6. **wire `engaged` 强制 0**：`smpl_joints`/`human_quat` 是占位零值，强制 0 让同
   topic 的 SMPL tokenizer 不接管；v3 链不读 `engaged`，**grip 才是离合**。
   `--auto-engage` 在 3point 下对 wire 无效。
7. **`--tracking 3point` 是 live-only**：不能和 `--tape-replay` / `--tape-once`
   组合。3point 会话照常录 intent tape，回放时不传 `--tracking 3point`
   （走 `_IntentTapeSource`），可作确定性回归。
8. **`get_time_stamp_ns()` 仍然不可靠**（见 §5.3）：3point 的 50 Hz 节奏完全用本地
   `time.monotonic()`，stamp 只出现在状态行里。单侧时效由 `_ctrl_t` + `max_age_s`
   承担，与 stamp 无关。
9. **默认腕符号左右相同，逐轴符号仍在 live A/B**（2026-10-21）：`DEFAULT_WRIST_SIGNS`
   两侧必须相同。第一轮左侧是右侧的全局取反，左腕 yaw/pitch 与手相反；第二轮两侧
   `(+1,-1,+1)` 后左手正常、右手 pitch 反；第三轮当前取 `(+1,+1,+1)` 实测。
   X2 两侧腕工具系平行（零位 roll 轴均为世界 +x），不镜像，不要恢复整套左右镜像；
   单轴不对时用 `--wrist-{yaw,pitch,roll}-sign-l/-r` 只改那一轴。详见 §10.3。
10. **X/A 长按 3 s 手臂回位（3point 专属）**：左手柄 X → 左臂、右手柄 A → 右臂，
   以 `home_speed_rad_s`（默认 0.5 rad/s ≈ 29°/s）的**每关节**速度回到
   `ARM_NEUTRAL`（与 harness/真机启动时的初始姿态一致）。回位期间该臂离合释放，
   完成后若仍握着 grip 会在回位姿态重新 anchor，不跳变；按住不放不会重复触发。
   full 模式没有此功能：长按 X 仍是整段录制开始，A 属于 A+B+X+Y / A+X 和弦。
