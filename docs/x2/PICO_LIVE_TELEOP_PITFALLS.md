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
