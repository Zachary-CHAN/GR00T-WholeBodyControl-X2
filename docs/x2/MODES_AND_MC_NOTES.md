# 模式数据流与 MC 版本注记（2026-10-09）

> 这不是 runbook,是一次代码级分析的记录,回答三个问题:
> ① 项目对厂商 MC（运动控制器）版本有什么要求;② 现有数据流是否支持
> "locomotion 策略 + Pico 上肢遥操"的混合形态;③ WHOLE_BODY 与
> LOCOMOTION 两种模式在数据流/控制流上的异同。
> 操作步骤看 [`F03_pico_teleop.md`](F03_pico_teleop.md) 与
> [`F10_real_robot_colcon.md`](F10_real_robot_colcon.md)。

---

## 1 MC 版本:没有版本要求,有一处刚度耦合

项目**全文没有任何对厂商 MC 固件/软件版本的要求**。它锁定的是自己的
**plant 版本**——`vendor_20260823`（执行器物理参数:armature、kp/kd、
默认姿态,在 `policy_parameters.hpp` 编译期植入,启动必须打出
`[plant] RUNTIME PLANT LOADED: vendor_20260823`）。模型血系必须与
plant 配对,不配的症状是"腰顶死在限位"（F06 排障表）。

与 MC 真正相关的只有四条**行为性**约定:

1. **F06 "Deploy tuning for this lineage"（唯一的 MC 特性耦合）**:该 plant
   血系的模型用 unity gains + **MC-stiffness 腰俯仰**——默认调参
   `trained_gains_s0_inc_waistmc.yaml` 文件名里的 `waistmc` 就是它,即腰
   pitch 关节刚度要按 MC 驱动下的实测表现调;`trained_gains_s0.yaml` 是
   去掉这层加强的版本,**两个预设别混用**。
2. **接管/交还仪式**:deploy 起来时先做上电姿态预检再 MC stop 接管;停止时
   `RAMP_OUT → HOLD_FOR_MC → MC 重启`（`x2_pc2_daemons.sh stop` 自动走）。
3. MC handover 时的站姿用 `x2_stand_default_pose.yaml`。
4. 机器人趴下时 **MC 站立模式会拒绝指令,这是预期行为,不要反复硬刚**
   （`gamepad_cheatsheet.md`）。

## 2 locomotion 策略 + Pico 上肢遥操:支持,是设计好的混合形态

F03 真机段写明:`X2_WHOLE_BODY_TELEOP=1`（默认）时,overlay 只在
**laptop 流新鲜且 operator engaged** 时驱动,否则会话退化为普通 pad 会话。
Pico 在 LOCOMOTION 模式下可用的通道:

| 通道 | 线路 | 说明 |
|---|---|---|
| 走/转 | 摇杆 + LT 死守 → `planner_cmd :5563` → kplanner | pose encoder 路径 |
| 腰 | `hold_torso` intent（带三个腰角度） | 同上 |
| 头 | `head_targets @25Hz` | 同上 |
| **双手手指** | `hand_finger_cmd` 独立线 | **模式无关,一直可用**,断链时 kplanner 锁存最后目标 |

缺的一块:**手臂关节**。Pico live 的上肢跟随只在 WHOLE_BODY engage 后走
SMPL 流;LOCOMOTION 下有现成的 `arm_targets :5564` 线（kplanner 的 arm
overlay,Quest VR 手臂遥操 `vr_arm_teleop_v2.py` 在用;Pico tape 回放
pre-flight 也用它摆初始臂位,`pico_intent_sender.py` 的 arms0 段）,但
**Pico live 循环目前不主动发它**。想要"走路 + 手臂遥操"有两条路:

- **现成的**:WHOLE_BODY engage 状态下走路（LT + 步伐）,
  `KPLANNER_WB_ARM_LATCH_GAIT=keep` 会在步态中锁存手臂位形。
- **要写代码的**:LOCOMOTION 模式下把 Pico 腕/臂目标转到 `arm_targets`
  线——线、overlay、速率限制（3 rad/s）全是现成的,只差发送逻辑。

## 3 WHOLE_BODY vs LOCOMOTION:数据流 + 控制流对比

**相同点（同一底座）**:同一个 deploy 二进制、同一 50Hz 策略环 + 500Hz
写出器 + 同一安全栈（倾角 watchdog / SAFE_HOLD / RECOVER / E-STOP）;
两条路最终都汇到 `x2_pose_merger → pose_watchdog :5558 → deploy`;手指
通道独立于模式。注意代码里的内部名叫 `StreamMode.ARM_MANIPULATION`,
显示名才是 "WHOLE_BODY"。

| 维度 | LOCOMOTION | WHOLE_BODY（engage） |
|---|---|---|
| **权威来源** | kplanner（pose encoder 路径） | SMPL token 流（deploy 把权限交给 token 流） |
| **Pico 数据用量** | 只用按钮/摇杆/握持;body 流闲置但仍判活 | 全身 31 SMPL 关节 + 根四元数 + 腕目标全用 |
| **数据流** | 摇杆→`IntentDecoder`→`planner_cmd :5563`→kplanner ONNX→`pose :5556`→merger | body→intent `:5573`→SMPL tokenizer→tokens `:5574`→pose_watchdog `:5558`（绕过 merger 的 kplanner 输入） |
| **kplanner 状态** | 活跃（持续流 idle anchor） | idle + release（B 按下瞬间先发） |
| **根位移** | 摇杆速度指令 | LT 死守 + 操作员步伐（micro-step 门控） |
| **上肢** | 默认位形/锁存（可被 arm overlay 叠加） | SMPL 直接跟随（手臂+腕+腰） |
| **engage 门** | 不需要 | body 新鲜（1.2s 内有新帧 + identical-streak<50）+ deploy 侧 `<0.50s` 双重门 |
| **退出** | — | B 退出,限速回落到 kplanner（不是瞬跳） |
| **断流行为** | 手柄输入不受 body 断流影响 | 立即强制 DISENGAGED,一拍内回落 kplanner/pad |

**上真机前建议**:第一次 engage 前在 sim 里专门验一次"WHOLE_BODY 下
LT + 走两步"——这是 live 独有、tape 回放完全验不到的行为,也是两种模式
数据流差异最大的一处。

### 附:两个模式问题的代码级依据

- 关节数:MJCF `x2_ultra.xml` 31 个 motor（leg 12 + waist 3 + arm 14 +
  head 2）与 `obs_config_x2_ultra.yaml num_dofs: 31` 对齐;OmniHand 手指
  不在这 31 个里（走 `hand_finger_cmd` 独立线）。
- pelvis 不固定:`<freejoint name="floating_base_joint"/>`;teleop 原地
  做上半身动作时双腿仍由策略动态平衡,不是腿被冻结。
- tape 回放不驱动 locomotion 的判决行:
  `pico_intent_sender.py:641`
  `dual_mode = not args.no_locomotion and (not args.tape_replay or full_tape)`
  ——live 默认 True;自带 tape 无 `full_session` 字段故为 False;live 录制
  的 tape 写 `full_session=True`（`live_pico_smpl_teleop.py:230`）,重放时
  行走可用。
