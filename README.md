# 峡谷追猎 · Gorge Chase — 开悟平台 PPO 智能体

> 腾讯开悟（AI Arena）强化学习竞技平台「峡谷追猎」赛道的自研智能体实现：
> 在 128×128 的迷宫地图中收集宝箱、拾取加速 Buff，并用「闪现」技能躲避不断加速、逐个登场的怪物。
>
> 本仓库是一个完整的**开悟代码包（code package）**，包含特征工程、奖励塑形、网络结构、PPO 算法与训练工作流。
> 框架 `kaiwudrl` 由平台下发（不入库），因此仓库需放入开悟环境后运行，不能脱离平台单独 `python train_test.py`。

---

## 1. 任务与环境

| 维度 | 设定 |
| --- | --- |
| 地图 | 128×128 网格迷宫，10 张地图（1~10 号）参与训练 |
| 视野 | 21×21 局部观测，**部分可观测**（POMDP） |
| 目标物 | 宝箱 10 个、加速 Buff 2 个（拾取后按 `buff_cooldown` 刷新） |
| 敌人 | 首只怪物开局存在；第二只在 `monster_interval` 步后出现；`monster_speedup` 步后怪物提速 |
| 技能 | 闪现（talent），8 个方向，冷却 `talent_cooldown` 步 |
| 动作空间 | **16 维离散**：8 方向移动 + 8 方向闪现 |
| 单局上限 | 1000 步；步数耗尽未死判为「存活成功」，被抓判为「失败」 |
| 观测 | 结构化 dict：`frame_state`（heroes/organs/monsters）、`env_info`、`map_info` 21×21 通行位图 |

难点在于**三件事同时发生**：目标稀疏（宝箱被吃一个少一个）、威胁动态升级（怪物数量与速度随时间增长）、观测不全（只能看到身边 21×21）。

---

## 2. 整体结构

```
                        env_obs (dict)
                             │
                    ┌────────▼─────────┐
                    │   Preprocessor   │  pb2struct(): 解析 frame_state / env_info / map_info
                    │                  │  维护 128×128 全局记忆地图 + 21×21 局部视野
                    └────────┬─────────┘
     feature (2961) ────────┤──────── legal_action (16) , reward (1)
                    ┌────────▼─────────┐
                    │      Model       │  CNN 分支 : 6×21×21 空间地图 → 128
                    │   (Actor-Critic) │  MLP 分支 : 315 维标量特征 → 64
                    │                  │  fusion   : 192 → 128 → actor(16) / critic(1)
                    └────────┬─────────┘
                             │ logits, value
                    ┌────────▼─────────┐
                    │  masked softmax  │──→ 采样动作 a ∈ [0,15]
                    └──────────────────┘
```

训练侧是开悟标准的三进程异步架构：

```
 Actor(aisrv) ──sample──► Reverb 样本池 ──batch──► Learner ──dump──► 模型池
      ▲                                                                    │
      └───────────────── 每 1 分钟拉取最新模型 ◄────────────────────────────┘
```

- `agent_ppo/workflow/train_workflow.py` 负责 Actor 侧对局采样，每局结束做 GAE 后把整局样本交回框架；
- `agent_ppo/algorithm/algorithm.py` 负责 Learner 侧 PPO 更新；
- 两端共用同一个 `Agent` 类，由 `conf/algo_conf_gorge_chase.toml` 装配。

---

## 3. 观测构造：2961 维特征

| 特征组 | 维度 | 说明 |
| --- | --- | --- |
| 归一化坐标 | 2 | 当前格 `(x, z) / 128` |
| 行 one-hot | 128 | 精确定位行坐标，弥补归一化坐标的分辨率损失 |
| 列 one-hot | 128 | 同上 |
| 器官状态 | 15 | 宝箱/Buff 的存活状态（`config_id` 对齐） |
| 最近怪物 | 6 | `[是否存在, 单位方向x, 单位方向z, 目标x, 目标z, 归一化距离]` |
| 最近宝箱 | 6 | 同上 |
| 历史位置 | 6 | 20 步前的落脚点，提供「我在往哪走」的动量信息 |
| 怪物汇总 | 8 | 数量、是否有第二只、平均速度、相对朝向、**阶段码**、**加速紧迫度**、**闪现冷却**、**是否被追** |
| 合法动作掩码 | 16 | 与动作空间对齐，供网络感知动作可用性 |
| 视野地图 | 6×21×21 | 宝箱图、障碍图、怪物图、Buff 图、**访问记忆图**、**探索/危险合成图** |

两个设计取舍：

1. **坐标既归一化又 one-hot**。纯归一化坐标在 128 格上只有 1/128 的分辨率，网络难以区分相邻格；one-hot 让「精确站位」线性可分，代价是 256 维。
2. **地图特征走 CNN 而非展平**。21×21 视野自带空间结构，卷积的局部感受野 + 平移等变性比全连接更省参数也更容易泛化到不同地图。

---

## 4. 奖励设计：14 项塑形 + 阶段权重

单一稀疏奖励（吃到 +1、被抓 -1）在 1000 步的对局里几乎学不动，因此把奖励拆成「即时项 + 势能项 + 惩罚项」：

| 类别 | 奖励项 | 作用 |
| --- | --- | --- |
| 生存 | `survive` | 每步小额正收益；怪物进入视野时减半，避免「无脑苟」 |
| 目标 | `treasure` / `buff` | 收集时给大额奖励，被追时/怪物极近时给倍率加成 |
| 势能 | `treasure_approach` / `buff_approach` | 按「距离减少量」给奖励（`min(delta, 3)`），把稀疏目标变成稠密梯度 |
| 风险 | `monster` | 距离 ≤4 格按线性衰减惩罚，<8 格轻惩罚 |
| 技能 | `flash` + `flash_cd` | 闪现位移 ≥5 格才奖励；逃命/死胡同逃脱额外加成；朝怪物闪、原地浪费、空放冷却均惩罚 |
| 反退化 | `stop` / `wander` | 原地不动、5/10 步位移过小、绕圈都会被扣分 |
| 探索 | `explore` | 首次到达新格子给奖励，安全区移动给小额奖励；另有「朝怪物反方向探索」加成 |
| 危险 | `danger_zone` / `dead_end` | 高危险区域、被堵死胡同额外惩罚 |

**阶段课程（curriculum）**：按步数划分 early / mid / late，用倍率表动态调整各奖励权重——前期鼓励探索与收集（explore×1.5、buff×1.5、survival×0.5），后期怪物加速后转向保命（survival×2.0、explore×0.3）。这让同一套超参在「开局发育」和「终局逃生」两个差异极大的阶段都能工作。

奖励全部集中在 `agent_ppo/feature/definition.py: reward_process()`，权重集中在 `agent_ppo/conf/conf.py`，调参时只改配置不动逻辑。

---

## 5. 决策增强：让 Agent「记得住、看得懂地形」

21×21 的视野不足以支撑长距离导航，因此在预处理层维护了一套显式记忆与推理模块（`agent_ppo/feature/preprocessor.py`）：

- **全局记忆地图**（128×128）：`visit`（访问频次，+0.05/次，衰减上限 1.0）、`terrain`（0 障碍 / 1 通行 / 2 宝箱 / 3 Buff / 4 死角）、`monster`、`danger`。每步把全局地图按英雄位置裁出 21×21 局部窗口（越界补零），喂给 CNN——等价于给网络一个「记忆外挂」。
- **危险场**：以怪物为中心按距离线性衰减铺开（半径 8；怪物提速后半径 12、强度 ×1.5），全场每步乘 0.9 衰减，上限 10。宝箱全部落在危险区时，用 `0.6×危险度 + 0.4×距离` 的加权分数重选目标（`_select_target_in_danger`）。
- **地形分析**：对视野内每个通行格数 8 邻域的通行数——1 个出口即死角，2 个出口即窄道，写入全局地形图；配合「怪物距离 <5 且身处死角」判定为**被堵死胡同**，触发闪现逃脱奖励。
- **包夹检测**：把 8 个方向切成扇区，按 `速度×5 / 距离` 累加威胁值，高危扇区 ≥5 个即判定被包夹，并给出威胁最低的逃脱方向。
- **反绕圈**：最近 10 步位置去重后仍有重复 → 判定绕圈，强制切到「朝怪物反方向」的探索目标，并用 `bad_move_ids` 屏蔽导致撞墙/原地不动的动作（最多屏蔽 4 个方向，且保证至少 4 个移动方向可用，避免把动作空间锁死）。

---

## 6. PPO 实现要点

| 超参 | 取值 |
| --- | --- |
| `gamma` / `lambda` | 0.99 / 0.95 |
| 学习率 | 3e-4（Adam，β=(0.9, 0.999)） |
| PPO clip | 0.2（策略与价值双裁剪） |
| 熵系数 | 5e-4 |
| 梯度裁剪 | 0.5 |
| batch size | 2048（Reverb 样本池，`preload_ratio=1.0`，`MinSize` 限流） |
| 模型落盘 | 每 100 步 dump，Actor 每分钟同步一次 |

实现上有三处值得说明：

1. **合法动作掩码贯穿采样与损失**。采样侧用 `logits - 1e20×(1-mask)` 做 masked softmax；损失侧用 `logits + 1e5×(mask-1)` 再 softmax，保证「非法动作概率≈0」在两个环节一致，避免 `old_prob` 与 `new_prob` 口径不一致导致 ratio 失真。
2. **价值函数裁剪**。`max((R-V)², (R-clip(V_old±0.2))²)`，与策略裁剪配套，抑制 critic 在单批样本上的过冲。
3. **GAE 在整局结束后反向递推**。`next_value` 取下一帧的 `V(s_{t+1})`，末帧用终局奖励（胜 +10 / 负 -10）闭合，`advantage = GAE`，`reward_sum = GAE + V`。

网络部分：CNN（32→64→64，3×3 卷积 + MaxPool，展平后 1600→128）与 MLP（315→128→64）分别提特征后拼接融合（192→128），再分叉出 actor(16)/critic(1) 两头，全程正交初始化，总参数量约 **33.8 万**——刻意压小，避免在开悟的 CPU 采样进程上成为吞吐瓶颈。

---

## 7. 目录结构

```
.
├── kaiwu.json                     # 平台工程标识（project_code = gorge_chase）
├── train_test.py                  # 单机联调入口（小 buffer 跑通 train→sample 闭环）
├── conf/
│   ├── app_conf_gorge_chase.toml  # 应用装配：rl_helper + AsyncPolicy
│   ├── algo_conf_gorge_chase.toml # 算法装配：ppo / diy 两套 agent 与 workflow
│   ├── configure_app.toml         # 样本池、batch、模型同步、预加载
│   └── .gitignore                 # 忽略平台下发的 kaiwudrl/
├── agent_ppo/                     # 主方案（本次实现）
│   ├── agent.py                   # Agent：predict / exploit / learn / save / load
│   ├── algorithm/algorithm.py     # PPO loss + 梯度裁剪 + 监控上报
│   ├── model/model.py             # CNN + MLP 混合 Actor-Critic
│   ├── feature/
│   │   ├── definition.py          # 样本结构、奖励函数、GAE
│   │   └── preprocessor.py        # 记忆地图、地形分析、危险场、掩码
│   ├── conf/                      # 超参、奖励权重、环境配置、监控面板
│   └── workflow/train_workflow.py # Actor 侧采样工作流
├── agent_diy/                     # 平台自带的最小脚手架（未改动，保留作对照）
└── ckpt/                          # 模型落盘目录（权重不入库）
```

`agent_diy` 是平台下发的空白模板（全 `pass`），保留它是为了对照「官方脚手架 vs 自研实现」的差异，也方便回退。

---

## 8. 如何运行

1. 在开悟平台创建「峡谷追猎」工程，把本仓库内容作为代码包上传（或放入平台工作区）；
2. 平台会自动下发 `kaiwudrl/` 与 `common_python/` 依赖（故不入库）；
3. 本地冒烟测试：修改并运行 `train_test.py`（内置 `replay_buffer_capacity=10`、`train_batch_size=2` 的小样本闭环，几秒内可验证「采样→训练→落盘→加载」链路）；
4. 正式训练：在平台提交任务，监控面板见 `agent_ppo/conf/monitor_builder.py`（reward / total_loss / policy_loss / value_loss / entropy_loss 五块面板，60 秒上报一次）。

---

## 9. 已知限制与改进方向

写在前面：这些是代码里**客观存在**的问题，记录在此作为后续迭代清单。

| 问题 | 影响 | 改进 |
| --- | --- | --- |
| 每批样本只做 1 个 epoch、不分 minibatch 打乱 | 样本利用率低，PPO 的「多轮复用」优势没吃到 | 改成 `epochs=4~8` + 随机切分 minibatch + 早停 KL |
| `Config.ENV_CONFIG` 与 `train_env_conf.toml` 两处配置 | 阶段边界按 500/700 算，实际环境是 300/500，`steps_to_speedup` 等特征漂移 200 步 | 统一从 `usr_conf` 读取，删掉静态副本 |
| 截断（到达 1000 步）时 `next_value` 置 0 | 价值目标有偏，末段状态估计偏低 | 截断时用 `V(s_T)` bootstrap，仅真正终止才置 0 |
| 死角与窄道都写 `terrain=4` | 语义混用，危险区惩罚会误伤正常窄道 | 拆成两个枚举值，或单独维护 `dead_end` 位图 |
| `FAIL_PENALTY` 定义但从未传入 `is_terminal` | 死代码，终局惩罚实际只由 workflow 的 ±10 兜底 | 删除或接入 |
| `get_legal_action()` 每步被调用两次 | 无谓开销 | 缓存到成员变量 |
| 奖励项偏多（14 项）且权重手工设定 | 调参空间大、归因困难 | 收敛到 5~6 项核心奖励，其余转为特征 |

---

## 10. 版权与致谢

- 代码骨架、`agent_diy` 脚手架与 `kaiwu.json` 工程描述来自**腾讯开悟（AI Arena）官方代码包模板**，文件头版权归腾讯所有（`Copyright © 1998 - 2026 Tencent. All Rights Reserved.`）。
- `agent_ppo/` 下的特征工程、奖励塑形、网络结构与训练流程为在此基础上完成的自研实现。
- 本仓库仅用于**个人学习与竞赛复盘**，不作为官方发布物，请勿用于商业分发。

---

## 11. 参考

- 腾讯开悟平台：<https://aiarena.tencent.com/>
- PPO: Schulman et al., *Proximal Policy Optimization Algorithms*, 2017
- GAE: Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, 2016
