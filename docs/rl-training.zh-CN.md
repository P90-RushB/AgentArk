# 强化学习训练：架构与语义

[English](rl-training.md) | 简体中文

AgentArk 把 Unity runtime pool 作为 HTTP 环境服务提供给 RL trainer。目前维护两条
GRPO 接入路径；操作命令以各自的运行指南为准：

- [ms-swift 运行指南](../integrations/ms_swift/README.md)：adapter 完整位于本仓库，
  当前基于已验证的 ms-swift 4.4.1，默认使用 AgentArk HTTP protocol v2。
- [VERL 接入指南](../integrations/verl/README.zh-CN.md)：AgentArk 侧桥接位于本仓库，
  trainer adapter 位于公开的 `agentark_rl` fork，当前 recipe 使用 protocol v1。

统一导航见[强化学习集成索引](../integrations/README.zh-CN.md)。本文只解释共享架构、
同步边界、GRPO 分组、任务选择与故障语义，不再复制安装和启动命令。

## 1. 共享架构

```text
┌──────────────── trainer Python 环境 ────────────────┐
│  dataset → rollout/model server → framework adapter │
└───────────────────────┬──────────────────────────────┘
                        │ HTTP v1 或 v2
┌───────────────────────▼──────────────────────────────┐
│ AgentArk Env Server（一个 host/port 一个服务进程）   │
│ task/seed selector · session/lease · runtime pool    │
└───────────────────────┬──────────────────────────────┘
                        │
             多个 runtime sandbox / Unity 子进程
```

AgentArk wrapper 和 Env Server 使用项目要求的 Python 3.10.12；ms-swift 或 VERL
使用自己的 Python 环境。trainer 通过 HTTP 交互，无需导入 `agent_ark`。

“Env Server 单进程”只约束一个地址不要运行多个会各自持有内存状态的 Server/uvicorn
worker，不限制 Unity 数量。`EnvSessionManager` 在同一个服务进程内分配不同
`worker_index`，既有 runtime sandbox 机制仍负责多开和隔离 Unity。

protocol v1 与 v2 使用隔离的 namespace。相同 `env_cfg` 才能复用同一 namespace 内的
runtime；因此预热必须使用所选 adapter 最终会发送的配置。

## 2. “异步环境交互”的准确含义

当前两条路径都会并发等待同一 rollout/generation batch 中多个环境的 reset/step，慢任务
不会阻塞 Python 逐个发起其他环境请求。ms-swift 的 `AgentArkScheduler` 使用协程，VERL
agent loop 也在 batch worker 内异步调度轨迹。

这不等于 optimizer 与 rollout 跨 batch 持续流水：

```text
收集 batch N 的全部轨迹
          ↓ barrier
用 batch N 做策略更新
          ↓
收集 batch N+1
```

所以当前 ms-swift 所说的“异步”主要是批内环境 I/O 并发；正式训练仍在 rollout batch 与
optimizer update 之间同步交替。当前 ms-swift 4.4.1 的多轮 Gym scheduler 未启用
`async_generate` 跨批流水。

## 3. GRPO group、task 与 seed

GRPO 对同一个 prompt 采样多条 sibling trajectories，并用组内 reward 建立相对优势。
AgentArk 需要保证 sibling 从同一个 task 和 seed 开始，但每条 trajectory 必须租用不同
Unity runtime。

### ms-swift

- dataset 的静态 ticket 保存 `group_uid`；Swift 将一行重复
  `G=num_generations` 次。
- sibling 共享 `group_uid`，但 Swift 为每条真实 trajectory 生成不同 request UUID；
  前者选择相同 task/seed，后者标识独立 lease 和 action。
- 同一 ticket 跨 epoch 再次出现时 `group_uid` 不变，因此当前 selector 仍映射到相同
  task/seed。把一个 epoch 所需的 ticket 数一次生成足够大，可以避免重复 ticket；是否
  跨 epoch 固定则仍由 ticket 身份和未来 curriculum 策略决定。

### VERL

- trainer 每次消费 prompt group 时生成一个 `uid`，并把它复制给该组的 n 条 sibling。
- 当前 server-managed dataset 仍显式保存 `group_seed`。因此默认语义是：`uid` 选择 task，
  row 的 `group_seed` 选择 seed，而不是二者都只由 `uid` 决定。
- 同一 row 跨 epoch 再消费时会得到新 `uid`，task 可以改变；row seed 保持不变。

固定 `task_name` 时，两条 adapter 都会把指定 task 转发给 Server。把 task/curriculum
选择保留在 AgentArk 侧，可以在不改 trainer 框架代码的前提下演进任务管理策略。

## 4. 策略损失范围

ms-swift adapter 提供两个明确选项：

- `all_turns`（默认）：轨迹中每一轮 assistant token 都参与策略损失，环境 observation
  token 不参与；
- `last_round`：中间 assistant 轮 mask 为 0，只保留最后一轮 assistant。

设置方法和当前 Swift driver 的 token/mask 处理见
[ms-swift 运行指南](../integrations/ms_swift/README.md)及
[接入架构](../integrations/ms_swift/ARCHITECTURE.zh-CN.md)。当前 VERL recipe 使用其
agent-loop response mask，对 assistant 输出计算策略损失、对环境 observation 屏蔽；它
没有复用本仓库的 Swift loss-scope 开关。

## 5. 框架接入对比

| 维度 | ms-swift | VERL |
| --- | --- | --- |
| adapter 位置 | 本仓库 `integrations/ms_swift` | 外部 fork；本仓库提供 bridge/preflight |
| 环境抽象 | Swift Gym Env + multi-turn scheduler | VERL async agent loop |
| 当前协议 | v2 | v1 |
| sibling 身份 | 静态 ticket `group_uid` + 独立 request UUID | 每次 occurrence 的 `uid` |
| task/seed 默认稳定性 | ticket 跨 epoch 固定 | task 可随新 uid 变化；row seed 固定 |
| assistant loss | `all_turns` / `last_round` | 当前 recipe 的 assistant response mask |
| 安全重试 | acquire/step/release operation ID、generation fencing | transport/5xx 重试；无端到端 exactly-once 证明 |
| 异常租约 | heartbeat + TTL 可回收 | 硬中断后通常重启 Server 并重建 v1 pool |
| 当前训练拓扑 | 已验证 colocate 路径 | 当前 recipe 为单机 FSDP2/vLLM 路径 |

protocol v1/v2 是 AgentArk Server 能力，不从属于某个 RL 框架。当前表格描述的是已有
adapter 的选择；未来 VERL 也可以迁移到 v2。

## 6. 与非 AgentArk 数据或环境混训

ms-swift 框架可以注册多个 Gym env，但当前 AgentArk launcher 全局指定：

```text
--multi_turn_scheduler agentark_scheduler
--gym_env agentark
```

`AgentArkScheduler` 会把 batch 中每个 request 都按 AgentArk ticket 解析。因此，当前实现
天然支持一个 AgentArk dataset 中混合多种 AgentArk task，但没有提供在同一个训练 job
中把另一些行路由到普通单轮数据或其他 Gym env 的 dispatcher。

要做这种跨环境混训，需要新增组合 scheduler/dispatcher，根据行级 `env_config.name`
选择 AgentArk、其他 Gym env 或非环境 rollout，同时定义各来源的 reward scale、batch
分组和 loss mask。仅仅拼接 dataset 不足以保证正确。不同 job 分别训练则不需要这层
路由。VERL 当前 recipe 同样是 AgentArk 专用 dataset/agent-loop 路径。

## 7. 容量和扩展

增加环境数量通常不需要修改 Server 核心代码：根据运行指南扩大最终 runtime 配置和
对应 namespace 的预热数量即可。仍需同时核对：

- generation/rollout batch 的峰值 lease 数；
- sandbox 磁盘、Unity CPU/内存和 Xvfb 容量；
- 模型上下文长度、视觉 token 与 GPU 显存；
- 不同 task 的 step 延迟差异和长尾等待。

ms-swift 的准确 generation batch 公式、ticket 容量和扩池命令见其运行指南。VERL
当前公式和精确配置预热见其本地接入指南。

## 8. 故障与恢复边界

Server 为阻塞 Unity reset/step/close 设置硬超时，损坏 runtime 会被丢弃并按需重建；
`max_interactions_per_runtime` 可限制长期进程增长。

v2 进一步提供稳定 operation ID、lease generation、幂等 replay、heartbeat 和 TTL。
响应丢失后，client 可以确认同一动作是否已执行，迟到请求也不能操作已经重新出租的
runtime。v1 client 可重试 transport/5xx，但没有这些身份信息，step 响应丢失后无法端到端
证明 action exactly once。

因此，正常 trainer 异常可依赖 adapter 的 cleanup；进程被 `SIGKILL`、机器故障或 OOM
硬终止后，ms-swift/v2 可等待 TTL 回收并检查 active lease，当前 VERL/v1 更稳妥的恢复
路径是重启 Server、用精确配置重新预热，再从 checkpoint 恢复。

更细的 ms-swift 数据流、脚本职责、mask 和 v2 生命周期见
[ms-swift 接入架构与训练流程](../integrations/ms_swift/ARCHITECTURE.zh-CN.md)。
