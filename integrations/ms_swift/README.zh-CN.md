# AgentArk × ms-swift GRPO

[English](README.md) | 简体中文

该目录用于把 AgentArk 环境接入 ms-swift GRPO。trainer 生成代码或工具调用，AgentArk
在 Unity 中执行这些 action，再将后续 observation 和 reward 返回给训练轨迹。

该集成支持多模态、多轮 rollout 和多个 Unity runtime 并发，并不限定具体模型或任务。

## 从这里开始

如果这是你第一次运行 AgentArk 训练，请直接按照
[Snake 训练教程](tutorial/README.zh-CN.md)操作。它是本集成可执行的端到端示例，涵盖准备
Snake、配置 Unity runtime、安装官方 ms-swift、运行一步 smoke，以及启动较长训练。

本 README 不再重复教程中的命令。不同目标请阅读对应文档：

| 目标 | 文档 |
| --- | --- |
| 跑通完整训练示例 | [Snake 训练教程](tutorial/README.zh-CN.md) |
| 先安装 AgentArk 并跑通评测 | [AgentArk 安装指南](../../docs/setup.zh-CN.md) |
| 理解 rollout、token、ticket 和 lease | [架构说明](ARCHITECTURE.zh-CN.md) |
| 了解仓库整体的 RL 接口 | [强化学习训练概览](../../docs/rl-training.zh-CN.md) |
| 让 Coding Agent 自动执行教程 | [教程执行指南](tutorial/SKILL.zh-CN.md) |

具体 shell 命令和已跑通配置以教程为准。本 README 只帮助你理解集成，并把示例改造成
自己的实验。

## 集成如何工作

```text
ms-swift trainer
    │  ticket / 模型生成的 action
    ▼
AgentArk Swift adapter ──HTTP──► AgentArk Env Server
                                  │
                                  ├── Unity runtime 1
                                  ├── Unity runtime 2
                                  └── Unity runtime N
    ▲                                  │
    └──── messages、图像、reward ──────┘
```

Swift trainer 和 AgentArk Env Server 使用两个独立的 Python 环境。adapter 安装在 trainer
环境中，通过 v2 协议访问 Server。Server 管理 Unity 进程，并为每条活动 trajectory
临时分配一个 runtime。

阅读教程和日志时，需要理解五个名词：

| 名词 | 含义 |
| --- | --- |
| **Runtime** | 一个独立的 Unity 环境进程。 |
| **Trajectory** | 一次模型与环境的对话，可以包含多轮。 |
| **GRPO group** | 从相同任务和 seed 开始的一组 sibling trajectory。 |
| **Ticket** | 标识一个 GRPO group 的轻量 dataset 行；真正的 prompt 和图像在 Unity reset 后产生。 |
| **Lease** | Server 将一个 runtime 临时分配给一条 trajectory 的记录。 |

## 从 Snake 示例改成自己的实验

建议先完全按照教程跑通一步 smoke，不要立刻改变训练拓扑。这样可以先确认安装和环境链路，
再单独排查实验配置。smoke 通过后，每次只改变一个维度。

### 更换模型

`AGENTARK_MODEL` 可以是本地模型目录，也可以是当前 Swift 支持的模型 ID。模型必须：

- 在当前 ms-swift 中有兼容的 template 和 processor；
- 能处理任务使用的文本与视觉 messages；
- 能使用选定的 vLLM 或 Transformers rollout backend；
- 能生成 task prompt 要求的 `<code>` 或 tool-call action 格式。

更换模型后，通常还要调整上下文长度、dtype、tensor parallel 和 GPU 显存配置。LoRA 是
显存需求较低的起点；全参数训练需要明显更多的 GPU 显存和 checkpoint 空间。

### 更换任务

Unity 包、runtime config 和可选的 `AGENTARK_TASK_NAME` 必须共同指向实际存在的任务。
如果不固定任务，AgentArk 可以为不同 group 选择不同任务。同一个 GRPO group 内的 sibling
trajectory 使用相同任务和初始 seed，但分别占用不同 Unity runtime。

### 调整训练规模

空闲 Unity runtime 数必须覆盖 Swift 的 generation batch。因此，增大 trainer batch、进程数
或 gradient accumulation 时，可能也要扩大 runtime pool。`num_generations` 决定每组
sibling trajectory 数量，并且必须能够整除 generation batch。

runtime config 中的 `warmup.num_envs` 与
`env_cfg.runtime_sandbox.pool_size` 应保持一致。在扩大教程配置前，请先阅读
[架构说明](ARCHITECTURE.zh-CN.md)中的 ticket 和 runtime 容量说明。

### 恢复训练

可恢复的实验不只有 checkpoint。run ID、ticket dataset、输出目录和 checkpoint 应一起
保留。复用同一份 ticket 可以保持 group identity 和 seed 派生稳定。其原因可参阅
[架构说明](ARCHITECTURE.zh-CN.md)中的 ticket identity 说明。

## 配置入口

通常只需要编辑或调用以下文件：

| 路径 | 用途 |
| --- | --- |
| [`tutorial/README.zh-CN.md`](tutorial/README.zh-CN.md) | 完整命令和已跑通的 Snake 流程。 |
| [`configs/agentark_grpo.env.example`](configs/agentark_grpo.env.example) | 通用的模型、trainer、rollout 和输出设置。请将所需值复制到本机配置，不要提交本机路径。 |
| [`tutorial/config/agentark_runtime_config.snake.example.yaml`](tutorial/config/agentark_runtime_config.snake.example.yaml) | Snake 示例使用的 runtime pool 模板。 |
| [`tutorial/config/deepspeed_zero2_adafactor.json`](tutorial/config/deepspeed_zero2_adafactor.json) | 已验证 Snake 长序列全参数训练使用的 ZeRO-2 配置，不启用 CPU optimizer offload。 |
| [`scripts/run_agentark_server.sh`](scripts/run_agentark_server.sh) | 启动 AgentArk Env Server。 |
| [`scripts/run_agentark_grpo.sh`](scripts/run_agentark_grpo.sh) | 检查容量并启动 `swift rlhf`。 |
| [`scripts/smoke_agentark_unity.sh`](scripts/smoke_agentark_unity.sh) | 检查两条并发 Unity trajectory 和 lease 清理。 |

launcher 会根据训练参数计算并检查 ticket 容量，因此应优先使用它提供的 `AGENTARK_*`
输入，而不是随意附加 Swift 参数。环境变量模板解释通用选项；教程则给出一组实际配合
使用过的完整配置。

## 排错方向

- **Server 不健康：**确认进程仍在运行，并确保 trainer 使用相同 URL 和协议版本。
- **空闲 runtime 不足：**停止训练，扩大最终 runtime config，重启 Server，再预热同一
  runtime pool。
- **Unity reset 超时：**检查 Unity build、task/Mods 内容、sandbox 路径、Xvfb、CPU 和
  主机内存。
- **CUDA OOM：**减小训练或 generation batch、序列长度或 vLLM 显存比例。全参数训练还要
  重新考虑 gradient checkpointing 和 tensor parallel。
- **视觉任务没有返回图片：**检查 observation 配置。只有明确的纯文本任务才应允许 smoke
  中没有图片。
- **HTTP 409/410 或残留 lease：**重新开始 trajectory。trainer 非正常退出后，Server 会在
  TTL 到期时回收 lease。

[教程的常见问题](tutorial/README.zh-CN.md#11-常见问题)记录了跑通示例时遇到的具体错误；
[架构说明](ARCHITECTURE.zh-CN.md)解释生命周期和故障恢复机制。

## 兼容性与安全

- adapter 和 rollout cleanup 当前接受 ms-swift `4.4.1`、`4.5.0.dev0` 和
  `4.6.0.dev0`。请使用 [modelscope/ms-swift 官方仓库](https://github.com/modelscope/ms-swift)；
  AgentArk agentic rollout 所需的 token 一致性修复已通过
  [PR #10012](https://github.com/modelscope/ms-swift/pull/10012)合入上游。
- 已验证部署环境为 Linux、NVIDIA CUDA 和单机 colocated rollout。多节点训练和 vLLM
  server mode 需要额外实现路由与容量管理。
- ms-swift 4.4.1 的 `async_generate` 与本集成的多轮 scheduler 不兼容。提供的 launcher
  使用批内环境并发，以及同步的 rollout/update 阶段。
- Env Server 默认监听 `127.0.0.1`，本身不提供认证或 TLS。跨机器部署时应使用受信网络、
  防火墙和认证代理。
- Unity/Roslyn 可能执行模型生成的代码或 tool action。训练任务应运行在适当隔离的环境中。
