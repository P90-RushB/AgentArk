# AgentArk 强化学习集成

[English](README.md) | 简体中文

AgentArk 将 Unity runtime pool 作为 HTTP 环境服务提供给 RL trainer，从而在
各自的 Python 环境中执行多轮、多模态 rollout。先按
[安装指南](../docs/setup.zh-CN.md) 跑通一次真实 Unity 评测，再从下面选择
训练框架。

## 选择集成

| 集成 | Adapter 位置 | 环境协议 | 从这里开始 |
| --- | --- | --- | --- |
| ms-swift GRPO | 本仓库 `integrations/ms_swift` | v2 | [ms-swift 运行指南](ms_swift/README.md) |
| VERL GRPO | 公开 [`agentark_rl` fork](https://github.com/P90-RushB/verl/tree/agentark_rl/agentark_recipe/agentark_env_agent) | 当前 recipe 使用 legacy v1 | [VERL 接入指南](verl/README.zh-CN.md) |

ms-swift 运行指南是仓库内 adapter 的完整安装和训练路径。VERL 指南先
准备并验证 AgentArk 侧，再交接到外部 fork 中的 dataset、agent loop 和
trainer 配置。

## 共用进程结构

```text
RL trainer（框架自身的 Python 环境）
                    |
                   HTTP
                    |
AgentArk Env Server（AgentArk Python 3.10.12 环境）
                    |
          runtime sandbox pool
                    |
           多个 Unity 子进程
```

同一 host 和 port 上只运行一个 Env Server 服务进程；该进程可以并发
管理多个 runtime sandbox 和 Unity 子进程。AgentArk 环境与 trainer 环境分开
运行，两者通过 HTTP 通信，各自使用所需的依赖栈。

protocol v1 和 v2 使用隔离的 pool namespace。按所选集成指南预热对应
namespace；在一个 namespace 中准备的环境不会被另一个复用。

## 首次运行验收

扩大训练规模前，确认：

1. 一次真实 AgentArk Unity 评测已完成。
2. 所选运行指南中的 Server 健康检查和 runtime pool 检查已通过。
3. smoke 或一步训练已完成 rollout、输出 reward，并达到该集成指南中
   说明的成功标志。

共享架构、GRPO 分组与任务选择语义及运维背景见
[强化学习训练说明](../docs/rl-training.zh-CN.md)。
[ms-swift 架构说明](ms_swift/ARCHITECTURE.zh-CN.md) 详细介绍了该 adapter 的
数据流，并与 VERL 实现进行了对比。
