---
name: agentark-snake-grpo
description: 使用 ms-swift 的 GRPO 流程训练 AgentArk Snake，并协助准备 Python、ms-swift、Unity、dataset、runtime server 和 TensorBoard。要求先审计实际路径和版本，再启动训练；当用户比较官方 no-TITO 与 TITO 时，必须使用可审计的独立 checkout。
---

# AgentArk Snake GRPO 运行 skill

## 目标

帮助用户在任意满足依赖的机器上运行 AgentArk Snake GRPO。不要假设固定的共享存储、`/tmp`、
用户目录、端口、GPU 数量、模型路径或数据规模。

## 执行原则

1. 先读取本教程和仓库中 AgentArk/ms-swift 集成说明。
2. 让用户提供或确认 AgentArk、ms-swift、Python、Unity、model、dataset、server 和
   output 的实际路径。
3. 先进行只读检查，不要因为路径看起来像示例就直接创建训练目录。
4. 验证 Python 可以实际执行，检查 venv 的 `pyvenv.cfg` 和 `bin/python` 是否为有效链接。
5. 验证 `agent_ark.__file__`、`swift.__file__` 和 git revision 来自目标 checkout。
6. 验证 Unity executable、Mods、task store、runtime config 和 runtime pool。
7. 生成或校验 Snake tickets，检查任务名、seed 范围、行数和唯一 group_uid。
8. 执行 ticket capacity 预检。
9. 启动 server 和 runtime warmup 后，再执行训练命令的 dry-run。
10. 只有用户明确授权时才启动正式训练。

## Python 环境规则

Python venv 不包含 base interpreter。不能只复制 venv 目录并假设 `bin/python` 仍然有效。
如果发现绝对软链接指向其他机器：

- 使用当前机器的匹配 Python 重新创建 venv，或；
- 由用户明确提供当前机器的 base Python，并修复 `bin/python` 与 `pyvenv.cfg`。

不要把 venv 链接到不匹配的 Python 小版本。不能因为 `bin/python` 这个目录项存在，就认为
解释器可执行。

## 训练配置

除非用户指定其他配置，否则向用户确认以下关键参数：

- task name 和环境版本；
- dataset 路径、group 数、seed 规则；
- `max_steps`、`max_turns`、`num_generations`、`generation_batch_size`；
- full parameter 或 LoRA；
- dtype、DeepSpeed、vLLM 和 GPU 拓扑；
- completion length limit 的作用域；
- output directory 和 TensorBoard。

不要擅自把教程中的示例值当成当前实验的强制值。

## 官方 no-TITO 与 TITO

需要对比官方 no-TITO 和 TITO 时：

- 官方实验必须加载官方 ms-swift checkout；
- TITO 实验必须加载 TITO checkout；
- 启动前打印两个 checkout 的 resolved path 和 revision；
- 不要用 `AGENTARK_DISABLE_TITO` 在 TITO checkout 上伪造官方 no-TITO；
- 检查 `PYTHONPATH`、wrapper、console script 和 worker 命令行，防止 checkout 串用。

## 安全边界

- 未经用户明确授权，不启动训练；
- 不删除 checkpoint；
- 不停止、kill 或重启用户已有任务；
- 不覆盖已有 output directory；
- 不使用 `pkill`、`killall` 或宽泛的进程匹配清理环境；
- dry-run 和 audit 不得产生训练 worker；
- 发现失效 Python 链接、旧机器路径、错误 checkout 或容量不足时，停止并报告。

## 参考脚本

- `scripts/generate_snake_tickets.sh`：参数化生成 Snake tickets；
- `config/agentark_runtime_config.example.yaml`：通用 runtime 配置模板；
- `config/deepspeed_zero2_cpu.json`：通用 ZeRO-2 CPU optimizer offload 示例。

具体启动参数应以当前 AgentArk 和 ms-swift checkout 中的 launcher、CLI help 和配置定义
为准。
