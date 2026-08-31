# 使用 ms-swift 训练 AgentArk Snake

本教程说明如何使用 ms-swift 的 GRPO 训练流程训练 AgentArk 环境中的 Snake 任务。
教程中的路径、端口、GPU 数量、数据规模和模型路径都需要根据运行机器填写，不依赖某个
集群或某台机器的目录结构。

## 1. 组件与目录

训练需要以下组件：

- AgentArk 源码及其 ms-swift 集成；
- AgentArk Unity 环境包；
- ms-swift 源码；
- 可执行 AgentArk 环境 server 的 Python 环境；
- 可执行 ms-swift 训练的 Python 环境；
- 本地模型；
- 可用的 GPU、Unity 运行依赖和显示环境。

建议先定义一组环境变量。下面的路径只是示例，必须替换成当前机器的实际路径：

```bash
export AGENTARK_ROOT=/path/to/AgentArk
export SWIFT_ROOT=/path/to/ms-swift
export AGENTARK_PYTHON_BIN=/path/to/agentark-python
export SWIFT_PYTHON_BIN=/path/to/swift-python
export UNITY_ROOT=/path/to/AgentArk-env
export MODEL_PATH=/path/to/model
export DATASET_PATH=/path/to/snake-tickets.jsonl
export OUTPUT_DIR=/path/to/output
export AGENTARK_SERVER_URL=http://127.0.0.1:PORT
export AGENTARK_RUNTIME_CONFIG=/path/to/agentark-runtime.yaml
```

除非命令中明确使用了绝对路径，本文后续命令均从 AgentArk 仓库根目录执行：

```bash
cd "$AGENTARK_ROOT"
```

训练时建议将源码、模型、数据集、日志和 checkpoint 放在当前机器性能稳定的本地存储
上。网络盘或其他共享存储是否适合运行，需要根据机器的 I/O 特性自行判断。

## 2. Python 环境

AgentArk server 和 ms-swift 可以使用不同的 Python 环境。环境必须满足项目当前版本的
依赖要求。

Python venv 不包含它依赖的 base interpreter。不要只复制一个 venv 目录，然后继续使用
其中指向另一台机器的绝对软链接。例如：

```text
venv/bin/python -> /some/other/machine/python3.10
```

在目标机器上应当使用已有的匹配版本 Python 重新创建 venv，或者确认复制后的 venv 的
`bin/python` 和 `pyvenv.cfg` 已经指向当前机器可用的 base interpreter。至少需要验证：

```bash
"$AGENTARK_PYTHON_BIN" -c 'import sys; print(sys.version); print(sys.prefix)'
"$SWIFT_PYTHON_BIN" -c 'import sys; print(sys.version); print(sys.prefix)'
```

验证 Python 包：

```bash
PYTHONPATH="$AGENTARK_ROOT/src" \
"$AGENTARK_PYTHON_BIN" -c 'import agent_ark; print(agent_ark.__file__)'

PYTHONPATH="$SWIFT_ROOT" \
"$SWIFT_PYTHON_BIN" -c 'import swift; print(swift.__file__)'
```

如果项目提供 requirements、lockfile 或镜像，请优先使用项目提供的依赖来源。不要把
不同 Python 小版本、不同 CUDA/PyTorch ABI 的环境直接混用。

## 3. 生成 Snake 数据集

可以使用本目录的脚本生成 Snake tickets：

```bash
AGENTARK_ROOT="$AGENTARK_ROOT" \
AGENTARK_PYTHON_BIN="$AGENTARK_PYTHON_BIN" \
AGENTARK_TICKET_DATASET="$DATASET_PATH" \
bash integrations/ms_swift/tutorial/scripts/generate_snake_tickets.sh \
  --count 600 \
  --task-name Snake \
  --group-seed-base 1234
```

参数含义：

- `--count`：生成的 ticket/group 数；
- `--task-name`：环境任务名，Snake 实验使用 `Snake`；
- `--group-seed-base`：第一个 group 的地图 seed；
- 第 `i` 个 group 使用 `group_seed = group_seed_base + i`；
- 每个 ticket 都应有唯一的 `group_uid`。

如果不使用脚本，也可以直接调用 AgentArk 的生成器：

```bash
PYTHONPATH="$AGENTARK_ROOT/src" \
"$AGENTARK_PYTHON_BIN" \
  "$AGENTARK_ROOT/integrations/ms_swift/scripts/generate_tickets.py" \
  --output "$DATASET_PATH" \
  --run-id snake-run \
  --count 600 \
  --task-name Snake \
  --group-seed-base 1234 \
  --force
```

生成后应检查行数、任务名、seed 范围和 `group_uid` 唯一性。数据集大小要覆盖训练过程
中的 rollout 消耗，并根据 `generation_batch_size`、`num_generations`、训练步数和
ticket reserve 策略进行 capacity 预估。

## 4. 配置和启动 AgentArk server

复制本目录的配置示例：

```bash
cp integrations/ms_swift/tutorial/config/agentark_runtime_config.example.yaml \
  "$AGENTARK_RUNTIME_CONFIG"
```

然后将其中的 Unity、Mods、task store 和 runtime pool 路径替换为当前机器的绝对路径。
不要假设 YAML 会自动展开 `${AGENTARK_*}` 环境变量；需要时直接写入已经解析好的绝对路径。

至少确认以下配置项：

- `env_cfg.env_path` 指向 Unity 可执行文件；
- `env_cfg.mod_path` 指向 Mods 目录；
- task store 存在；
- `runtime_sandbox.template_root` 与 Unity package 对应；
- `runtime_sandbox.pool_root` 位于当前机器可写目录；
- `server.port` 与训练使用的 server URL 一致；
- `warmup.num_envs` 与实际所需并发量匹配。

启动 server 的示例：

```bash
export AGENTARK_REPO_ROOT="$AGENTARK_ROOT"
export AGENTARK_RUNTIME_CONFIG="$AGENTARK_RUNTIME_CONFIG"
export PYTHONPATH="$AGENTARK_ROOT/src"

"$AGENTARK_PYTHON_BIN" \
  -m agent_ark.ark_env.serving.run_server \
  --host 127.0.0.1 \
  --port PORT
```

在另一个终端预热 runtime：

```bash
PYTHONPATH="$AGENTARK_ROOT/src" \
"$AGENTARK_PYTHON_BIN" \
  -m agent_ark.ark_env.serving.warmup_envs \
  --config "$AGENTARK_RUNTIME_CONFIG" \
  --protocol-version v2 \
  --num-envs NUM_ENVS
```

正式训练前应确认 server health、idle runtime 数、Unity task store 和 runtime pool 都
正常。不要用一个机器上的运行中 runtime pool 作为另一台机器的复制源。

## 5. 使用 ms-swift 启动 GRPO

AgentArk 的集成 launcher 位于：

```text
$AGENTARK_ROOT/integrations/ms_swift/scripts/run_agentark_grpo.sh
```

先设置公共训练参数：

```bash
export AGENTARK_REPO_ROOT="$AGENTARK_ROOT"
export AGENTARK_SWIFT_PYTHON="$SWIFT_PYTHON_BIN"
export AGENTARK_SWIFT_BIN="$SWIFT_ROOT/swift/cli/swift"
export AGENTARK_SWIFT_SOURCE_ROOT="$SWIFT_ROOT"
export AGENTARK_SWIFT_COMPAT_DIR="$SWIFT_ROOT"
export AGENTARK_MODEL="$MODEL_PATH"
export AGENTARK_TICKET_DATASET="$DATASET_PATH"
export AGENTARK_SERVER_URL="$AGENTARK_SERVER_URL"
export AGENTARK_RUNTIME_CONFIG="$AGENTARK_RUNTIME_CONFIG"
export AGENTARK_OUTPUT_DIR="$OUTPUT_DIR"
export AGENTARK_PROTOCOL_VERSION=v2
export AGENTARK_REPORT_TO=tensorboard

export AGENTARK_TUNER_TYPE=full
export AGENTARK_TORCH_DTYPE=bfloat16
export AGENTARK_MAX_TURNS=6
export AGENTARK_NUM_GENERATIONS=16
export AGENTARK_GENERATION_BATCH_SIZE=16
export AGENTARK_ENABLE_THINKING=true
export AGENTARK_TICKET_RESERVE_PERCENT=0
export PYTHONPATH="$SWIFT_ROOT:$AGENTARK_ROOT/integrations/ms_swift/src"
```

按照实际显卡数量设置 `AGENTARK_WORLD_SIZE`，并根据显存设置最大长度、vLLM 显存比例、
batch size 和梯度累积。完整参数训练通常需要配置合适的 DeepSpeed，包括 optimizer
offload。示例配置见：

```text
integrations/ms_swift/tutorial/config/deepspeed_zero2_cpu.json
```

启动示例：

```bash
bash "$AGENTARK_ROOT/integrations/ms_swift/scripts/run_agentark_grpo.sh" \
  --deepspeed integrations/ms_swift/tutorial/config/deepspeed_zero2_cpu.json \
  --completion_length_limit_scope per_round
```

启动前应检查 launcher 最终解析出来的 model、dataset、Swift source、Python、server、
output directory 和 DeepSpeed 配置。建议先执行项目已有的 capacity preflight，并先做
不启动训练的命令检查。

## 6. 官方 ms-swift 与 TITO 版本

如果实验需要比较官方 no-TITO 和 TITO，必须使用两个独立、可审计的 ms-swift checkout：

- 官方 checkout：保持官方 rollout 和多轮历史处理逻辑；
- TITO checkout：包含 TITO 所需的 token-preserving 修改。

实验 2 不能在 TITO checkout 上设置某个环境变量来伪造官方 no-TITO。环境变量开关可能
只影响训练侧 token 使用，不能恢复官方版本的多轮 rollout 行为。

两个 checkout 应分别通过 `PYTHONPATH`、wrapper 或 launcher 配置选择，并在启动前打印：

```bash
git -C "$SWIFT_ROOT" rev-parse HEAD
PYTHONPATH="$SWIFT_ROOT" "$SWIFT_PYTHON_BIN" \
  -c 'import swift; print(swift.__file__)'
```

不要让旧 checkout、全局 site-packages 或当前 shell 中残留的 `PYTHONPATH` 覆盖目标版本。

## 7. 训练前检查清单

正式训练前确认：

- Python 解释器可执行且版本符合项目要求；
- `agent_ark.__file__` 来自目标 AgentArk checkout；
- `swift.__file__` 来自目标 ms-swift checkout；
- AgentArk server health 正常；
- Unity executable、Mods 和 task store 存在；
- runtime 数量满足 rollout 并发需求；
- dataset 的任务名、行数、seed 和 group_uid 正确；
- capacity preflight 通过；
- model 路径可读；
- output directory 可写且不会覆盖其他实验；
- TensorBoard 配置与 output directory 一致；
- 多卡、DeepSpeed、vLLM 和 dtype 配置与实际硬件匹配；
- 没有失效软链接、旧机器 shebang 或错误的绝对路径。

## 8. 常见问题

### `bin/python: No such file or directory`

通常是 venv 中的 `bin/python` 指向了另一台机器的绝对路径。检查：

```bash
readlink /path/to/venv/bin/python
cat /path/to/venv/pyvenv.cfg
```

在当前机器用匹配版本 Python 重新创建 venv，或修复链接和 `pyvenv.cfg`。不要把它链接
到不同 Python 小版本。

### 找不到 task store

检查 runtime YAML 中的 `mod_path`、task store 和 Unity package 是否对应。不要把没有
展开的环境变量占位符直接交给不支持变量展开的 YAML 读取器。

### capacity 不足

根据 `max_steps`、训练 batch、梯度累积、`generation_batch_size`、`num_generations`、
`num_iterations` 和 reserve 策略重新计算所需 ticket 数。不要只根据 dataset 行数猜测
是否足够。

### 多轮 rollout 与训练 token 不一致

需要先确认使用的 ms-swift checkout 和 TITO 实现，再解释 tokenization、assistant 历史
消息和 loss mask。不要只通过一个环境变量推断整个 rollout 链路已经切换。
