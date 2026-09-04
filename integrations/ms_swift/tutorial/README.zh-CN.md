# 使用官方 ms-swift 全参数训练 AgentArk Snake

[English](README.md) | 简体中文

本文给出一条已经实际跑通的端到端路径：准备 16 个并发 Unity runtime，然后使用官方
ms-swift 对 Qwen3.5-9B 进行 8 卡 GRPO 全参数训练。目标是帮助你验证 AgentArk 的
多模态 rollout、Unity 交互、reward、反向传播和 checkpoint 链路，不要求复现实验中的
具体 reward 或耗时。

[Snake](https://p90-rushb.github.io/agentark-hub/tasks/snake/) 是 AgentArk Hub 托管的一个
2D 网格任务：agent 根据画面控制蛇寻找食物，同时避免撞墙或撞到自身。它的规则简单，
但仍要求模型结合当前位置、身体结构和后续碰撞风险进行规划。Hub 页面提供任务说明、
Windows/Linux task 包和公开评测结果。

Hub 中发布的任务默认使用 `20×20` 地图。本文把逻辑地图临时改为 `8×8`，是为了减少
早期探索空间，让短训练更快出现有效 reward、便于确认训练链路确实跑通；这不是 Snake
任务的正式评测规格。完成集成验证后，可以换回默认地图或使用其他任务继续训练。

如果你第一次接入 AgentArk，请先完成上级
[`README.zh-CN.md`](../README.zh-CN.md) 的“一步 GRPO”快速开始，再照本文扩大训练。本文假定你已经
用过 ms-swift，并能在 Linux/NVIDIA 环境运行 `swift rlhf`；路径、端口和硬件参数都必须
按当前机器调整，不能原样照抄。

## 1. 跑通示例配置

下面是一组已经跑通的参考参数。它描述的是可工作的资源组合，不是必须逐项复现的基准：

| 项目 | 配置 |
| --- | --- |
| AgentArk Unity 包 | `AgentArk-env-1.0.3-linux` |
| ms-swift | 官方仓库（跑通时为 `4.6.0.dev0`） |
| PyTorch / vLLM | `2.10.0+cu128` / `0.19.0` |
| 任务 | Snake，逻辑地图 `8×8` |
| 模型 | Qwen3.5-9B，本地 BF16 权重 |
| GPU | 8 × NVIDIA H800 80GB |
| 训练方式 | 全参数训练，8 卡 DeepSpeed ZeRO-2 |
| optimizer | Adafactor |
| lr scheduler | `constant` |
| loss | `dapo` |
| DeepSpeed / ZeRO | ZeRO-2，见 `config/deepspeed_zero2_adafactor.json` |
| CPU optimizer offload | 未使用，`device=none` |
| rollout | vLLM colocate，TP=1 |
| vLLM 显存比例 | `0.35` |
| vLLM 最大上下文 | `16384` |
| Swift 最大序列长度 | `12288` |
| 每轮 completion 上限 | `4096` |
| `max_new_tokens` | `4096` |
| `response_length` | `4096` |
| 每卡训练 batch | `1` |
| 梯度累积 | `2` |
| effective optimizer batch | `1 × 8 × 2 = 16` |
| `generation_batch_size` | `16` |
| `num_generations` | `16` |
| 每条轨迹最大轮数 | `6` |
| ticket | 600 个，seed `1234..1833` |
| optimizer step | 600 |
| Unity runtime | 16 |

`600` step 用于在跑通之后继续观察 reward 和训练稳定性，不是 AgentArk 的最低要求。
集成验收只需要先完成第 8 节的 1-step smoke。GPU 数量或显存较少时，优先使用上级
README 的 LoRA quickstart，再按实际资源缩小模型和 generation batch。

这组配置中：

```text
global micro-batch         = 1 × 8 = 8
effective optimizer batch = 8 × 2 = 16
groups per rollout batch  = 16 / 16 = 1
```

因此每次 rollout 只包含一个 GRPO group，该 group 有 16 条 sibling trajectory。
它们共享 `group_uid`、task 和 seed，但各自持有独立的 request ID 和 Unity lease。
Unity pool 必须按 rollout 并发量配置为 16，不能按 8 卡训练 micro-batch 配成 8。

## 2. 前置条件与路径

需要提前准备：

- AgentArk 源码；
- AgentArk Linux Unity 环境包；
- 可以启动 AgentArk/ML-Agents 的 Python 环境；
- 已经安装并可运行的官方 ms-swift 环境；
- 本地 Qwen3.5-9B 模型；
- 8 张约 80GB 的 NVIDIA GPU；
- Linux 图形运行依赖和 Xvfb。

AgentArk server 和 ms-swift trainer 可以使用两个独立的 Python 环境。该示例分别使用
Python 3.10 和 Python 3.12。

### 2.1 获取官方 ms-swift

PR [#10012](https://github.com/modelscope/ms-swift/pull/10012) 的精确 token-in/token-out
修复已经合入官方仓库，直接使用官方源码即可：

```bash
export SWIFT_ROOT=/absolute/path/to/official-ms-swift
git clone https://github.com/modelscope/ms-swift.git "$SWIFT_ROOT"
```

不再需要旧的临时 fork。安装依赖和 AgentArk adapter 的命令见上级
[`README.zh-CN.md`](../README.zh-CN.md)。

### 2.2 定义本机路径

所有路径都应替换成当前机器的绝对路径：

```bash
export AGENTARK_ROOT=/absolute/path/to/AgentArk
export UNITY_SOURCE=/absolute/path/to/AgentArk-env-1.0.3-linux
export UNITY_ROOT=/absolute/path/to/AgentArk-env-1.0.3-linux-snake-8x8
export MODEL_PATH=/absolute/path/to/Qwen3.5-9B

export AGENTARK_PYTHON_BIN=/absolute/path/to/agentark-env/bin/python
export SWIFT_PYTHON_BIN=/absolute/path/to/swift-env/bin/python
export SWIFT_BIN=/absolute/path/to/swift-env/bin/swift

export RUN_ROOT=/absolute/path/to/agentark-runs/snake-8x8
export DATASET_PATH="$RUN_ROOT/snake-8x8-600.jsonl"
export AGENTARK_RUNTIME_CONFIG="$AGENTARK_ROOT/integrations/ms_swift/tutorial/config/agentark_runtime_config.snake.local.yaml"
export AGENTARK_SERVER_URL=http://127.0.0.1:28182
```

创建运行目录，并确认两个 Python 环境加载的是预期源码：

```bash
mkdir -p "$RUN_ROOT"

PYTHONPATH="$AGENTARK_ROOT/src" \
"$AGENTARK_PYTHON_BIN" -c 'import agent_ark; print(agent_ark.__file__)'

PYTHONPATH="$SWIFT_ROOT" \
"$SWIFT_PYTHON_BIN" -c \
  'import importlib.metadata as m, swift; print(m.version("ms-swift")); print(swift.__file__)'

"$SWIFT_BIN" --help >/dev/null
```

第二条命令应输出当前 ms-swift 版本和目标官方 checkout 中的源码路径。不要让全局
site-packages 或旧 `PYTHONPATH` 覆盖目标源码。后续命令均从 AgentArk 仓库根目录执行：

```bash
cd "$AGENTARK_ROOT"
```

## 3. 准备单任务 Snake Unity 包

建议复制一份 Unity 包专门用于训练，再修改副本。`UNITY_SOURCE` 是原始环境包，
`UNITY_ROOT` 是本次训练专用副本；如果已经准备好副本，可以直接设置 `UNITY_ROOT` 并
跳过复制。以下操作必须在启动 Env Server 之前完成。

先确认源包存在，并复制到一个尚不存在的目标路径：

```bash
test -d "$UNITY_SOURCE" \
  && test ! -e "$UNITY_ROOT" \
  && mkdir -p "$(dirname "$UNITY_ROOT")" \
  && cp -a "$UNITY_SOURCE" "$UNITY_ROOT" \
  && chmod +x "$UNITY_ROOT/AgentArk.x86_64"
```

### 3.1 只保留 Snake

任务目录位于：

```text
$UNITY_ROOT/AgentArk_Data/Resources/Mods/all_tasks
```

为了让该 Unity 包严格只提供 Snake，可以把其他任务移动到 task store 外的备份目录。
以下命令只应对刚复制的 Unity 包执行；不要在 `UNITY_SOURCE` 原始包上执行。

```bash
export MODS_DIR="$UNITY_ROOT/AgentArk_Data/Resources/Mods"
export TASK_STORE="$MODS_DIR/all_tasks"
export DISABLED_TASKS="$MODS_DIR/disabled_tasks_backup"

mkdir -p "$DISABLED_TASKS"
for task_dir in "$TASK_STORE"/*; do
  [[ -d "$task_dir" ]] || continue
  [[ "${task_dir##*/}" == "Snake" ]] || mv -- "$task_dir" "$DISABLED_TASKS/"
done
```

操作后，`all_tasks` 下应只剩 `Snake`。其他任务只是被移入 `disabled_tasks_backup`，需要时
可以恢复。

### 3.2 将默认任务切到 Snake

同步修改下面两个根配置，避免 Unity 启动时仍查找已经移走的默认任务：

```text
$MODS_DIR/config.yaml
$MODS_DIR/config.json
```

YAML 中应包含：

```yaml
task_name: Snake
```

JSON 中应包含：

```json
{"task_name": "Snake"}
```

只需修改现有 JSON 的 `task_name` 字段，不要用上面这一行覆盖整个文件。

### 3.3 将地图从 20×20 改成 8×8

编辑 `$TASK_STORE/Snake/cfg/task_config.yaml`，将 Snake 的逻辑地图设置为：

```yaml
task_params:
  initialSize: 3
  gridWidth: 8
  gridHeight: 8
  maxStepsWithoutFood: 2
  foodRestoresLife: false
```

同时确认任务允许 6 个环境 step：

```yaml
max_attempts: 1
max_steps_per_attempt: 6
```

`gridWidth` 和 `gridHeight` 是游戏逻辑地图大小，不是视觉 observation 分辨率。示例保留
Snake 的 `width: 480`、`height: 320`，只缩小地图以加快训练链路验证。

## 4. 配置 16 个 Unity runtime

复制本教程提供的 Snake 模板：

```bash
test ! -e "$AGENTARK_RUNTIME_CONFIG" \
  && cp integrations/ms_swift/tutorial/config/agentark_runtime_config.snake.example.yaml \
    "$AGENTARK_RUNTIME_CONFIG"
```

如果目标文件已经存在，不要直接覆盖；先确认它是否属于其他实验，或换一个新的
`AGENTARK_RUNTIME_CONFIG` 路径。

编辑复制后的 `.local.yaml`，至少替换以下路径：

```text
env_cfg.env_path
env_cfg.mod_path
env_cfg.runtime_sandbox.template_root
env_cfg.runtime_sandbox.template_env_path
env_cfg.runtime_sandbox.template_mod_path
env_cfg.runtime_sandbox.shared_task_store_path
env_cfg.runtime_sandbox.pool_root
```

模板已经设置：

- Env Server 端口 `28182`；
- Unity base port `5200`，16 个 worker 使用 `5200..5215`；
- `warmup.num_envs: 16`；
- `runtime_sandbox.pool_size: 16`；
- 每个 runtime 只运行一个 Unity env；
- Xvfb 和 `GALLIUM_NUM_THREADS=1`；
- protocol v2 所需的独立 runtime sandbox。

端口冲突时，可以修改 server 端口或 Unity base port，但 server URL、YAML 和训练变量必须
保持一致。`pool_root` 应使用一个新的空闲路径；不要复用其他实验正在运行的 pool。

## 5. 生成并检查 600 个 Snake ticket

每一行 ticket 表示一个 GRPO prompt group。生成 600 行，seed 从 1234 递增到 1833：

```bash
"$SWIFT_PYTHON_BIN" \
  integrations/ms_swift/scripts/generate_tickets.py \
  --output "$DATASET_PATH" \
  --run-id snake-8x8-600 \
  --count 600 \
  --task-name Snake \
  --group-seed-base 1234
```

如果目标文件已经存在，生成器会拒绝覆盖。新实验应使用新的 dataset 路径或 run ID。

正式训练前执行容量检查：

```bash
"$SWIFT_PYTHON_BIN" \
  integrations/ms_swift/scripts/check_ticket_capacity.py \
  --dataset "$DATASET_PATH" \
  --max-steps 600 \
  --per-device-train-batch-size 1 \
  --world-size 8 \
  --gradient-accumulation-steps 2 \
  --num-generations 16 \
  --generation-batch-size 16 \
  --num-iterations 1 \
  --reserve-percent 0
```

关键结果应为：

```json
{
  "generation_batch_size": 16,
  "global_train_batch_size": 8,
  "steps_per_generation": 2,
  "generation_calls": 600,
  "groups_per_batch": 1,
  "required_rows": 600,
  "dataset_rows": 600,
  "unique_group_uids": 600,
  "ok": true
}
```

## 6. 启动 Server 并预热 runtime

### 6.1 启动 Env Server

在终端一运行，并在整个训练期间保持该进程存活：

```bash
export AGENTARK_REPO_ROOT="$AGENTARK_ROOT"
export AGENTARK_SERVER_URL
export AGENTARK_RUNTIME_CONFIG

AGENTARK_PYTHON_BIN="$AGENTARK_PYTHON_BIN" \
PYTHONPATH="$AGENTARK_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
bash integrations/ms_swift/scripts/run_agentark_server.sh
```

确认 health：

```bash
curl -sS "$AGENTARK_SERVER_URL/health"
```

新 Server 此时通常显示 `env_count: 0`。

### 6.2 预热 16 个 v2 runtime

在终端二运行：

```bash
PYTHONPATH="$AGENTARK_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
"$AGENTARK_PYTHON_BIN" \
  -m agent_ark.ark_env.serving.warmup_envs \
  --config "$AGENTARK_RUNTIME_CONFIG" \
  --protocol-version v2 \
  --num-envs 16
```

然后检查容量：

```bash
"$AGENTARK_PYTHON_BIN" \
  integrations/ms_swift/scripts/check_agentark_server.py \
  --server-url "$AGENTARK_SERVER_URL" \
  --protocol-version v2 \
  --required-idle 16
```

必须看到 `healthy: true`、`idle_started_envs: 16`、`in_use_envs: 0` 和
`required_idle: 16`。

如果修改过任务、runtime YAML 或 Unity 包，应停止旧 Server，使用新的 pool 路径重新启动并
预热，不能把旧 pool 的 health 结果当作新配置已经生效。

## 7. 设置训练参数

在终端三设置公共参数：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29583
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SWIFT_SINGLE_DEVICE_MODE

export AGENTARK_REPO_ROOT="$AGENTARK_ROOT"
export AGENTARK_SWIFT_PYTHON="$SWIFT_PYTHON_BIN"
export AGENTARK_SWIFT_BIN="$SWIFT_BIN"
export AGENTARK_MODEL="$MODEL_PATH"
export AGENTARK_TICKET_DATASET="$DATASET_PATH"
export AGENTARK_SERVER_URL
export AGENTARK_RUNTIME_CONFIG
export AGENTARK_PROTOCOL_VERSION=v2

# 确保 trainer 使用目标官方 ms-swift checkout。
export PYTHONPATH="$SWIFT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# 全参数 BF16；使用 DeepSpeed ZeRO-2 分片，不启用 CPU optimizer/model offload。
export AGENTARK_TUNER_TYPE=full
export AGENTARK_TORCH_DTYPE=bfloat16
export AGENTARK_FREEZE_LLM=false
export AGENTARK_FREEZE_VIT=false
export AGENTARK_FREEZE_ALIGNER=false
export AGENTARK_OPTIM=adafactor
export AGENTARK_LEARNING_RATE=1e-6
export AGENTARK_LR_SCHEDULER_TYPE=constant
export AGENTARK_LOSS_TYPE=dapo
export AGENTARK_GRADIENT_CHECKPOINTING=true

# 8 卡训练 batch 与一个 16-sibling GRPO group。
export AGENTARK_WORLD_SIZE=8
export AGENTARK_PER_DEVICE_TRAIN_BATCH_SIZE=1
export AGENTARK_GRADIENT_ACCUMULATION_STEPS=2
export AGENTARK_NUM_GENERATIONS=16
export AGENTARK_GENERATION_BATCH_SIZE=16
export AGENTARK_NUM_ITERATIONS=1

# 六轮多模态 rollout。
export AGENTARK_MAX_TURNS=6
export AGENTARK_MAX_LENGTH=12288
export AGENTARK_MAX_COMPLETION_LENGTH=4096
export AGENTARK_ENABLE_THINKING=true
export AGENTARK_ASSISTANT_LOSS_SCOPE=all_turns

# vLLM 与训练模型共卡。
export AGENTARK_USE_VLLM=true
export AGENTARK_VLLM_GPU_MEMORY_UTILIZATION=0.35
export AGENTARK_VLLM_TENSOR_PARALLEL_SIZE=1
export AGENTARK_VLLM_MAX_MODEL_LEN=16384
export AGENTARK_VLLM_MM_PROCESSOR_CACHE_GB=0
export AGENTARK_SLEEP_LEVEL=1

export AGENTARK_TICKET_RESERVE_PERCENT=0
export AGENTARK_LOGGING_STEPS=1
export AGENTARK_REPORT_TO=none
```

launcher 会把 `AGENTARK_LR_SCHEDULER_TYPE=constant` 和
`AGENTARK_LOSS_TYPE=dapo` 分别透传为 `--lr_scheduler_type constant` 和
`--loss_type dapo`。这两个参数属于预检配置，不要在命令尾部重复覆盖。

`vllm_gpu_memory_utilization=0.35` 只控制 vLLM 的显存预算，不代表整个训练进程只占 35%。
全参数训练模型、梯度、optimizer 和 vLLM colocate 的总显存会明显更高。

示例使用 `AGENTARK_REPORT_TO=none`，指标写入 `logging.jsonl`。如果希望从训练开始
生成 TensorBoard event 文件，改为：

```bash
export AGENTARK_REPORT_TO=tensorboard
```

## 8. 先运行 1-step smoke

不要直接启动 600 step。先用完全相同的拓扑验证一次 rollout、reward、backward 和保存：

```bash
export AGENTARK_RUN_ID=snake-8x8-full-vllm-8gpu-smoke
export AGENTARK_OUTPUT_DIR="$RUN_ROOT/smoke"
export AGENTARK_MAX_STEPS=1
export AGENTARK_SAVE_ONLY_MODEL=true
export AGENTARK_SAVE_STEPS=1
export AGENTARK_SAVE_TOTAL_LIMIT=1

bash integrations/ms_swift/scripts/run_agentark_grpo.sh \
  --deepspeed "$AGENTARK_ROOT/integrations/ms_swift/tutorial/config/deepspeed_zero2_adafactor.json" \
  --max_new_tokens 4096 \
  --response_length 4096 \
  --completion_length_limit_scope per_round
```

smoke 成功的最低标准：

- 进程 exit code 为 0；
- 日志显示 `global_step/max_steps: 1/1`；
- `grad_norm` 为有限值；
- reward 和 `completions.jsonl` 已写入；
- 输出目录下存在 `checkpoint-1`；
- Env Server 的 `active_v2_leases` 回到 0；
- 16 个 runtime 再次全部 idle。

更新版使用长序列和 ZeRO-2；这里不要求 reward、平均长度或轮数匹配某个固定值，它们会随
模型、seed 和采样结果变化。`max_turns=6` 是上限；Snake 提前结束时，实际平均轮数可以
小于 6。跑通标准是 rollout、有限梯度、保存和 lease 回收均正常。

## 9. 启动 600-step 正式训练

smoke 完成且 lease 全部释放后，切换正式输出目录：

```bash
export AGENTARK_RUN_ID=snake-8x8-full-vllm-8gpu-600
export AGENTARK_OUTPUT_DIR="$RUN_ROOT/train-600"
export AGENTARK_MAX_STEPS=600

# 保存完整恢复状态，每 100 step 保存，最多保留最近两个 checkpoint。
export AGENTARK_SAVE_ONLY_MODEL=false
export AGENTARK_SAVE_STEPS=100
export AGENTARK_SAVE_TOTAL_LIMIT=2

bash integrations/ms_swift/scripts/run_agentark_grpo.sh \
  --deepspeed "$AGENTARK_ROOT/integrations/ms_swift/tutorial/config/deepspeed_zero2_adafactor.json" \
  --max_new_tokens 4096 \
  --response_length 4096 \
  --completion_length_limit_scope per_round
```

本文使用 8 卡 DeepSpeed ZeRO-2 + Adafactor。`deepspeed_zero2_adafactor.json` 的
`zero_optimization.stage` 为 `2`，但 `offload_optimizer.device` 为 `none`，因此不使用
CPU optimizer offload；Swift 层的 `offload_model` 和 `offload_optimizer` 也保持关闭。
`--max_new_tokens`、`--response_length` 和 `--completion_length_limit_scope per_round`
需要与本节的长序列配置一起使用。

Swift 会在 `AGENTARK_OUTPUT_DIR` 下创建带时间戳的 `v0-*` 子目录。运行期间可以查看：

```text
v0-*/logging.jsonl
v0-*/completions.jsonl
v0-*/checkpoint-100
v0-*/checkpoint-200
...
```

由于 `save_total_limit=2`，训练期间只保留最近两个 checkpoint。

## 10. 验收结果

训练结束后检查 `$AGENTARK_OUTPUT_DIR/v0-*/checkpoint-600/trainer_state.json`，其中应有：

```json
{
  "global_step": 600,
  "epoch": 1.0
}
```

本次已验证实跑结果如下。它用于记录这组配置，不是其他模型或 seed 组合的验收阈值。

| 指标 | 结果 |
| --- | --- |
| 训练状态 | `600/600`，exit code 0 |
| `train_runtime` | 13,930.41 秒（约 3 小时 52 分 10 秒） |
| 平均速度 | 约 23.22 秒/step（`0.043` step/s） |
| 前 20 step 平均 reward | 0.593750 |
| 全 600 step 平均 reward | 1.471146 |
| 最后 20 step 平均 reward | 1.859375 |
| 最终 step reward | 2.5 |
| 最终模型参数 | 9,409,813,744 |
| 最终 BF16 权重大小 | 18,819,635,168 bytes |

再次检查 Env Server：

```bash
"$AGENTARK_PYTHON_BIN" \
  integrations/ms_swift/scripts/check_agentark_server.py \
  --server-url "$AGENTARK_SERVER_URL" \
  --protocol-version v2 \
  --required-idle 16
```

训练完成后应有 16 个 idle runtime、0 个 in-use runtime。

一个 optimizer step 对应 16 条 sibling trajectory，因此 600 step 共进行 9,600 条
trajectory rollout。GRPO 的单步 loss 可能非常接近 0；判断训练链路是否有效时，应同时
检查非零且有限的 `grad_norm`、组内 reward 方差、reward 趋势和 completion 内容。

## 11. 常见问题

### 单卡或每卡 batch 太大导致 OOM

不要把 `generation_batch_size=16` 误写成 `per_device_train_batch_size=16`。本次成功配置是
每卡 batch 1、8 卡、梯度累积 2。长序列全参数 colocate 运行使用
`deepspeed_zero2_adafactor.json` 做 ZeRO-2 梯度分片；单卡或普通 DDP 在该长度配置下可能在
backward 阶段 OOM。

### Swift 报 vLLM 与 `device_map` 不兼容

确认 `CUDA_VISIBLE_DEVICES` 中有 8 张卡、`NPROC_PER_NODE=8`、
`AGENTARK_WORLD_SIZE=8`，且 `SWIFT_SINGLE_DEVICE_MODE` 未设置。

### vLLM 没有可用 cache block

适当提高 `AGENTARK_VLLM_GPU_MEMORY_UTILIZATION`，或者缩短上下文、减小模型或调整并行
方式。可用阈值依赖模型、GPU 和 vLLM 版本，必须通过 1-step smoke 验证。

### DeepSpeed CPUAdam 报 CUDA 版本不匹配

这通常表示系统 CUDA toolkit、PyTorch CUDA wheel 和 DeepSpeed extension 不兼容。
更新版不使用 CPU optimizer offload，而是使用 `deepspeed_zero2_adafactor.json`：ZeRO
stage 2、`offload_optimizer.device=none`，并使用 Adafactor。只有需要 CPU offload 时才
必须先修正版本组合；不要把 `deepspeed_zero2_cpu.json` 当作本文的训练配置。

### Triton 编译报 `Python.h: No such file or directory`

先查找当前 Python 的真实头文件目录。如果头文件位于 `/usr/local/include/python3.12`，
可以在启动训练前设置：

```bash
export CPATH=/usr/local/include/python3.12
export C_INCLUDE_PATH=/usr/local/include/python3.12
export CPLUS_INCLUDE_PATH=/usr/local/include/python3.12
```

不要在 Python 版本或头文件路径不同的机器上照抄该路径。

### runtime 数量不足或有残留 lease

- `generation_batch_size=16` 要求至少 16 个 idle runtime；
- 正常异常路径会释放 lease；
- OOM、`SIGKILL` 或主机故障后，protocol v2 会在 TTL 到期后回收 lease；
- 修改 Unity 或 runtime 配置后，应重启 Server 并使用新 pool 重新 warmup。

### 只有 Snake 后 Unity 启动失败

确认 `Mods/config.yaml` 和 `Mods/config.json` 的默认 `task_name` 都已改成 `Snake`。
如果根配置仍指向已移走的任务，Unity 会在加载 task store 时失败。

## 12. 给 Coding Agent 的执行约束

自动化执行时，应按以下门禁逐步推进，不要直接启动 600-step 训练：

1. 读取本教程、上级 README、runtime 模板和 launcher；
2. 只读确认 AgentArk、官方 ms-swift、两个 Python、Unity、模型和输出路径；
3. 确认 `swift.__file__` 来自官方 checkout，并记录实际包版本；
4. 只在训练专用 Unity 副本内调整 Snake，不修改或删除原包；
5. 生成 ticket 后运行 capacity check；
6. 启动 Server、warmup，并要求 16 个 idle runtime；
7. 先完成 1-step smoke，检查 reward、有限 `grad_norm`、checkpoint 和零 active lease；
8. 只有用户明确授权后才启动长时间正式训练；
9. 不覆盖已有 output，不 kill 不相关进程，不删除 checkpoint；
10. 正式训练结束后检查 trainer state、completion、reward 趋势与 runtime 回收。

更完整的自动化安全规则见 [`SKILL.zh-CN.md`](SKILL.zh-CN.md)。
