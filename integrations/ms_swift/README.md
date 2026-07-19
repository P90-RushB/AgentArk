# AgentArk × ms-swift GRPO

这是 AgentArk 的 ms-swift GRPO 训练接入。它把 AgentArk 注册为 Swift Gym 环境，
通过 AgentArk Env Server 并发调度 Unity runtime，并将模型生成的代码或工具调用作为
环境 action。训练轨迹可以包含多轮文本与视觉 observation，环境直接返回 reward。

适配层不绑定具体模型。只要所选模型能够被当前 ms-swift 与 vLLM 加载、能够处理
AgentArk 返回的 OpenAI 多模态 messages，并能生成任务要求的代码或工具调用格式，就可以
使用同一套训练流程。模型大小主要由训练机器的 GPU、内存和并发配置决定。

## 快速开始

下面的顺序是一条完整的最小运行路径：

```text
跑通 AgentArk 评测
→ 安装 Swift trainer
→ 填写本地训练配置
→ 启动 Env Server
→ 运行双环境 Unity smoke
→ 运行一步 GRPO
```

所有命令均从 AgentArk 仓库根目录执行。

### 1. 确认 AgentArk 评测可运行

先按 [`docs/setup.zh-CN.md`](../../docs/setup.zh-CN.md) 完成 AgentArk 安装并跑通一次真实
Unity 评测，再继续下面的训练配置。

### 2. 安装 Swift trainer

AgentArk Env Server 和 Swift trainer 使用各自的 Python 环境。当前发布版本固定并验证于
ms-swift `4.4.1`；下面是已经回归通过的软件栈：

```bash
python -m pip install -U uv
uv venv .venv-swift --python 3.12
source .venv-swift/bin/activate
uv pip install -e integrations/ms_swift \
  "torch==2.10.0" \
  "vllm==0.19.0" \
  "transformers==5.14.1" \
  "trl==0.29.1" \
  "peft==0.19.1" \
  "accelerate==1.14.0" \
  "datasets==4.8.4" \
  --torch-backend=auto
```

这套版本用于 Linux、NVIDIA CUDA 和 vLLM colocate。若硬件需要其他 Torch/vLLM wheel，
保持 `ms-swift==4.4.1`，按照相应 CUDA 兼容关系安装，并重新执行本页 smoke。

### 3. 创建本地配置

复制两个模板：

```bash
cp config/ark_env/agentark_runtime_config.example.yaml \
  config/ark_env/agentark_runtime_config.local.yaml
cp integrations/ms_swift/configs/agentark_grpo.env.example \
  integrations/ms_swift/configs/agentark_grpo.env.local
```

两个 `*.local` 文件会被 Git 忽略。编辑
`integrations/ms_swift/configs/agentark_grpo.env.local`，至少填写：

```bash
AGENTARK_SWIFT_PYTHON="$PWD/.venv-swift/bin/python"
AGENTARK_MODEL=/path/to/local-model
# 也可以使用 Swift 支持的模型 ID，例如：AGENTARK_MODEL=organization/model-name
AGENTARK_PYTHON_BIN=/path/to/agentark-python/bin/python
AGENTARK_RUNTIME_CONFIG="$PWD/config/ark_env/agentark_runtime_config.local.yaml"
```

`AGENTARK_MODEL` 可以是本地目录，也可以是当前 Swift 支持的模型 ID。首次运行建议使用
已经下载到本地的模型，便于把模型下载问题和环境交互问题分开排查。

每个 Server、smoke 和训练终端都先加载配置：

```bash
set -a
source integrations/ms_swift/configs/agentark_grpo.env.local
set +a
```

### 4. 启动 Env Server

终端一：

```bash
set -a
source integrations/ms_swift/configs/agentark_grpo.env.local
set +a
./integrations/ms_swift/scripts/run_agentark_server.sh
```

这个终端在训练期间保持运行。一个 Server 进程可以管理多个 runtime sandbox 和 Unity
子进程。启动脚本默认从 `AGENTARK_SERVER_URL` 推导监听地址和端口；高级部署可以用
`HOST`、`PORT` 覆盖 bind address，但要保证训练终端仍能通过该 URL 访问服务。

### 5. 验证双环境 Unity smoke

终端二：

```bash
set -a
source integrations/ms_swift/configs/agentark_grpo.env.local
set +a
./integrations/ms_swift/scripts/smoke_agentark_unity.sh
```

smoke 会并发 reset 两个不同的 Unity runtime，并验证：

- 两条 sibling trajectory 使用不同 `env_id`；
- 同一个 GRPO group 得到一致的初始 messages；Server info 返回 task/seed 时也验证其一致性；
- 视觉任务能够返回 inline image；
- smoke 完成后两条 lease 都被释放。

纯文本任务可以设置 `AGENTARK_SMOKE_ALLOW_NO_IMAGE=1`。固定 smoke 任务和 seed 的示例：

```bash
AGENTARK_SMOKE_TASK_NAME=Pushbox \
AGENTARK_SMOKE_GROUP_SEED=1234 \
./integrations/ms_swift/scripts/smoke_agentark_unity.sh
```

### 6. 运行一步 GRPO

默认模板使用单机、一步、`G=2` 和 LoRA，适合先验证完整链路：

```bash
./integrations/ms_swift/scripts/run_agentark_grpo.sh
```

launcher 会在加载模型前自动完成：

1. 检查 Swift、plugin、runtime config 和 Env Server；
2. 计算本次训练所需的唯一 GRPO ticket 数；
3. 生成或验证 ticket dataset；
4. 检查空闲 Unity runtime 是否覆盖 generation batch；
5. 启动 `swift rlhf`。

成功运行后，输出目录中应包含 Swift 日志、`completions.jsonl` 和 checkpoint。一步 smoke
验证的是 Unity、多模态 rollout、reward、GRPO backward 和保存链路，不代表模型已经学会
任务。

## 选择模型和训练方式

### 模型要求

模型需要同时满足：

- 在当前 ms-swift 版本中有可用的 template/processor；
- 在当前 vLLM 版本中可以生成；
- 能接收任务使用的文本与视觉 messages；
- 能输出 AgentArk task prompt 定义的 `<code>` 或 tool-call action。

仓库端到端回归使用 Qwen3.5-0.8B，是测试机器显存条件下选择的轻量模型，不构成模型
架构、参数规模或训练方式限制。更换模型后通常需要重新调整上下文长度、视觉 token、
dtype、tensor parallel 和显存比例。

### LoRA

LoRA 是默认 quickstart：

```bash
export AGENTARK_TUNER_TYPE=lora
export AGENTARK_LEARNING_RATE=1e-5
export AGENTARK_LORA_RANK=8
export AGENTARK_LORA_ALPHA=16
```

模型需要特殊 target modules 时，可以把对应 Swift 参数追加到 launcher 命令末尾。容量
相关参数通过 `AGENTARK_*` 变量设置，launcher 会保证实际训练参数与预检一致。

### 全参数训练

切换为 full：

```bash
export AGENTARK_TUNER_TYPE=full
export AGENTARK_LEARNING_RATE=1e-6
export AGENTARK_OPTIM=adafactor
export AGENTARK_GRADIENT_CHECKPOINTING=true
```

Swift 对多模态模型的 ViT 和 aligner 有独立冻结开关。训练全部组件时显式设置：

```bash
export AGENTARK_FREEZE_LLM=false
export AGENTARK_FREEZE_VIT=false
export AGENTARK_FREEZE_ALIGNER=false
```

仅训练语言部分时，可以保留 `AGENTARK_FREEZE_VIT=true` 和
`AGENTARK_FREEZE_ALIGNER=true`。全参数训练需要按模型大小规划 optimizer state、
checkpoint 空间和 vLLM colocate 显存；`AGENTARK_SAVE_ONLY_MODEL=true` 可以减小 checkpoint，
但该 checkpoint 不包含完整 optimizer 状态，不能用于精确恢复训练。

### 模型相关参数

常用设置：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `AGENTARK_MODEL` | 必填 | 本地模型目录或 Swift 模型 ID |
| `AGENTARK_TUNER_TYPE` | `lora` | `lora` 或 `full` |
| `AGENTARK_TORCH_DTYPE` | `bfloat16` | 模型训练 dtype |
| `AGENTARK_ENABLE_THINKING` | 未设置 | 设置后传给模型 template；未设置时使用 Swift/model 默认行为 |
| `AGENTARK_FREEZE_LLM` | 未设置 | 是否冻结语言模型 |
| `AGENTARK_FREEZE_VIT` | 未设置 | 是否冻结视觉/音频 encoder |
| `AGENTARK_FREEZE_ALIGNER` | 未设置 | 是否冻结多模态 projector/aligner |
| `AGENTARK_GRADIENT_CHECKPOINTING` | full 默认 `true` | 用计算换显存 |
| `AGENTARK_VLLM_TENSOR_PARALLEL_SIZE` | `1` | 单机 vLLM tensor parallel |
| `AGENTARK_VLLM_GPU_MEMORY_UTILIZATION` | `0.30` | colocate vLLM 显存比例 |

## 扩大到正式训练

### 1. 计算 Unity 并发数

令 `D` 为 Swift 的 generation batch：

```text
D = generation_batch_size

未显式设置 generation_batch_size 时：
D = per_device_train_batch_size
  × AGENTARK_WORLD_SIZE
  × gradient_accumulation_steps
```

同时需要的空闲 Unity runtime 数就是 `D`。`G=num_generations` 决定每个 prompt group
包含几条 sibling trajectory；当 `D` 不变时，单独增大 G 不会增加 Unity 数，但必须满足
`D % G == 0`。

### 2. 扩大 runtime pool

停止旧 Server，在最终的 runtime config 中设置：

```yaml
warmup:
  num_envs: 8

env_cfg:
  runtime_sandbox:
    pool_size: 8
```

重启 Server，然后使用 AgentArk Python 预热同一份配置：

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
"$AGENTARK_PYTHON_BIN" -m agent_ark.ark_env.serving.warmup_envs \
  --config "$AGENTARK_RUNTIME_CONFIG" \
  --num-envs 8 \
  --protocol-version v2
```

smoke 脚本只准备最小的两个环境；正式训练所需的 pool 由上面的 warmup 命令准备。一个
Server 进程使用一份最终 runtime config，可以避免不同 pool 配置混合造成容量误判。

### 3. 设置正式 run

示例：

```bash
export AGENTARK_RUN_ID=run-001
export AGENTARK_GENERATED_DATA_DIR=/persistent/path/agentark-data
export AGENTARK_OUTPUT_DIR=/persistent/path/run-001
export AGENTARK_MAX_STEPS=1000
export AGENTARK_PER_DEVICE_TRAIN_BATCH_SIZE=4
export AGENTARK_GRADIENT_ACCUMULATION_STEPS=2
export AGENTARK_NUM_GENERATIONS=4
export AGENTARK_NUM_ITERATIONS=1

./integrations/ms_swift/scripts/run_agentark_grpo.sh
```

未指定 `AGENTARK_TICKET_DATASET` 时，launcher 根据训练参数自动生成足量且唯一的 ticket，
并增加默认 10% reserve。增加 `max_steps` 只增加 ticket 数；增加 batch、训练进程数或
gradient accumulation 通常会增加 `D`，需要同步扩大 Unity pool。

当前 `AGENTARK_WORLD_SIZE` 表示本机 trainer 进程数，也就是 `NPROC_PER_NODE`。多卡时，
可见 GPU 数需要覆盖这些进程，并且 `AGENTARK_VLLM_TENSOR_PARALLEL_SIZE` 必须整除
`AGENTARK_WORLD_SIZE`；launcher 会在加载模型前检查整除关系。

## 任务分布和 ticket

dataset 中每一行是一个 GRPO group ticket。真实 system/user/视觉 messages 在 Unity
reset 后注入。Swift 会把同一 ticket 复制 `G=num_generations` 次；同组轨迹使用相同
task/seed，并分别租用不同 Unity runtime。

默认由 AgentArk task selector 按 `group_uid` 选择任务。固定任务时：

```bash
export AGENTARK_TASK_NAME=Pushbox
```

launcher 会为每个 group 稳定派生不同 seed。也可以显式设置：

```bash
export AGENTARK_GROUP_SEED=1234
# 或者让第 i 个 ticket 使用 base+i：
export AGENTARK_GROUP_SEED_BASE=100000
```

新的独立实验使用新的 `AGENTARK_RUN_ID`。恢复同一实验时复用原 ticket dataset。

## Assistant 策略损失

默认对整条多轮轨迹中的每一轮 assistant 计算策略损失：

```bash
export AGENTARK_ASSISTANT_LOSS_SCOPE=all_turns
```

只训练最后一轮：

```bash
export AGENTARK_ASSISTANT_LOSS_SCOPE=last_round
```

环境 observation 不进入策略损失。adapter 返回逐 token loss mask，并保留 vLLM 实际
生成的 assistant token IDs。

## 恢复和停止

恢复时同时复用 checkpoint、run ID、ticket dataset 和输出目录：

```bash
export AGENTARK_RUN_ID=run-001
export AGENTARK_TICKET_DATASET=/persistent/path/agentark-data/run-001.jsonl
export AGENTARK_OUTPUT_DIR=/persistent/path/run-001

./integrations/ms_swift/scripts/run_agentark_grpo.sh \
  --resume_from_checkpoint /persistent/path/run-001/v0-*/checkpoint-N
```

将通配符替换为实际 checkpoint 路径。正常停止时先对 trainer 发送 Ctrl-C，等待 trajectory
cleanup，然后停止 Env Server。状态检查：

```bash
curl -s http://127.0.0.1:18080/health
"$AGENTARK_SWIFT_PYTHON" integrations/ms_swift/scripts/check_agentark_server.py \
  --server-url "$AGENTARK_SERVER_URL" \
  --protocol-version v2
```

`active_v2_leases` 应回到 0。trainer 被 OOM、`SIGKILL` 或机器故障终止时，Server 会在
lease TTL 到期后回收 runtime。

## 常用训练变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `AGENTARK_MAX_STEPS` | `1` | optimizer steps |
| `AGENTARK_PER_DEVICE_TRAIN_BATCH_SIZE` | `2` | 每个本地训练进程 batch |
| `AGENTARK_WORLD_SIZE` | `1` | 单机训练进程数 |
| `AGENTARK_GRADIENT_ACCUMULATION_STEPS` | `1` | 梯度累积步数 |
| `AGENTARK_GENERATION_BATCH_SIZE` | 自动 | 显式设置 Swift generation batch |
| `AGENTARK_NUM_GENERATIONS` | `2` | 每个 GRPO group 的轨迹数 G |
| `AGENTARK_NUM_ITERATIONS` | `1` | 同一 rollout batch 的策略更新复用次数 |
| `AGENTARK_MAX_TURNS` | `2` | 每条环境轨迹最大 assistant 轮数 |
| `AGENTARK_MAX_LENGTH` | `6144` | Swift 训练最大序列长度 |
| `AGENTARK_MAX_COMPLETION_LENGTH` | `512` | 每轮 rollout completion 上限 |
| `AGENTARK_VLLM_MAX_MODEL_LEN` | 两者之和 | vLLM 最大上下文 |
| `AGENTARK_TICKET_RESERVE_PERCENT` | `10` | 自动 ticket 的容量余量 |
| `AGENTARK_RUN_ID` | UTC 时间 + PID | 实验与 ticket 标识 |
| `AGENTARK_OUTPUT_DIR` | generated runs | checkpoint 输出目录 |

不同 task 的图片数量、视觉 token、文本长度和 step 延迟可能差别很大。正式训练前根据实际
rollout 日志调整长度、显存比例、TTL 和 Unity 并发。

## 常见问题

- `server is not healthy`：确认 Env Server 终端仍在运行，且两个终端使用相同端口。
- `idle started envs` 不足：扩大最终 runtime config 的 pool，重启 Server，并执行 v2
  warmup。
- Unity reset timeout：检查 runtime、Mods、task store、sandbox、Xvfb、CPU 和内存。
- CUDA OOM：降低 batch、generation batch、上下文长度或 vLLM 显存比例；全参数训练还可
  启用 gradient checkpointing 或增加 tensor parallel。
- smoke 没有图片：视觉任务应检查 observation 配置；纯文本任务设置
  `AGENTARK_SMOKE_ALLOW_NO_IMAGE=1`。
- HTTP 409/410：当前 trajectory 的 operation ID 或 lease 已失效，重新开始该 trajectory。
- Swift 版本校验失败：使用兼容性小节列出的 ms-swift 版本重新创建 trainer 环境。

## 兼容性与安全

- 当前 adapter 和 rollout cleanup 固定验证于 ms-swift `4.4.1`。
- 已验证运行平台为 Linux、NVIDIA CUDA、单机 vLLM colocate。
- bundled launcher 提供单机多卡参数和预检，当前端到端回归基线为单卡；多节点和 vLLM
  server mode 需要额外的容量与路由实现。
- ms-swift 4.4.1 的 `async_generate` 与多轮 scheduler 不兼容；当前训练采用批内环境并发、
  rollout 与 optimizer update 同步的流程。
- Env Server 使用单个 uvicorn worker 管理 lease；这不限制它并发管理多个 Unity runtime。
- `scripts/compat/sitecustomize.py` 为已验证 Torch 组合启用 PyTorch causal-conv1d fallback。
  已确认 native extension 与所选模型兼容时，可将 `AGENTARK_SWIFT_COMPAT_DIR` 指向空目录。
- Unity/Roslyn 会执行模型产生的代码或 tool action。Server 默认监听 `127.0.0.1`，API
  本身不提供 auth/TLS；跨机器部署时使用受信网络、防火墙和认证代理。

## 深入阅读

- rollout、token、ticket、lease、故障恢复和 VERL 对比：
  [`ARCHITECTURE.zh-CN.md`](ARCHITECTURE.zh-CN.md)
- AgentArk 强化学习框架入口：
  [`docs/rl-training.zh-CN.md`](../../docs/rl-training.zh-CN.md)
- Swift Gym 环境接口：
  [ms-swift 官方文档](https://swift.readthedocs.io/zh-cn/latest/Instruction/GRPO/DeveloperGuide/gym_env.html)
  （用于理解 Gym API；本接入的具体兼容行为以 `4.4.1` 为准）

发布或修改 adapter 后可运行：

```bash
PYTHONPATH=src "$AGENTARK_PYTHON_BIN" -m unittest discover -s tests -q
PYTHONPATH=integrations/ms_swift/src "$AGENTARK_SWIFT_PYTHON" \
  -m unittest discover -s integrations/ms_swift/tests -t integrations/ms_swift -q
```
