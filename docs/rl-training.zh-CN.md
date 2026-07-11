# 强化学习训练

[English](rl-training.md) | 简体中文

AgentArk 提供 Unity 环境服务。当前 GRPO 训练集成位于公开的 verl fork：

`https://github.com/P90-RushB/verl/tree/agentark_rl/agentark_recipe/agentark_env_agent`

该设计有意把 Python 环境分开：

- AgentArk 环境服务：Python 3.10.12，使用本仓库。
- verl 训练进程：使用 verl 所需的 Python 版本和依赖栈。

trainer 无需导入 `agent_ark` 包，而是通过 HTTP 与环境服务通信。

## 1. 准备 AgentArk

按照 [setup.zh-CN.md](setup.zh-CN.md) 完成运行时下载、`.env` 配置；无桌面的 Linux
环境还需安装 Xvfb。

训练时保持运行时沙箱开启。示例配置为：

`config/ark_env/agentark_runtime_config.example.yaml`

预热池大小应覆盖 rollout 环境的最大并发数。例如，若 `TRAIN_BATCH_SIZE=2` 且
`ROLLOUT_N=8`，至少预热 16 个环境。

## 2. 启动环境服务

在本仓库中运行：

```bash
bash scripts/run_env_server_mlagents.sh
```

该脚本会加载 `.env`、设置 `PYTHONPATH=src`，并启动：

```bash
python -m agent_ark.ark_env.serving.run_server --host 127.0.0.1 --port 18080
```

预热环境：

```bash
python -m agent_ark.ark_env.serving.warmup_envs \
  --config config/ark_env/agentark_runtime_config.example.yaml \
  --output tmp/warmup_snapshot.json
```

检查服务状态：

```bash
curl http://127.0.0.1:18080/health
curl http://127.0.0.1:18080/v1/envs
```

## 3. 安装 verl 集成

克隆公开的 verl fork 并切换到 AgentArk 分支：

```bash
git clone https://github.com/P90-RushB/verl.git
cd verl
git switch agentark_rl
```

AgentArk 集成位于：

```text
agentark_recipe/agentark_env_agent/
```

verl 侧的环境安装、配置、数据集生成和训练启动细节，以该目录中的 README 为准。
此 fork 不使用旧的 `recipe/agentark_env_agent` 路径。

## 4. 生成数据集行

在 verl 环境中运行：

```bash
DATA_DIR=/path/to/agentark_data

python agentark_recipe/agentark_env_agent/generate_agentark_dataset.py \
  --local-save-dir "${DATA_DIR}" \
  --num-train 1000 \
  --num-test 200 \
  --seed 1234
```

默认情况下，数据集行保留 `task_name=None`。环境服务随后把 GRPO group id（`uid`）
确定性地映射到任务和 seed。同一 GRPO group 中的所有样本使用相同任务和 seed，
不同 group 则分布到配置的任务列表中。

如果要固定一个任务：

```bash
python agentark_recipe/agentark_env_agent/generate_agentark_dataset.py \
  --local-save-dir "${DATA_DIR}" \
  --task-name Snake \
  --num-train 1000 \
  --num-test 200
```

## 5. 启动 GRPO

示例：

```bash
PYTHON=python \
MODEL_PATH=/path/to/model \
DATA_DIR=/path/to/agentark_data \
NNODES=1 NGPUS_PER_NODE=8 \
ROLLOUT_TP=8 ROLLOUT_N=16 \
TRAIN_BATCH_SIZE=1 PPO_MINI_BATCH_SIZE=1 \
AGENT_NUM_WORKERS=8 \
TOTAL_EPOCHS=1 \
TOTAL_TRAINING_STEPS=1000 \
SAVE_FREQ=250 TEST_FREQ=-1 \
TRAINER_LOGGER='["console","tensorboard"]' \
TENSORBOARD_DIR=/path/to/tensorboard/agentark \
bash agentark_recipe/agentark_env_agent/run_qwen3_5_9b_agentark_env_grpo.sh
```

常用参数：

- `TOTAL_TRAINING_STEPS`：设为正整数可固定训练步数，设为 `-1` 则运行完整 epoch。
- `ROLLOUT_N`：每个 prompt/GRPO group 的样本数。
- `AGENT_NUM_WORKERS`：并发 agent-loop worker 数量。
- `SAVE_FREQ`：checkpoint 间隔；设为 `-1` 可禁用。
- `TRAINER_LOGGER`：包含 `tensorboard` 时写入 TensorBoard 日志。

训练期间不要运行无关的 GPU 占用脚本；它们可能使 Unity reset/step 得不到资源并触发
超时。

## 任务选择语义

训练请求可以固定任务，也可以由服务端管理。

固定任务：

- 数据集行设置 `extra_info.task_name`。
- 可选的 `group_seed` 和 `unity_env_id` 会转发给服务端。
- 环境精确 reset 到该任务。

服务端管理任务：

- 数据集行保留 `task_name=None`。
- trainer 转发 `uid`。
- `EnvSessionManager` 通过确定性的 `TaskSelector` 将
  `uid -> (task_name, group_seed)`。
- 使用同一个 `uid` 的所有样本共享相同任务和 seed。

这样可以把任务 curriculum 策略保留在 AgentArk 侧；即使任务选择逻辑演进，RL 框架
代码也能保持稳定。

## 环境韧性

长时间训练会反复 reset Unity 并执行动态编译的代码。环境服务包含多项保护：

- 为阻塞的 Unity `reset`、`step` 和 `close` 调用设置硬超时。
- 丢弃并重建损坏的运行时。
- HTTP client 使用退避机制重试瞬时故障。
- verl agent loop 会把环境故障转换为有效的失败 rollout，而不是中止整个训练 step。

使用 soak test 对环境服务进行压力测试：

```bash
python -m agent_ark.tools.env_soak_test \
  --workers 6 \
  --rounds 150 \
  --fault-mode gentle
```
