# AgentArk VERL GRPO 接入指南

[English](README.md) | 简体中文

本目录是 AgentArk 仓库内的 VERL 接入入口。Unity runtime、sandbox pool、Env
Server 和 HTTP v1 API 由本仓库提供；dataset、`AgentArkEnvAgentLoop`、VERL 配置和
训练 launcher 位于公开的
[`P90-RushB/verl` `agentark_rl` 分支](https://github.com/P90-RushB/verl/tree/agentark_rl/agentark_recipe/agentark_env_agent)。
两端通过 HTTP 通信，VERL Python 环境不需要安装或导入 `agent_ark`。

本地桥接脚本会：

- 离线检查外部 checkout 是否包含已审计的 AgentArk recipe 和 HTTP v1 contract；
- 解析外部 recipe 实际使用的 `env_cfg.yaml`，再用完全相同的 `env_cfg` 预热 v1 pool；
- 在调用外部 launcher 前检查单机拓扑和 rollout 分片，安全导入 AgentArk 路径，
  并修正当前 launcher 的 `TOTAL_TRAINING_STEPS=-1` 行为。

当前兼容基线和必需文件记录在 [compatibility.json](compatibility.json)。兼容检查只证明
代码谱系和 contract 结构，不替代真实 Unity rollout。

## 1. 前置条件与 checkout

先按 [AgentArk 安装指南](../../docs/setup.zh-CN.md) 在 AgentArk Python 3.10.12
环境中跑通一次真实 Unity 评测。VERL 使用独立的 Python 环境；按 VERL 仓库说明安装
其训练依赖，不要把 VERL、Ray、vLLM 依赖装进 AgentArk 环境。

```bash
git clone --branch agentark_rl --single-branch \
  https://github.com/P90-RushB/verl.git /absolute/path/to/verl

export AGENTARK_REPO_ROOT=/absolute/path/to/AgentArk
export VERL_ROOT=/absolute/path/to/verl
cd "$AGENTARK_REPO_ROOT"

python integrations/verl/check_compatibility.py --checkout "$VERL_ROOT"
```

返回 `[COMPATIBLE]` 后继续。CI 或发布检查可加 `--strict`；普通开发 checkout 中，
推荐分支之后的 recipe 修改会给出 warning，并要求重新做真实 Unity smoke。

## 2. 规划 v1 环境容量

当前 VERL recipe 一次 generation batch 的峰值 lease 数为：

```text
ENV_CONCURRENCY = TRAIN_BATCH_SIZE * ROLLOUT_N
```

并满足：

```text
ENV_CONCURRENCY % AGENT_NUM_WORKERS == 0
```

`AGENT_NUM_WORKERS` 负责把 batch 切给 Ray actor，不是环境并发上限。环境不足时 rollout
会等待并分波执行。首次 smoke 可使用
`TRAIN_BATCH_SIZE=1`、`ROLLOUT_N=8`、`AGENT_NUM_WORKERS=4`，因此预热 8 个环境。

外部配置中的 `runtime_sandbox.pool_size` 是初始准备尺寸，不是硬上限；启用
`auto_prepare` 时，AgentArk 会按新的 `worker_index` 扩展 sandbox。预热脚本会保留该值，
因为擅自改动任何 `env_cfg` 字段都会产生另一个 pool 指纹。

## 3. Terminal A：启动 Env Server

在 AgentArk Python 环境中运行，并保持这个终端常驻：

```bash
export AGENTARK_REPO_ROOT=/absolute/path/to/AgentArk
cd "$AGENTARK_REPO_ROOT"
./integrations/verl/scripts/run_agentark_server.sh
```

“单 Server”表示同一 host/port 只运行一个 FastAPI 服务进程；它仍会通过既有的 runtime
sandbox 机制并发管理多个 Unity 子进程。不要用多个 uvicorn worker 启动同一服务。
当前外部 recipe 连接 `127.0.0.1:18080`，因此本指南对应单机训练。

## 4. Terminal B：用 VERL 的精确配置预热 v1 pool

打开另一个 AgentArk Python 终端：

```bash
export AGENTARK_REPO_ROOT=/absolute/path/to/AgentArk
export VERL_ROOT=/absolute/path/to/verl
cd "$AGENTARK_REPO_ROOT"

./integrations/verl/scripts/warmup_agentark_v1.sh \
  --verl-root "$VERL_ROOT" \
  --num-envs 8
```

脚本先用 OmegaConf 解析
`$VERL_ROOT/agentark_recipe/agentark_env_agent/config/env_cfg.yaml` 中的
`${oc.env:...}`，再生成被忽略的临时文件
`tmp/verl_v1_runtime_config.json`，最后显式以 `--protocol-version v1` 预热。
这避免了用 AgentArk 通用示例配置预热、训练却因配置指纹不同而重新冷启动一池的情况。

检查成功状态：

```bash
curl -fsS http://127.0.0.1:18080/health
curl -fsS http://127.0.0.1:18080/v1/envs
```

应满足：

- `tmp/verl_v1_warmup_snapshot.json` 中 `protocol_version` 为 `v1`；
- `/health` 返回 `"ok": true`；
- `/v1/envs` 至少有计划容量的条目，且对应条目的
  `protocol_namespace="v1"`、`started=true`、`in_use=false`。

v1 与 ms-swift 默认使用的 v2 是隔离的 pool namespace；v2 health/capabilities 正常不能
证明 VERL 所需的 v1 pool 已准备好。

## 5. Terminal C：生成 dataset

切换到独立的 VERL Python 环境，并从 VERL 仓库根目录运行：

```bash
cd "$VERL_ROOT"
export DATA_DIR=/absolute/path/to/agentark_data

python agentark_recipe/agentark_env_agent/generate_agentark_dataset.py \
  --local-save-dir "$DATA_DIR" \
  --num-train 1000 \
  --num-test 200 \
  --seed 1234
```

即使设置 `TEST_FREQ=-1`，当前 VERL trainer 仍会创建 validation dataloader，因而
`test.parquet` 必须存在且非空。若使用固定训练步数，建议至少满足：

```text
num_train >= TOTAL_TRAINING_STEPS * TRAIN_BATCH_SIZE
```

默认 dataset 不固定 `task_name`。当前实现中，VERL 为每次消费的 prompt group 生成
`uid`，服务端用 `uid` 选择 task；dataset row 中显式的 `group_seed` 决定 seed。同一个
GRPO group 的 sibling trajectories 共享 task 和 seed。跨 epoch 再次消费同一 row 时
会生成新 `uid`，因此 task 可以变化，row seed 保持不变。

## 6. 启动训练

模型、GPU 数量、tensor parallel、FSDP 和 vLLM 参数以外部 recipe 的
[README](https://github.com/P90-RushB/verl/tree/agentark_rl/agentark_recipe/agentark_env_agent)
和 launcher 为准。当前 launcher 名称及默认配置针对 Qwen3.5-9B/FSDP2/vLLM；更换模型
或拓扑时应在 VERL 侧验证相应配置。

下面通过本仓库的安全 wrapper 启动，而不是直接调用外部 shell：

```bash
export AGENTARK_REPO_ROOT=/absolute/path/to/AgentArk
export VERL_ROOT=/absolute/path/to/verl
export MODEL_PATH=/absolute/path/to/model
export DATA_DIR=/absolute/path/to/agentark_data
export CKPT_DIR=/absolute/path/to/checkpoints/agentark_verl

cd "$AGENTARK_REPO_ROOT"
PYTHON="/absolute/path/to/verl-env/bin/python" \
NNODES=1 NGPUS_PER_NODE=8 \
ROLLOUT_NAME=vllm ROLLOUT_TP=8 \
TRAIN_BATCH_SIZE=1 ROLLOUT_N=8 AGENT_NUM_WORKERS=4 \
PPO_MINI_BATCH_SIZE=1 \
TOTAL_EPOCHS=1 TOTAL_TRAINING_STEPS=10 \
SAVE_FREQ=5 TEST_FREQ=-1 \
TRAINER_LOGGER='["console","tensorboard"]' \
TENSORBOARD_DIR=/absolute/path/to/tensorboard/agentark_verl \
./integrations/verl/scripts/run_verl_training.sh \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.resume_mode=disable \
2>&1 | tee /tmp/agentark_verl_train.log
```

wrapper 会先运行兼容检查，并确认：

- `NNODES=1`；当前 recipe 的 Server 和 distributed init 地址都是本机地址；
- `TRAIN_BATCH_SIZE * ROLLOUT_N` 能被 `AGENT_NUM_WORKERS` 整除；
- `train.parquet` 和 `test.parquet` 都非空；
- 外部 launcher 只接收五个 `AGENTARK_*` runtime 路径，不会 source 含其他 API key 的
  AgentArk `.env`；
- `TOTAL_TRAINING_STEPS=-1` 被转换为 Hydra `null`，由 dataloader 和 epoch 推导总步数；
  正整数保持固定步数。也可以显式传 `TOTAL_TRAINING_STEPS=null`。

只检查 checkout、dataset、运行时变量和参数而不启动训练时，把
`--preflight-only` 作为 wrapper 的第一个参数。

首次训练的成功标志是：日志显示预期的 total training steps，并完成至少一次
multimodal rollout、reward 计算和 actor update，而不是在 global step 1 直接退出。
达到保存间隔后，`CKPT_DIR` 下应出现 `global_step_*` checkpoint。

## 7. 中断与恢复

正常退出会在 `finally` 中尝试 release v1 env。若 trainer 被硬中断，v1 没有 v2 的
lease TTL 和 operation-ID 安全重放：先停止并重启 Terminal A，再重新执行 Terminal B，
然后用完全相同的训练配置恢复。

自动恢复：

```bash
./integrations/verl/scripts/run_verl_training.sh \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.resume_mode=auto
```

指定 checkpoint：

```bash
./integrations/verl/scripts/run_verl_training.sh \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="$CKPT_DIR/global_step_5"
```

实际恢复时仍需带上首次运行的模型、batch、rollout、GPU 和日志环境变量。确认日志从指定
`global_step_*` 接续，而不是创建新的 step 0 训练。

## 8. 问题定位

- Unity reset/step、sandbox、pool、task selection 或 Env Server：在本仓库排查。
- VERL import、Ray、vLLM、FSDP、Hydra、dataset 或 trainer：在外部 fork 排查。
- checkout/协议漂移：运行 `check_compatibility.py`；它完全离线、只读，不会 fetch、
  checkout、import 或执行外部代码。退出码 `0` 表示结构兼容，`1` 表示明确不兼容，
  `2` 表示无法判断（例如浅克隆缺少基线提交）。

架构、GRPO 分组语义与 v1/v2 差异见
[强化学习训练说明](../../docs/rl-training.zh-CN.md)。
