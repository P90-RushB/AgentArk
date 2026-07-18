# 环境安装

[English](setup.md) | 简体中文

本指南介绍如何安装 Python wrapper 和打包后的 AgentArk Unity 运行时。Python 包位于
本仓库；运行时构建、任务 Mod 和记录位于 Hugging Face dataset：

`https://huggingface.co/datasets/P90-RushB/AgentArk`

该 dataset 提供带版本的 `manifest.json` 和 `registry/*.jsonl`。请优先读取这些
registry 文件，而不是硬编码目录扫描。当前版本为 `env-1.0.1`。

## Python

AgentArk 运行时、评测、回放和环境服务进程需要 Python 3.10.12 或更早的 Python
3.10 patch 版本，推荐使用 Python 3.10.12。建议使用 `uv` 创建虚拟环境，这样即使
系统已经安装其他 Python 版本，也能安装所需的 patch 版本。

Linux/macOS shell：

```bash
git clone https://github.com/P90-RushB/AgentArk.git
cd AgentArk
python -m pip install -U uv
uv venv .venv --python 3.10.12
source .venv/bin/activate
uv pip install -e .
```

Windows `cmd`：

```bat
git clone https://github.com/P90-RushB/AgentArk.git
cd AgentArk

REM 使用任意可用 Python 安装 uv，再创建 Python 3.10.12 venv。
python -m pip install -U uv
uv venv .venv --python 3.10.12

.\.venv\Scripts\activate
uv pip install -e .
```

如果要在不安装包的情况下直接调试源码，请设置 `PYTHONPATH=src`。

## 下载运行时

当前打包运行时包含 32 个首发任务。可以通过直链或 Hugging Face CLI 下载对应系统的
运行时。

### 直接下载

- Linux：`https://huggingface.co/datasets/P90-RushB/AgentArk/resolve/main/artifacts/envs/1.0.1/linux/AgentArk-env-1.0.1-linux.zip`
- Windows：`https://huggingface.co/datasets/P90-RushB/AgentArk/resolve/main/artifacts/envs/1.0.1/windows/AgentArk-env-1.0.1-windows.zip`

### Hugging Face CLI

如果系统中没有 `hf`，请先安装 CLI：

```bash
uv pip install -U huggingface_hub
```

Linux：

```bash
hf download P90-RushB/AgentArk artifacts/envs/1.0.1/linux/AgentArk-env-1.0.1-linux.zip \
  --type dataset \
  --local-dir downloads/agentark-assets
```

Windows `cmd`：

```bat
hf download P90-RushB/AgentArk artifacts/envs/1.0.1/windows/AgentArk-env-1.0.1-windows.zip --type dataset --local-dir downloads/agentark-assets
```

解压 zip。Linux 下如有需要，请为 Unity 可执行文件添加执行权限：

```bash
chmod -R 755 /path/to/AgentArk-env-1.0.1-linux
```

## 配置本地运行时路径

将 `.env.example` 复制为 `.env`：

```bash
cp .env.example .env
```

Linux 示例：

```dotenv
AGENTARK_ENV_PATH=/path/to/AgentArk-env-1.0.1-linux/AgentArk.x86_64
AGENTARK_MOD_PATH=/path/to/AgentArk-env-1.0.1-linux/AgentArk_Data/Resources/Mods
AGENTARK_TASK_STORE_PATH=${AGENTARK_MOD_PATH}/all_tasks
AGENTARK_RUNTIME_TEMPLATE_ROOT=/path/to/AgentArk-env-1.0.1-linux
AGENTARK_RUNTIME_POOL_ROOT=/tmp/agentark_runtime_pool
MLAGENTS_PYTHON_BIN=/path/to/agentark/.venv/bin/python
```

Windows 示例：

```dotenv
AGENTARK_ENV_PATH=C:\path\to\AgentArk-env-1.0.1-windows
AGENTARK_MOD_PATH=C:\path\to\AgentArk-env-1.0.1-windows\AgentArk_Data\Resources\Mods
AGENTARK_TASK_STORE_PATH=${AGENTARK_MOD_PATH}\all_tasks
AGENTARK_RUNTIME_TEMPLATE_ROOT=C:\path\to\AgentArk-env-1.0.1-windows
AGENTARK_RUNTIME_POOL_ROOT=C:\path\to\agentark-runtime-pool
MLAGENTS_PYTHON_BIN=C:\path\to\AgentArk\.venv\Scripts\python.exe
```

下面两个变量是可选项，只用于同时开发多个 Unity 工程或 Git worktree。
普通 AgentArk 用户无需添加；两者都未设置时，端口解析与修改前保持一致，
继续使用有效的 Mods 配置和原有默认值：

```dotenv
AGENTARK_EDITOR_BASE_PORT=5004
AGENTARK_PLAYER_BASE_PORT=5005
```

当 `env_path` 为 `None`、Python 连接 Unity Editor 时，统一使用
`AGENTARK_EDITOR_BASE_PORT`；启动打包程序时统一使用
`AGENTARK_PLAYER_BASE_PORT`。显式的 `env_cfg.base_port` 或命令行
`--base-port` 仍然具有更高优先级。因此，每个 worktree 只需维护自己的
`.env`，一般的评测和回放 YAML 不再需要重复填写 Player 端口。

这里的 `env_cfg.base_port` 是 Python 调用方或评测 YAML 显式传入的顶层值，
不是 Mods `config.yaml` 中的 `base_port`。AgentArk 自带的运行时和评测配置
默认不会设置顶层 `env_cfg.base_port`；worktree 的可选环境变量优先于从 Mods
读取的端口。标准多 worktree 流程应保持 `env_cfg.base_port` 和 `--base-port`
未设置，让各自 `.env` 决定端口；只有确实要覆盖某一次运行时才显式填写。

Python 包会从当前目录或其父目录中自动加载最近的 `.env`。显式设置的 shell/CI
环境变量优先级更高。设置 `AGENTARK_AUTO_LOAD_DOTENV=0` 可关闭自动加载。

## 无桌面的 Linux 环境

视觉观测需要 display。无桌面会话的 Linux 服务器应安装 Xvfb：

```bash
sudo apt update
sudo apt install -y xvfb
```

然后编辑 `AGENTARK_MOD_PATH` 所指目录中的 `config.yaml`，设置：

```yaml
virtual_display: true
```

环境服务工作流还应在对应的 AgentArk server 配置中保持
`virtual_display: true`。示例 server 配置已在 `env_cfg.env_config_overrides`
中启用该选项。

## 冒烟测试

先测试一个打包任务。`ObjectRotationMatch` 体量较小，适合检查运行时能否启动。

Linux/macOS：

```bash
python -m agent_ark.ark_env.ark_sub_env \
  --task-name ObjectRotationMatch \
  --group-seed 1 \
  --env-id 0 \
  --skip-step
```

Windows `cmd`：

```bat
python -m agent_ark.ark_env.ark_sub_env --task-name ObjectRotationMatch --group-seed 1 --env-id 0 --skip-step
```

如果 reset 成功，再用一个 tool call 执行一步：

Linux/macOS：

```bash
python -m agent_ark.ark_env.ark_sub_env \
  --task-name ObjectRotationMatch \
  --group-seed 1 \
  --env-id 0 \
  --max-steps 1 \
  --action '<tool_call>{"name":"RotateControlled","arguments":{"axis":"Y","degrees":0}}</tool_call>'
```

Windows `cmd`：

```bat
python -m agent_ark.ark_env.ark_sub_env --task-name ObjectRotationMatch --group-seed 1 --env-id 0 --max-steps 1 --action "<tool_call>{\"name\":\"RotateControlled\",\"arguments\":{\"axis\":\"Y\",\"degrees\":0}}</tool_call>"
```

## 浏览任务

[AgentArk Hub](https://p90-rushb.github.io/agentark-hub/) 是已发布 AgentArk 任务的
公开浏览器，提供任务卡片、任务详情页、预览媒体、公开聚合排行榜以及可下载 artifacts
的链接。编辑评测配置前，可先在这里选择 `task_name`。

持续增加的一部分任务也发布在
[Kaggle AgentArk Bench](https://www.kaggle.com/benchmarks/xunyiljg/agentark-bench)，
可直接用于 Kaggle 托管的模型评测。

## 额外任务 Mod

打包运行时已经包含 32 个首发任务。额外任务 Mod archive 列在
`registry/tasks.jsonl` 中，每个任务和平台对应一行。

下载 registry：

```bash
hf download P90-RushB/AgentArk \
  --type dataset \
  --include registry/tasks.jsonl \
  --local-dir downloads/agentark-assets
```

每个任务 archive 包含 `catalog_*.json`、`cfg/task_config.yaml`、任务 bundle 和
任务 DLL 等文件。将 archive 解压到：

```text
<runtime>/AgentArk_Data/Resources/Mods/all_tasks/<TaskName>/
```

配置评测或训练时，请使用 registry 记录中的 `task_name`。

## 运行时池

并行评测和强化学习训练应使用沙箱化的运行时池。每个 worker 获得私有的可写运行时，
同时通过 `Mods/all_tasks` 共享任务资源。

可以显式准备运行时池：

```bash
python -m agent_ark.tools.prepare_runtime_pool \
  --runtime-platform linux \
  --template-root "$AGENTARK_RUNTIME_TEMPLATE_ROOT" \
  --template-env-path "$AGENTARK_ENV_PATH" \
  --template-mod-path "$AGENTARK_MOD_PATH" \
  --shared-task-store-path "$AGENTARK_TASK_STORE_PATH" \
  --pool-root "$AGENTARK_RUNTIME_POOL_ROOT" \
  --pool-size 8 \
  --link-mode auto
```

多数配置也支持 `runtime_sandbox.auto_prepare: true`，让 launcher 在启动时自动准备
运行时池。
