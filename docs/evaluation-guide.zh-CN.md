# 模型评测与回放

[English](evaluation-guide.md) | 简体中文

本指南介绍本仓库中的本地模型评测流程。AgentArk 运行时安装请参阅
[setup.zh-CN.md](setup.zh-CN.md)；公开任务页面和聚合排行榜见
[AgentArk Hub](https://p90-rushb.github.io/agentark-hub/)。

## 评测入口

根据目标选择以下脚本：

| 目标 | 命令 |
| --- | --- |
| 在一个本地环境中运行一个任务的一个或多个 case | `python -m agent_ark.ark_eval.run_api_agent --config config/ark_env/eval_seed1.example.yaml` |
| 在多个环境中运行一个任务的多个 seed/model job | `python -m agent_ark.ark_eval.run_parallel_api_eval --config config/ark_env/parallel_api_eval.example.yaml` |
| 不调用 API，回放已保存的模型动作 | `python -m agent_ark.ark_eval.run_replay --config config/ark_env/replay.example.yaml` |

请先使用 `run_api_agent`。只有单环境可以稳定 reset 和 step 后，再使用并行评测。

## 配置层级

AgentArk 评测 YAML 主要包含四部分：

- `env_cfg`：运行时路径、任务类型和可选的运行时沙箱配置。
- `env_cfg.env_config_overrides`：对运行时 `Mods/config.yaml` 的临时覆盖，例如
  `num_parallel_envs`、`virtual_display` 或 history 设置。
- `eval`：case 选择、seed 范围、输出路径和并行度。
- `models`：OpenAI 兼容的模型 provider；本地 Codex SDK 评测使用
  `provider: codex`。

评测时保持 `env_cfg.env_config_overrides.num_parallel_envs: 1`。并行评分通过启动多个
独立的 `ArkEnv` 实例实现，而不是增加单个 `ArkEnv` 内的 sub-env 数量。

## 单任务评测

编辑 [config/ark_env/eval_seed1.example.yaml](../config/ark_env/eval_seed1.example.yaml)：

```yaml
eval:
  output_path: tmp/marble_stop_seed1.jsonl
  cases:
    - case_id: marble-stop-seed-0001
      task_name: MarbleStop
      group_seed: 1
      env_id: 0
```

然后选择一种模型 provider。

### OpenAI 兼容 Provider

此方式适用于 OpenAI、OpenRouter、DashScope 兼容端点和其他 OpenAI 兼容的 HTTP
端点。`APIAgent` 会以 chat-completions 风格请求运行这些配置。

```yaml
models:
  - name: openrouter-model
    provider: openrouter
    model: replace-with-openrouter-model-id
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
    temperature: 0.0
```

根据端点替换 `provider`、`model`、`base_url` 和 `api_key_env`。也可以在本地私有
配置中直接设置 `api_key`。如果模型或代理提示 `temperature` 已弃用或不受支持，
请设置 `temperature: null`，让请求省略该参数。

对于无状态 HTTP provider，建议返回完整消息上下文：

```yaml
env_cfg:
  env_config_overrides:
    env_wrapper_cfg:
      context_manager:
        messages:
          enabled: true
          only_return_messages: true
          append_only: true
          return_mode: full
```

Kaggle 上的 [AgentArk Bench](https://www.kaggle.com/benchmarks/xunyiljg/agentark-bench)
通过 Kaggle Model Proxy 运行。其排行榜任务在端点允许时使用 `temperature: 1.0`，
不允许指定该参数时使用 `null`。这与 AgentArk Hub 最初使用 `temperature: 0.0` 的
本地结果属于不同评测设置。

### Codex SDK Provider

此方式用于本地 Codex SDK 评测，不是 OpenAI 兼容 HTTP 端点；
`provider: codex` 不使用 `base_url`、`api_key` 或 `api_key_env`。

安装包含 `codex` extra 的 AgentArk：

```bash
python -m pip install -U pip
python -m pip install -e ".[codex]"
```

如果软件源尚未镜像 beta Codex SDK，请从 PyPI 显式安装：

```bash
python -m pip install -i https://pypi.org/simple openai-codex
```

```yaml
models:
  - name: codex-gpt55
    provider: codex
    model: gpt-5.5
    sandbox: read_only
    timeout_s: 600
    # 可选：none、minimal、low、medium、high、xhigh。
    # 省略时使用 Codex SDK/模型默认值。low 是更稳妥的低成本设置；
    # 启用 Codex tools 时 minimal 可能被拒绝。
    reasoning_effort: low
    thread_mode: per_agent
```

AgentArk 会把 OpenAI 风格的评测消息转换成 Codex 文本和图像输入。Data URI 图像观测
会直接作为 Codex `ImageInput` 传入。需要更低成本或更快响应时，可以用
`reasoning_effort` 覆盖 Codex turn 的推理深度。

`thread_mode` 默认为 `per_agent`，推荐使用该设置。除了 AgentArk 可见的
`request_messages`，Codex SDK turn 还包含 Codex 侧运行时、系统和工具上下文；这些
内容不会由环境显示，但会计入 Codex token 用量。本地 MarbleStop 检查显示，即使首个
turn 只有数千个可见文本字符和一张图片，报告的输入 token 也可能达到数万；而
`per_agent` 在后续 turn 中有更高的缓存 token 复用率。持久 thread 可以复用 Codex
自身初始化和 reset/base prompt，环境随后只需发送 step delta。

推荐的有状态 Codex 模式使用增量消息上下文：

```yaml
env_cfg:
  env_config_overrides:
    env_wrapper_cfg:
      context_manager:
        messages:
          enabled: true
          only_return_messages: true
          append_only: true
          return_mode: delta
          history_only_on_attempt_start: true
```

如果 Codex 使用 `thread_mode: per_turn`，请改用 `return_mode: full`。使用
`thread_mode: per_agent` 时，组合 `append_only: true` 与 `return_mode: delta`，
让持久 Codex thread 在 reset/base prompt 后只接收新的 step delta。

运行：

```bash
python -m agent_ark.ark_eval.run_api_agent \
  --config config/ark_env/eval_seed1.example.yaml
```

`task_name` 必须匹配 `Mods/all_tasks` 下的目录。打包运行时包含 32 个首发任务，
例如 `MarbleStop`、`Snake`、`Pushbox`、`ObjectRotationMatch` 和
`StarterRouteJump3D`。

### 黑盒玩家反馈

使用 [player_feedback.example.yaml](../config/ark_env/player_feedback.example.yaml)，
可以让同一个 Codex SDK thread 游玩打包任务，并在 rollout 后报告玩家可观察到的任务
质量问题。玩家只接收正常任务提示、图像观测、可见/history 消息和动作结果；不会注入
任务源码、隐藏状态、oracle 动作、设计说明或 reviewer evidence。

实际游玩循环应使用 SDK player，而不是普通 workflow subagent。SDK adapter 已经把
AgentArk 多模态消息作为原生图像输入处理，在多个 step/attempt 之间保留同一个玩家
thread，并记录精确的任务、seed、动作、结果、轨迹和回放证据。普通 workflow
subagent 通常继承仓库及设计/审查上下文，适合启动运行和检查产物，但不适合替黑盒玩家
选择动作。

```bash
python -m agent_ark.ark_eval.run_api_agent \
  --config config/ark_env/player_feedback.example.yaml
```

模型项必须使用 `provider: codex`、`thread_mode: per_agent`，并启用：

```yaml
player_feedback:
  enabled: true
```

此模式会自动增加两项保护：

- 动作提示要求 Codex 只使用玩家可见消息和图像。
- Codex thread 从一个新的空临时 cwd 启动，而不是从源码仓库启动；agent 关闭时会删除
  该目录。

终止观测后，`run_api_agent` 会要求同一 thread 输出一个结构化
`<player_feedback>` JSON 报告。报告区分具体任务缺陷与未解出任务、难度、玩家错误或
合理探索需求等非缺陷，结果记录将其保存在 `player_feedback` 下。请启用
`trajectory_save`，设置 `condition: all` 和 `include_images: true`，确保即使玩家没有
成功，报告的问题也能回放。如果最终报告缺失、失败或不符合 JSON schema，评测结果会
标记为 `status: error`，避免 resume/retry 逻辑误把玩家 gate 当作已完成；rollout 和
轨迹证据仍会保留。

报告还包含 `information_reveal_assessment`，其玩家侧分类为以下四种之一：
`complete_initially`、`intentional_exploration`、
`suspected_missing_information_defect` 或 `unclear`，并记录玩家实际考虑的可见证据和
attempt index。`ArkEnv` 可能有意让早期 attempt 用于探索，并在后续 attempt 边界展示
后果或配置的 history。因此，只要渐进发现符合设计且有效，初始不确定性并不是缺陷。
只有承诺或必需的信息始终无法发现，或实际信息揭示行为违背玩家可见约定时，才构成
疑似缺陷。

玩家分类只是证据，不是最终结论，也不会直接请求源码修改。workflow 会把解析后的报告
和包含图像的回放交给 task reviewer。reviewer 独立对照已批准的信息揭示设计、
task description、实际 `max_attempts`、seed/history 语义和打包回放，再把候选问题
分类为 `confirmed defect`、`non-defect` 或 `inconclusive`。只有确认的缺陷才交给
builder；结论不明确时可追加一次定向运行；有意探索仍记录为非缺陷。

串行和并行 runner 都会拒绝缺少以下设置的玩家反馈配置：
`trajectory_save.enabled: true`、`condition: all`、`include_images: true` 和输出路径。
该模式还拒绝人工交互替代、非 Codex provider、非只读沙箱、显式源码 cwd 以及无状态
Codex thread。

隔离 cwd 和提示词只能降低污染，并非严格安全边界：当前 Codex SDK adapter 没有禁用
所有文件系统或 shell 工具。除非部署的 SDK/运行时也强制执行更严格的工具限制，否则
不要把运行描述为严格防源码泄漏的黑盒测试。

## 多个 Seed

使用 [config/ark_env/eval_seeds_1_n.example.yaml](../config/ark_env/eval_seeds_1_n.example.yaml)：

```yaml
eval:
  task_names:
    - MarbleStop
  group_seeds:
    start: 1
    end: 5
```

运行：

```bash
python -m agent_ark.ark_eval.run_api_agent \
  --config config/ark_env/eval_seeds_1_n.example.yaml
```

也可以提供显式列表：

```yaml
eval:
  task_names:
    - MarbleStop
  group_seeds: [1, 2, 3, 10, 20]
```

如果 `eval.cases` 非空，它的优先级最高；每个 case 都可以指定精确的 `task_name`、
`group_seed` 和 `env_id`。

## 并行 API 评测

并行评测会把多个独立 seed/model job 分发到多个 Unity 运行时：

```bash
python -m agent_ark.ark_eval.run_parallel_api_eval \
  --config config/ark_env/parallel_api_eval.example.yaml
```

重要字段：

- `eval.max_parallel_envs`：同时存活的 `ArkEnv` 实例上限。
- `eval.worker_index_base`：分配给 slot 0 的 worker index。
- `eval.task_names` 或 `eval.cases`：当前 runner 期望每次 eval run 只包含一个任务。
- `models`：每个模型都会在每个选定 seed/case 上评测。
- `env_cfg.runtime_sandbox`：`max_parallel_envs > 1` 时保持启用。

运行时沙箱为每个 worker 提供私有可写的 runtime/Mods 目录，同时通过
`Mods/all_tasks` 共享任务资源，避免 Unity 或 Python 重写活动配置和 bundle 文件时
发生竞态。

## 浏览器 Viewer 与人工动作

启用本地 chat viewer：

```yaml
hooks:
  visualization:
    enabled: true
    host: 127.0.0.1
    port: 18181
    open_browser: true
    keep_open_on_end: true
```

viewer 会展示发送给模型的消息、观测图像和 assistant 响应，适合调试提示词和任务行为。

手动调试任务时，再启用：

```yaml
hooks:
  human_interaction:
    enabled: true
```

此模式不会调用模型 API，而是在浏览器中手动输入任务要求格式的动作，例如：

```xml
<tool_call>{"name":"RotateControlled","arguments":{"axis":"Y","degrees":20}}</tool_call>
```

人工交互适合单环境调试，不适合无人值守的并行评测。

## 动作模式

AgentArk 任务在运行时配置中选择动作模式。

`action_mode: func` 接收结构化 tool call：

```xml
<tool_call>{"name":"ExecutePlan","arguments":{"plan":"L4,U7"}}</tool_call>
```

任务提示会暴露工具文档。Python 验证工具名称和参数，渲染最小 C# 调用，再发送给
Unity/Roslyn。

`action_mode: code` 接收完整 C# 脚本：

```xml
<code>
using UnityEngine;

public class ArkAct_Step0 : MonoBehaviour
{
    void Start()
    {
        var router = GetComponent<ActRouter>();
        router.Call("ExecutePlan", "L4,U7");
    }
}
</code>
```

能使用 tool-call 模式时应优先使用。需要跨帧控制、更丰富的程序逻辑或多次调用任务 API
时再使用完整代码模式。

## 评分

评测会运行到实际任务配置允许的最后一次 attempt；较早 attempt 成功不会终止 rollout。

主要结果字段：

- `score_reward`：case 的正式分数，等于最后一次 attempt 的 reward。
- `best_attempt_reward`：所有 attempt 中最高 reward，仅用于诊断。
- `ever_attempt_success`：是否有任意 attempt 成功。
- `final_attempt_success` / `rollout_success`：最后一次 attempt 是否成功。
- `first_success_attempt_index`：首次成功的 attempt index（如果存在）。
- `rollout_truncated`：rollout 是否耗尽配置预算。

`total_reward` 为调试保留，不是主要评分。

## 回放

回放会复用 JSONL 评测记录中保存的动作并在环境中重新执行，不调用模型 API。

```bash
python -m agent_ark.ark_eval.run_replay \
  --config config/ark_env/replay.example.yaml \
  --records tmp/delay_train_seed1_5.jsonl \
  --index 0
```

常用回放配置：

```yaml
replay:
  records_path: tmp/delay_train_seed1_5.jsonl
  record_index: 0
  step_delay_s: 0.5
  use_record_max_attempts: true
  require_match: true
```

常用 selector 包括 `record_index`、`record_indices`、`record_lines`、`case_id`、
`model_name`、`task_name`、`group_seed`、`env_id` 和 `limit`。

启用浏览器 viewer 时回放最有价值，因为图像会由运行时重新生成，而不是从原评测记录
读取。

## 轨迹保存与加载

评测可以把 attempt 前缀保存到 trajectory bank：

```yaml
eval:
  trajectory_save:
    enabled: true
    output_path: tmp/trajectory_bank.jsonl
    condition: ever_success
    prefix_attempts: 4
    include_images: true
```

之后加载这些前缀：

```yaml
eval:
  trajectory_load:
    enabled: true
    path: tmp/trajectory_bank.jsonl
    prefix_attempts: 4
```

这会恢复 attempt 边界上的 history 上下文，但不会恢复 step 中途的物理状态。
