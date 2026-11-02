# 使用 LiteLLM 后端评测 SimpleAgent

[English](litellm-evaluation.md) | 简体中文

LiteLLM 是 SimpleAgent 正式支持的可选模型后端，沿用相同的评测入口、环境观察、
宿主工具、上下文预算、压缩和动作提交流程。已有配置仍默认使用
`simple_agent.backend: native`；安装可选依赖不会自动切换后端。运行时安装和路径见
[安装指南](setup.zh-CN.md)，harness、并行评测和回放见[评测指南](evaluation-guide.zh-CN.md)。

## 分清 harness、后端、API 和供应商

| 配置 | 含义 |
| --- | --- |
| `agent_type: simple` | 选择 SimpleAgent harness。省略时仍沿用历史 APIAgent/CodexAgent 选择规则。 |
| `simple_agent.backend` | 默认 `native`；显式设为 `litellm` 才使用可选 SDK。 |
| `simple_agent.litellm.model` | 必填的 LiteLLM `provider/model` 路由，实际 SDK 供应商由这里的前缀决定。 |
| `simple_agent.litellm.api` | 默认 `chat_completions`，调用 `acompletion`；`responses` 调用 `aresponses`。 |
| 顶层 `provider` | 保留评测器的 provider 标签并参与评测身份，不负责选择 LiteLLM 供应商；不能与 `codex` 组合。 |
| 顶层 `model` | 记录在结果中的展示／评测标签。本文示例填写供应商实际接收的模型名，Kaggle 尤其如此；SDK 调用路由由内层模型决定。 |
| 顶层 `base_url` | 可选的供应商／代理 API 基址。省略时采用 SDK 供应商默认地址，不继承原生后端历史上的 DashScope 默认地址。 |

Chat Completions 可使用固定版本 SDK 支持的 Chat 供应商，但具体模型仍须支持任务需要的
视觉、函数工具和参数。Responses 目前要求 `openai/` 路由，也可指向兼容 OpenAI 的代理。
AgentArk 直接调用原生 `aresponses`，不使用 Chat→Responses 桥接。后端本身不会从 URL
或展示模型名推断供应商或 API 类型。

在 `api: responses` 下，接受 `openai/responses/X` 作为 `openai/X` 的别名，并在调用和
身份计算前规范化，仍走原生 `aresponses`。该拼法在 `api: chat_completions` 下明确报错；
新配置建议使用普通 `openai/X` 并显式选择 API。

## 安装和运行

在 AgentArk Python 包根目录，使用已有的 Python 3.10 环境执行；支持版本范围为
3.10.1–3.10.12：

```bash
python -m pip install -e ".[litellm]"
python -m pip check
```

可选依赖固定为 `litellm[mcp]==1.106.6` 和 `tenacity==9.1.2`：MCP 满足 SDK 工具请求的
导入需求，tenacity 提供重试运行时。安装可能升级共享依赖，本次验证环境中的 OpenAI SDK
从 2.8.1 升级为 2.54.0。请随评测条件记录依赖版本；只有显式选用 LiteLLM 的配置才会走新后端。

按[安装指南](setup.zh-CN.md)在当前进程设置 `AGENTARK_ENV_PATH`、`AGENTARK_MOD_PATH`
和 `OPENAI_API_KEY`。以下是直接调用 OpenAI 的完整配置：

```yaml
env_cfg:
  env_path: "${AGENTARK_ENV_PATH}"
  mod_path: "${AGENTARK_MOD_PATH}"
  task_type: RLTask
  env_config_overrides:
    override_by_task: true
    num_parallel_envs: 1

eval:
  output_path: tmp/simple_agent_litellm_results.jsonl
  stop_on_error: false
  fixed_env_id: 0
  task_names: [MarbleStop]
  group_seeds:
    start: 1
    end: 2

models:
  - name: simple-astra-litellm
    agent_type: simple
    provider: openai
    model: gpt-6-astra
    api_key_env: OPENAI_API_KEY
    temperature: null
    reasoning_effort: medium
    max_completion_tokens: 30000
    max_retries: 0
    timeout_s: 600
    simple_agent:
      backend: litellm
      litellm:
        model: openai/gpt-6-astra
        api: responses
      tool_choice: required
      context:
        compaction_policy: message_chunks_v1
        output_reserve_tokens: 30000
      tools:
        enabled: true
        workspace_parent: tmp/simple_agent_litellm_workspaces
```

仓库中也提供了
[simple_agent_litellm.example.yaml](../config/ark_env/simple_agent_litellm.example.yaml)，运行：

```bash
python -m agent_ark.ark_eval.run_api_agent --config config/ark_env/simple_agent_litellm.example.yaml
```

任务必须存在于所用运行时中。环境覆盖、种子范围、轨迹保存和并行 job 沿用普通评测 schema，
不需要另写 LiteLLM runner 或重建运行时。此示例是直接 OpenAI 的配置方法；下方真实远端验证
使用的是 Kaggle 的兼容代理。

## OpenRouter Chat Completions

保留前面的 `env_cfg` 和 `eval`，将 `models` 替换为以下内容，在进程中设置
`OPENROUTER_API_KEY`。不填 `base_url` 时 SDK 可以选用 OpenRouter 默认端点，也可显式
填写兼容地址。

```yaml
models:
  - name: simple-qwen-litellm
    agent_type: simple
    provider: openrouter
    model: qwen/qwen3.7-plus
    api_key_env: OPENROUTER_API_KEY
    temperature: null
    max_completion_tokens: 30000
    max_retries: 0
    timeout_s: 600
    simple_agent:
      backend: litellm
      litellm:
        model: openrouter/qwen/qwen3.7-plus
        api: chat_completions
      output_token_parameter: max_tokens
      tool_choice: required
      context:
        output_reserve_tokens: 30000
      tools:
        enabled: true
        workspace_parent: tmp/simple_agent_litellm_workspaces
```

这是路由配置示例，不代表该模型或所有 OpenRouter 推理模式都做过真实调用验证。
仅在所选模型／SDK 路由支持时添加通用 `reasoning_effort`。已有的原生 OpenRouter 示例和
`thinking` 映射仍单独保留在[评测指南](evaluation-guide.zh-CN.md#原生-openai-兼容-provider)。

其他 Chat 供应商遵循同一规则。例如以下路由片段保留评测器的中性 `auto` 标签，而实际走
Anthropic 前缀；将 `your-claude-model-id` 替换为受支持的目标模型，并沿用完整配置中的
环境、预算和工具设置：

```yaml
provider: auto
model: your-claude-model-id
api_key_env: ANTHROPIC_API_KEY
simple_agent:
  backend: litellm
  litellm:
    model: anthropic/your-claude-model-id
    api: chat_completions
```

## Kaggle 模型代理和 Notebook

使用单独的 [agentark_litellm_task.ipynb](../benchmarks/kaggle/agentark_litellm_task.ipynb)。
它默认评测 MarbleStop seeds 1–10，开头说明也提供已验证的 Door（`DoorSpin3Turns`）
seeds 1–2 配方及 −1 到 1 的分数范围。环境仍为 `env-1.0.6`。Notebook 将 AgentArk
wrapper 源码固定到不可变 commit；发布 benchmark 时应保留该 pin 和评测条件。
已有原生 Notebook 保留各自的 pin 和行为。执行方法见
[Kaggle benchmark 说明](../benchmarks/kaggle/README.md)。

Notebook helper 为 Kaggle 的 `openai/gpt-6-astra`、`gpt-6-astra` 显式选择 Responses，
其他模型选择 Chat Completions。需要时可通过 `DEFAULT_LITELLM_API` 或
`AGENTARK_KAGGLE_LITELLM_API` 覆盖；helper 会将选定 API 写入普通 AgentArk 配置。

新 Notebook 和共享 Kaggle 模板提供 `DEFAULT_TOOL_CHOICE = None`。在 LiteLLM 模式下，
对于最后一段斜杠分隔模型名恰好为 `claude-opus-5-5` 或 `claude-opus-5.5`
（可带 `-default` 或 `@default` 后缀）的 Kaggle Opus 5.5 别名，从首次请求起写入
`tool_choice: auto`。Kaggle CLI slug `claude-opus-5-5-default` 选择的 wire model 为
`anthropic/claude-opus-5-5@default`，对应内层 LiteLLM 路由为
`openai/anthropic/claude-opus-5-5@default`。
原生模式、Astra、Gemini 和未知模型名仍使用 `required`。可将 `DEFAULT_TOOL_CHOICE` 或
`AGENTARK_KAGGLE_TOOL_CHOICE` 显式设为 `required` 或 `auto`，环境变量优先；无效值会在
生成配置时失败。这是 Kaggle Notebook／模板的映射规则，不是核心 LiteLLM 适配器的模型规则。
[Anthropic 文档](https://platform.claude.com/docs/en/api/errors)说明 Opus 5.5 不支持强制工具选择，
关闭 thinking 也不会解除这一限制。

`auto` 仍提供函数工具，模型也仍被要求使用 `submit_action`，但不强制模型调用工具。
按 SimpleAgent 既有行为，纯文字响应会结束本次决策，可能成为无效的环境动作。
本次调整不添加提醒、重试或基于异常的回退。工具选择已经包含在评测指纹中；改变条件时应使用
新的 benchmark 名称。

Notebook 返回 `normalized_mean_score_reward`：将每个 case 的原始最终尝试
`score_reward` 按 `(score - min) / (max - min)` 归一化，再对 seeds 求平均。
生成的 Python 任务返回原始 `mean_score_reward`。创建 benchmark 前须设置任务实际分数范围；
两种指标都不是累计 episode reward 或成功率。选择 LiteLLM 不改变这些公式。

手动评测应使用结尾为 `/models/openapi` 的 Kaggle OpenAPI 基址和已有代理凭据。
请把下方示例主机替换成真实地址；Notebook 会从 Kaggle 代理环境派生该地址。

```yaml
models:
  - name: kaggle-astra-litellm
    agent_type: simple
    provider: openai
    model: openai/gpt-6-astra
    base_url: https://your-proxy-host/models/openapi
    api_key_env: MODEL_PROXY_API_KEY
    temperature: null
    max_completion_tokens: 30000
    max_retries: 0
    timeout_s: 600
    simple_agent:
      backend: litellm
      litellm:
        model: openai/openai/gpt-6-astra
        api: responses
      tool_choice: required
      context:
        output_reserve_tokens: 30000
      tools:
        enabled: true
        workspace_parent: tmp/simple_agent_litellm_workspaces
```

内层模型的第一个 `openai/` 选择 LiteLLM 的 OpenAI 适配器，剩下的
`openai/gpt-6-astra` 才是 Kaggle 接收的模型名，因此双前缀是有意的。直接 OpenAI 则使用
顶层 `gpt-6-astra`、内层 `openai/gpt-6-astra`。Astra 使用 `temperature: null`；
通用 `reasoning_effort` 可选。不要把代理的 `/genai` 地址用于此 OpenAPI 路由，也不要在
Responses 基址后追加 `/chat/completions`。

## 凭据、参数和预算

- 建议显式使用 `api_key_env` 指定所需凭据，也可在本地私有配置填写 `api_key`。
  指定的环境变量不存在时，在请求前报错；两者都省略时由 LiteLLM 读取对应供应商的标准
  环境凭据。AgentArk 不会套用原生后端跨供应商的 API key 回退顺序。
- `thinking` 必须省略或为 `null`，即使顶层保留 `provider: openrouter` 或 `volcengine`
  也不会应用原生映射。可选的通用 `reasoning_effort` 交给 SDK，支持值和实际行为取决于路由。
- `temperature: null` 省略温度。Chat 的 `output_token_parameter` 可选
  `max_completion_tokens`（默认）或 `max_tokens`；Responses 总是发送 `max_output_tokens`。
  不支持的参数会明确失败，不会静默丢弃参数或换一套配置重试。
- `max_completion_tokens` 与运行时动作预算共同限制动作生成。摘要沿用独立的
  `summary_output_tokens`，默认 32768，不会因动作上限为 30000 而被降为 30000。
  推理和可见正文共用供应商的输出额度。Responses 的有效输出预算不足 16 时在模型调用前报错。
- SimpleAgent 默认不重试模型请求。`max_retries: N` 通过一层重试机制请求最多 N 次 SDK
  重试。SDK 进程级的 `drop_params`、`modify_params`、模型 fallback 和非零全局重试设置
  必须关闭，否则 AgentArk 明确拒绝，避免暗改评测条件。请求超时及 SimpleAgent 决策／调用
  预算继续有效。
- 宿主工具和 `submit_action` 保持 SimpleAgent 原有语义。对于任何支持函数工具但拒绝
  required tool choice 的端点，本地手动 YAML 配置都可在 `simple_agent` 下显式设置
  `tool_choice: auto`，作为另一个评测条件。它允许而不强制工具调用；纯文字响应
  会结束决策，可能成为无效的环境动作，不会因此额外重试。摘要没有可执行工具。
  提高预算前先阅读[完整 harness 限制](evaluation-guide.zh-CN.md#推荐-simpleagent上下文压缩与宿主工具)。

## 推理历史、压缩和 Trace

LiteLLM 适配器使用独立、有界、带类型校验的 assistant 状态，绑定 backend、API 模式、
端点和规范化后的 SDK 模型，不改动原生 HTTP 后端的 `NativeAssistantState` 契约。

Chat 已支持的状态包括 `reasoning_items`、`thinking_blocks`、`reasoning_content` 和
`provider_specific_fields`，也包括受支持的工具调用签名字段。Responses 保留受支持的有序
输出条目，包括 assistant 消息和 phase、推理／加密内容、函数调用及其 `call_id`，工具调用
仍与结果配对。这些是 SDK 返回的条目；LiteLLM 可能规范化供应商数据、略去 null 字段或修正
外来调用 ID，不能据此承诺所有供应商协议都无损。

历史仍由 SimpleAgent 管理。摘要只接收可移植的图文及工具历史，不包含不透明的原生状态；
只有摘要的最终正文成为模型记忆。成功压缩会清除**全部工作上下文中的原生状态**，包括
保留的近期消息上的状态，原始有界归档仍保留原件。因此既有策略可能在首次回传前清除推理，
这不等于未经解释的传输丢失。尚未结束的原生宿主工具续轮禁止重写其前缀；放不进预算时
按现有限制明确失败。

保存的模型请求通过 `request_transport.kind: logical_sdk_request` 标明逻辑 SDK 请求及所用的
`acompletion` 或 `aresponses` 方法，表示 AgentArk 交给 LiteLLM 的输入，不冒充最终供应商
HTTP 报文的抓包。响应元数据和原生状态与公开 assistant 正文、任务提交动作分开。
HumanView／回放可以展示保存的调用、摘要和事件，不能重建未保存的上游流量或解读不透明推理。

## 续跑、回放和已有评测

选择 LiteLLM 会形成独立评测条件。backend、规范化 SDK 模型、API 模式和端点与原有 harness
设置一起参与结果指纹。变更条件时使用新输出路径，原生结果不会被静默当作 LiteLLM 完成结果。
接受的 Responses 别名会规范化为同一 SDK 模型身份。

省略 `backend` 或显式填写 `backend: native` 均保留原生路径和既有指纹。APIAgent、
CodexAgent、SimpleAgent 原生 OpenRouter 处理和 RL 集成保留原有行为，旧记录继续可读。
保存动作的回放使用 `ReplayAgent`，不调用模型后端，因此不需要 LiteLLM extra，也不恢复模型
私有工作上下文。回放仍要求兼容任务／运行时、种子和参数；依赖升级本身不证明远端模型行为
或分数完全相同。

## 验证范围

| 验证 | 已建立的范围 |
| --- | --- |
| 离线 fake SDK 与本地 HTTP loopback | Chat／Responses 请求、多模态和工具续轮、签名及状态校验、包含 phase 的 Responses 条目顺序、重试、预算、压缩、trace 和评测身份。 |
| 真实 Kaggle Astra／Responses／打包 `DoorSpin3Turns`，seeds 1–2 | 两个 case 各执行 10 个 Unity 动作，原始最终尝试 reward 均为 1；合计 20 次动作请求、14 次摘要请求，模型调用错误 0、SDK 重试 0。 |
| 该次真实运行的推理审计 | 返回 7 个推理条目，4 个在后续 6 次请求中回传，3 个在首次回传前被既有压缩清除；额外无法解释的缺失为 0。 |
| 真实运行的覆盖边界 | 响应没有 phase 字段；phase 保留和宿主工具中间续轮由受控 loopback 覆盖。真实运行的每个动作均直接 `submit_action`。 |
| 真实 Kaggle Opus 5.5／Chat Completions／`tool_choice: auto`／`DoorSpin3Turns`，run 5800062 | Seed 1 完成 10 个环境轮次，原始最终尝试分数为 1.0，最终成功。Seed 2 完成 4 个环境轮次后，因下一请求估算费用 $0.647248 超出剩余 Kaggle 配额而报 APIError 并停止。原先强制工具选择导致的 HTTP 400 已解决；这是因配额中断的双 seed 部分验证，不是任务代码错误。原始输出审计确认全部 14 次已接受的原生 `submit_action` 与环境动作一致；另有两次混合批次被既有机制拒绝并纠正。 |
| 未建立的结论 | 未做原生 Responses 直接 A/B，不宣称性能／效果零损失、全部供应商签名校验通过，也不把上方配置示例等同于全面真实调用认证。 |

相对外层仓库根目录的本地证据位于
`local_save/task_validation/SimpleAgent/litellm_door_20261102/`：
[执行报告](../../local_save/task_validation/SimpleAgent/litellm_door_20261102/execution_report.json)、
[推理续传审计](../../local_save/task_validation/SimpleAgent/litellm_door_20261102/readonly_reasoning_continuation_audit.json)
和[端到端审计](../../local_save/task_validation/SimpleAgent/litellm_door_20261102/readonly_e2e_audit.json)。
这些报告及原始轨迹、图像、日志是已忽略的 local-only 证据，fresh clone 中不可用；
它们不会把逻辑 SDK trace 变成上游报文抓包。

## 常见问题

| 现象 | 检查方法 |
| --- | --- |
| 找不到 LiteLLM／MCP／tenacity | 用实际运行评测的同一 Python 安装固定的 `.[litellm]` extra，再执行 `python -m pip check`。 |
| 函数工具 HTTP 400 | 检查模型和 API 路由是否支持函数工具。仅 required tool choice 不受支持时，手动 YAML 在 `simple_agent` 下设置 `tool_choice: auto`，新 Kaggle 模板使用 `AGENTARK_KAGGLE_TOOL_CHOICE=auto`。已知 LiteLLM Kaggle Opus 5.5 别名默认使用 auto；其他路由需显式选择。不要启用全局丢参数。 |
| 模型错误或 model-not-found | 检查内层供应商前缀、顶层 wire model、API 模式和端点。Kaggle Astra 内层需要 `openai/openai/gpt-6-astra`，直接 OpenAI 是 `openai/gpt-6-astra`。 |
| `openai/responses/` 配置错误 | 选择 `api: responses`；新配置使用普通 `openai/` 路由。Responses 目前拒绝其他 SDK 供应商前缀。 |
| temperature／thinking 不受支持 | Astra 使用 `temperature: null`，移除原生 `thinking`，只在支持时添加 `reasoning_effort`。 |
| 凭据或端点与预期不同 | 显式设置 `api_key_env`，查看逻辑 trace 端点。省略 `base_url` 表示 SDK 供应商默认地址；已指定但未设置的 key 会报错。 |
| 全局参数修改／fallback 报错 | 在调用进程中关闭 SDK 全局丢参数、自动修改、fallback 和重试设置，在 AgentArk 配置里显式选择条件。 |
| 摘要之后看不到推理 | 检查压缩事件和归档原生状态；成功压缩会按既有策略清除全部工作原生状态。 |
| 已有结果没有跳过 | 对照 backend／API／模型／harness 指纹，为目标条件使用独立输出路径；仍可回放已保存动作。 |
