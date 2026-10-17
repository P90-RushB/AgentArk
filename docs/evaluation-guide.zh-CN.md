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
    thinking:
      type: enabled
    reasoning_effort: high
    max_completion_tokens: 30000
    temperature: 0.0
```

根据端点替换 `provider`、`model`、`base_url` 和 `api_key_env`。也可以在本地私有
配置中直接设置 `api_key`。如果模型或代理提示 `temperature` 已弃用或不受支持，
请设置 `temperature: null`，让请求省略该参数。
HTTP API 模型的 `max_completion_tokens` 默认为 `30000`，限制推理 token 与可见
输出 token 的总量；设置为 `null` 时省略该参数。

HTTP 模型可以使用统一的 `thinking.type`（`enabled` 或 `disabled`）和
`reasoning_effort` 配置。AgentArk 会按 provider 转换请求字段：

| provider | 实际请求字段 |
| --- | --- |
| `volcengine` | `thinking.type` 和顶层 `reasoning_effort` |
| `openrouter` | `reasoning.enabled` 和 `reasoning.effort` |
| `dashscope` / `qwen` | `enable_thinking`；不映射 `reasoning_effort` |
| `openai` / `generic` | 顶层 `reasoning_effort`；不映射 `thinking` |

如果 provider 无法无歧义地映射配置，AgentArk 会在发请求前报错。省略这些配置时，
保留原有 provider 默认行为。

#### 火山方舟在线推理

火山方舟在线推理兼容 OpenAI SDK。使用 `provider: volcengine`，让 AgentArk 发送
Chat Completions 请求，并按火山方舟格式传递深度思考配置：

```yaml
models:
  - name: doubao-seed-2-1-pro-260628
    provider: volcengine
    model: doubao-seed-2-1-pro-260628
    base_url: https://ark.cn-beijing.volces.com/api/v3
    api_key_env: ARK_API_KEY
    thinking:
      type: enabled
    reasoning_effort: high
    max_completion_tokens: 30000
    timeout_s: 180
    temperature: 1.0
```

运行前设置 API Key：

```bash
export ARK_API_KEY="your-api-key"
```

这里的版本化模型 ID 和 `/api/v3` 地址属于按量计费的在线推理。不要与 Coding Plan
的 `/api/coding/v3` 地址及 `doubao-seed-2.0-pro` 模型名混用。AgentArk 当前通过
Chat Completions API 调用模型；`thinking.type: enabled` 显式开启深度思考，
`reasoning_effort: high` 设置推理力度，`max_completion_tokens: 30000` 限制推理
token 与可见输出 token 的总量，`temperature: 1.0` 显式设置采样温度。
视觉任务使用 `high` 时可能超过示例的 180 秒超时；如果
可以接受更长延迟，请相应提高 `timeout_s`。

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

### 可选 SimpleAgent：上下文压缩与宿主工具

`APIAgent`、`SimpleAgent`、`CodexAgent` 是独立的 agent。新增的 `agent_type`
选择 agent，`provider` 继续选择模型后端。不填写 `agent_type` 时沿用原来的行为：
`provider: codex` 使用 CodexAgent，其他 provider 使用基础 APIAgent。
已有配置、基础 agent 和 Codex 的上下文管理保持原样。

使用 `config/ark_env/simple_agent.example.yaml`：

```bash
python -m agent_ark.ark_eval.run_api_agent --config config/ark_env/simple_agent.example.yaml
```

模型项显式填写 `agent_type: simple`，并在 `simple_agent.context` 中设置预算。
评测入口只为该 agent 的独立环境配置启用 `append_only: true`、`return_mode: delta`。
SimpleAgent 接收新增图文/动作反馈，自行保留近期完整轮次，超预算时让模型摘要旧轮次。
普通无状态 HTTP 请求仍发送整理后的完整工作上下文。摘要是模型记忆，不是环境事实；
原始有界归档与工作上下文分开。当前观察、任务指令、完整代码和必要反馈不会被静默截断。
图像 token 数量使用可配置估算，并同时检查编码字节、图像数量和原始图像字节上限。
工作上下文和摘要输入的编码字节上限均默认为 16000000，包含 base64 图像，
为 API 请求封装保留余量。

通用默认预算为 262144（256 * 1024）个估算上下文 token，其中包含 20480 个动作输出预留；
工作上下文最多 256 张图，保护最近两个输入回合（含当前输入）。摘要请求的输入上限为
48000 个估算 token，另有 32768 个生成 token 预算。只有响应的最终 `content` 正文
会成为摘要记忆；单独返回的可读推理保留在原始 trace 中，并继续计入服务端输出用量。
正文为空、达到输出上限、摘要过长或未缩减原文时，下一次摘要请求会收到简短的失败原因，
要求重新给出有效正文。重试共用已有的每决策 `max_summary_calls` 上限（默认 4 次）、
总调用预算和决策时限；成功压缩也计入同一额度。无效摘要不会替换工作历史。
不另增摘要留存长度参数。
这些是可覆盖的上限，不代表固定压缩比例，也不保证所选模型的原生窗口足够大。

SimpleAgent 的 HTTP 后端在 Chat Completions 历史中保留已支持的 OpenRouter 推理字段，
以及火山的原名 `reasoning_content` / `encrypted_content` 字段组合。
回复正文与原生推理分开保存；按响应字段选择支持的回传形式，无需新增配置项。
不保证兼容采用其他协议的端点。
回传、压缩和记录行为详见[原生推理历史说明](simple-agent-native-reasoning.md)。

SimpleAgent 默认使用 HTTP API。使用已有 ChatGPT 订阅作为模型调用后端时，
可为任意任务配置以下模型条目：

```yaml
models:
  - name: simple-codex
    agent_type: simple
    provider: codex
    model: gpt-6.1-sol
    reasoning_effort: medium
```

未填写的 `simple_agent` 设置继承通用默认值。环境、case 和工具工作目录的设置
参照 `config/ark_env/simple_agent.example.yaml`。
需要可选的 `openai-codex` 依赖和已有 ChatGPT 文件登录凭据；SDK 自带 CLI 不支持
目标模型时，通过 `AGENTARK_CODEX_BIN` 指定兼容版本。
上下文、摘要和宿主工具仍由 SimpleAgent 管理。每次模型请求新建隔离 SDK thread，
用严格 JSON 封装把工具选择交回 SimpleRuntime；这不是 Codex 原生 function tool。
SDK 推理、缓存 token 细节和实际用量留在审计记录中。SDK 没有硬输出 token 上限接口，
因此动作/摘要预算是提示要求，返回后仍检查运行时文本、字节和摘要 token 估算界限；
trace 会明确标记该限制。一次 SDK turn 不保证只有一次底层 HTTP 请求。
trace 中的 `simple_agent.sdk_max_retries` 描述 HTTP 后端重试，不控制 Codex CLI 的网络恢复。
Codex 错误通知会留在审计中，适配器继续等待同一 turn 的终态，受八条通知和请求时限约束；
适配器不会为此额外重发一个外层模型请求。

`simple_agent.tools.enabled: true` 启用四个宿主工具：`list_files`、按行 `read_file`、
实际图像 `view_image`、`write_note`。同时设置 `workspace_parent` 为已有运行可写目录。
每条 case/agent 有独立私有目录；只读记录来自已经提供给模型的观察及它自己的完成输出，
私有笔记可写。同 case 的自动 attempt reset 保留记忆，新的 case 清空；close 清理私有工作目录。
工具不提供任意主机目录、任务源码、隐藏 seed/config、shell 或 Unity 控制接口。
工具循环完成后才提交一个原有格式的环境 action，工具/摘要调用不消耗环境步。
存在宿主工具注册表时，SimpleAgent 还声明保留的 `submit_action` 控制函数，参数为
`format`（`tool_call`、`params`、`code` 或 `raw`）和字符串 `content`。
模型提交不带 XML 包装的动作正文，适配器将单独的一次提交转换为环境原有 action；
代码源码保留原样，`params` 沿用原有提取器的首尾空白处理。正文不得包含动作包装标签
（代码中的这类字符串可使用 C# Unicode 转义）。提交本身不执行宿主工具，也不推进游戏。
这将任务 XML 输出约定与模型原生工具调用分开。后续上下文保留原生提交调用和明确
标注的“已选定、尚未执行”确认，真实模型文本也可能包含普通文本动作；环境单独回显的
动作会去重；出现原生提交时，动作选择以该提交为准。实际结果由下一条环境观察提供。
摘要请求将旧工具链转为带来源标注的文本并保留原图，不提供可执行工具。
Trace 分别记录模型响应和动作转换。
混合宿主/动作提交、格式错误会明确报错；原生工具参数含 JSON 转义的 UTF-8 大小最多
128 KiB，解码后的动作还受配置的输出预算约束。原生动作轮默认设置
`simple_agent.tool_choice: required`，模型自行选择宿主工具或最终动作提交，缺少原生调用
会明确报错。端点不支持该选项时可显式设置 `auto`，此时普通 assistant 文本保留为兼容
路径；没有隐式降级或重试。摘要和不带原生工具的请求不发送该参数，关闭工具的 agent
继续使用普通文本路径。较严格工具选择的端点支持差异参见
[OpenRouter 文档](https://openrouter.ai/blog/tutorials/tool-calling/)。
Python 调用者也可传入 `SimpleAgent(..., tool_setup=callback)`，在每个 case 的新
`ToolRegistry` 上用 `register(ToolSpec(...), function)` 注册可信通用工具。
该回调不从 YAML 加载，注册表不会在 case 间共享。ToolRegistry 在调用前验证有界的
JSON schema 子集（对象、数组、字符串、整数、布尔值）；默认文件记录上限为 512 条，
发布字节上限为 512 MiB，更长任务应显式调整 `tools.limits`。原始存档的总字节、
图片字节上限各为 512 MiB，累计文本上限为 32 Mi 个字符；摘要不会回收这些累计额度。
容量耗尽时停止并保留失败证据。

上下文、原始归档、模型调用、工具轮次、结果文本/图片、记录和笔记都有明确上限。
预算不足或摘要失败会报错，模型后端失败不会自动换接口参数重试；显式 `max_retries`
控制 HTTP 客户端重试，新 agent 默认 0。动作输出上限取 `max_completion_tokens`
和 `output_reserve_tokens` 的较小值；摘要另用独立的 `summary_output_tokens`
预算（默认 32,768），不再受动作 `max_completion_tokens` 上限限制。
这不是无限历史或磁盘会话恢复系统，也不是对恶意本地进程的操作系统沙箱。
摘要恢复额度耗尽后，评测器记录当前 case 错误和请求、用量 trace。通用示例采用
`eval.stop_on_error: false`，继续其它 case，不会自动重跑失败的 case。
凭证、参数、拒绝、取消、超时及存档容量错误不会触发自动摘要生成重试。
摘要失败不消耗环境 step，也不会提交占位动作。
思考 token 同样消耗每次请求的 completion 上限。摘要继续继承动作的 reasoning/thinking
参数，Codex SDK 后端同样如此。只有 assistant 正文会进入摘要记忆，推理字段只保留为审计数据。
分开字段不会为正文预留输出 token，因此独立摘要预算需同时容纳推理和正文。
首次摘要 prompt 即声明总 token 预算（包含思考）和正文字符/UTF-8 字节限制，只要求精炼笔记。
`length` 结束的响应会报错，不提交残缺动作或接受残缺笔记。
应针对所选强度分别设置动作和摘要预算。Qwen 示例选择低思考强度；
当前预算仍需针对所选模型验证，先前有限 Player 联测不等于已验证这套新默认值。
直接构造实例仍显式启用文件工具并提供可写 `workspace_parent`；通用示例已启用它们。

结果步骤的 `raw_response_by_agent` 包含 `model_requests`、`agent_events`、
`agent_usage` 与 `simple_agent` 配置元数据，区分真实动作模型请求、摘要和宿主工具续轮。
图像请求在文本 trace 中省略原件；完整模型可见环境图像由开启图片的 trajectory 记录保存。
实际请求同时记录图像 SHA-256 和字节数，失败结果保留已完成调用与已知用量。
环境回放继续使用最终动作；它不会重新调用模型或恢复工具/笔记状态。
RL rollout 接入保持独立，新增评测 agent 不自动修改 Swift 或 VERL 的训练轨迹。

实现位于 `src/agent_ark/agent/simple_agent/`；基础 API 与 Codex 保留各自原模块。
未来 agent 可以通过 `agent/agent_registry.py` 的 `AgentType` 和 `register_agent_type`
注册自己的模块、配置关键字和消息模式，YAML 不接受任意 Python 模块导入。

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
    # GPT-6 Astra 可选：low、medium、high、xhigh、max。
    # 省略时使用 Codex SDK/模型默认值。
    reasoning_effort: medium
    thread_mode: per_agent
    codex_context_mode: lean
    # 如个人 Codex 还配置了其他 MCP，请把 server name 追加到这里。
    codex_disabled_mcp_servers: [node_repl, unityMCP]
```

AgentArk 会把 OpenAI 风格的评测消息转换成 Codex 文本和图像输入。Data URI 图像观测
会直接作为 Codex `ImageInput` 传入。需要更低成本或更快响应时，可以用
`reasoning_effort` 覆盖 Codex turn 的推理深度。

AgentArk 的 `provider: codex` 默认使用 `codex_context_mode: lean`。这是当前 runner
创建 SDK thread 时传入的局部配置，不会修改 `~/.codex/config.toml`，也不会影响 Codex
应用、CLI、IDE 插件或其他 SDK session。lean thread 使用空的 Codex base/developer
instructions、`deny_all` approval、ephemeral thread 和空临时 cwd，并关闭 web/image/shell
工具、apps、plugins、multi-agent 等 feature。当前环境常见的 `node_repl` 与
`unityMCP` 也会被显式关闭。

Codex 配置中的 MCP map 会按名称合并，无法用一个空 map 覆盖所有个人 MCP。因此，若
个人配置里还有其他 MCP server，必须把其名称加到 `codex_disabled_mcp_servers`。需要做
包含标准 Codex 上下文的对照实验时，可显式设置 `codex_context_mode: default`；这同样
只影响该次 AgentArk SDK 评测。结果 JSONL 会记录实际的 context mode 和禁用的 MCP
名称。当前支持 GPT-6 的安装应使用 `openai-codex>=0.147.0`。如果 SDK 自带 CLI
尚不支持所选模型，请设置 `AGENTARK_CODEX_BIN` 指向当前 Codex App/CLI 可执行文件。

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

这里的 delta 由 AgentArk 环境的 `messages.return_mode` 决定，不是 `CodexAgent` 无条件
产生的。启用 messages 时，`return_mode` 的代码默认值是 `delta`；reset 仍返回一次完整
base message，之后 step/attempt 更新才返回新增消息。为保证批量评测可复现，建议仍像
上例一样显式写出 `append_only: true`、`return_mode: delta` 和
`thread_mode: per_agent`。

运行：

```bash
python -m agent_ark.ark_eval.run_api_agent \
  --config config/ark_env/eval_seed1.example.yaml
```

`task_name` 必须匹配 `Mods/all_tasks` 下的目录。打包运行时包含 51 个首发任务，
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

默认的 lean player 模式会自动增加这些保护：

- 动作提示要求 Codex 只使用玩家可见消息和图像。
- Codex thread 从一个新的空临时 cwd 启动，而不是从源码仓库启动；agent 关闭时会删除
  该目录。
- Codex 的默认 base/developer instructions 置空，并在线程级关闭工具、apps、plugins、
  multi-agent 和列出的 MCP server。

终止观测后，`run_api_agent` 会要求同一 thread 输出一个结构化
`<player_feedback>` JSON 报告。报告区分具体任务缺陷与未解出任务、难度、玩家错误或
合理探索需求等非缺陷，结果记录将其保存在 `player_feedback` 下。请启用
`trajectory_save` 并设置 `condition: all`。同时按最终打包任务实际交给玩家的消息表面设置
`eval.player_observation_modality: multimodal|text`：multimodal 必须设置
`include_images: true`，text 可以设置 `include_images: false`。这样即使玩家没有成功，
报告的问题也能以对应模态回放。如果最终报告缺失、失败或不符合 JSON schema，评测结果会
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
和对应模态的回放交给 task reviewer。reviewer 独立对照已批准的信息揭示设计、
task description、实际 `max_attempts`、seed/history 语义和打包回放，再把候选问题
分类为 `confirmed defect`、`non-defect` 或 `inconclusive`。只有确认的缺陷才交给
builder；结论不明确时可追加一次定向运行；有意探索仍记录为非缺陷。

串行和并行 runner 都会拒绝缺少以下设置的玩家反馈配置：
`trajectory_save.enabled: true`、`condition: all` 和输出路径。默认
`player_observation_modality` 为 `multimodal`，这种模式还要求 `include_images: true`；
显式 `text` 模式允许不保存图像，但验证时必须确认文本 observation/action 非空且玩家请求
中没有 image block。
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

显式的 `action_mode: code` 会启用受限 Code Runtime。任务 prompt 会提供 capability
manifest，模型输出一个普通托管 `AgentAction` 源文件：

```xml
<code>
public sealed class AgentAction : CodeAction
{
    public override void Start()
    {
        int target = Context.Read("read_target").AsInt32();
        Context.Call("apply_target", CodeValue.FromInt32(target));
        Context.CompleteStep();
        Context.StopSelf();
    }
}
</code>
```

上面的 member 名称仅作示例；生成代码只能使用当前任务 manifest 中声明的 member，不能
直接访问 Unity、场景层级、`ActRouter`、反射、文件、网络或线程。受限 tool call 足够时
使用 Func Mode；需要分支、循环、持久程序状态，或跨多个由 host 驱动的
`Tick`/`FixedTick` 回调调用可信 capability 时使用 Code Mode。

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
