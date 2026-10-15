# Common action explanations

The Python wrapper selects the common response recipe with
`prompt.explanation_mode`. Omission and `legacy_think` preserve the existing
common templates exactly. `brief_rationale` asks for one or two ordinary
sentences describing the intended action and its purpose, followed by the
existing action format. The explanation has no wrapper tag and is not a request
for private model reasoning.

An evaluation can opt in without editing a Unity task or the immutable Mods
baseline:

```yaml
env_cfg:
  env_config_overrides:
    prompt:
      explanation_mode: brief_rationale
```

A task can also set the following in its own `cfg/task_config.yaml` (or prefab
task configuration). This Python-only field works with older baseline files
that do not declare it:

```yaml
prompt:
  explanation_mode: brief_rationale
```

Evaluation overrides take precedence over task configuration. Each reset still
starts from `config.yaml.bak`; one task's selected mode does not become another
task's baseline. Unknown values fail configuration validation rather than
silently changing the reply protocol.

For Func tasks, the ordinary response is a short explanation followed by
`<tool_call>{"name":"...","arguments":{...}}</tool_call>`. The established
action extraction keeps only that call. Exact Code tasks use an explanation
followed by `<code>...</code>` on a new line; multiline source uses code tags on
separate lines. The wrapper passes only the source to Unity.
The missing/legacy action modes keep their existing MonoBehaviour contract.

With SimpleAgent's native tools, the common prompt defers to the harness:
assistant-message `content` holds the explanation, and `submit_action` carries
the action. The latter's argument also named `content` contains **only** the
task action payload. The Codex SDK transport maps ordinary assistant content to
its `text` field. No additional explanation tag is introduced. This is prompt
guidance, not a new requirement that providers return nonempty content.

Only wrapper-owned common templates are selected. User task descriptions,
payload strings and conversation history are never searched and rewritten.
This option does not alter APIAgent's native reasoning handling or existing
archives, nor guarantee that a provider will accept all task prompts.

Tool selection and error handling are unchanged. Under SimpleAgent
`tool_choice: auto`, a text-only completion can still reach the environment's
legacy action parser. In Func mode, an invalid action is reported in the next
observation's `step_msg` and consumes a step. `done_on_script_error: false`
normally allows the next decision to correct it; `true` ends the attempt.
Provider errors and malformed native submissions are separate failure paths.
