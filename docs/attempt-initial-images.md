# Additional images at attempt start

`ArkEnv` can attach task-authored images to the first actionable observation of
each attempt, before its first action. Use this for puzzle inputs, product
photos, screenshots, diagrams, or other images whose meaning the task prompt
explains. The generic label is `Additional image N` (`附加图像N` with Chinese
image captions). These images are separate from Unity camera observations.

## Configuration and files

Add this fragment to the task's `task_config.yaml`:

```yaml
env_wrapper_cfg:
  attempt_initial_images:
    - all_tasks/YourTask/images/additional.png
```

The field is an ordered `list[str]`. An absent field or `[]` disables the
feature and performs no image-file access. It belongs to the Python wrapper,
outside Unity `task_params`. The existing valid `config.yaml.bak` does not need
this key added: task configuration overlays already merge `env_wrapper_cfg`.
Keep that immutable baseline intact so the next task receives its own settings.
Putting images in the baseline or caller `env_config_overrides` intentionally
applies those settings more broadly.

Relative paths resolve from the **effective Mods root**, including a copied
sandbox Mods root, rather than from the process working directory or the task
folder. Include the files at those relative paths in the runtime copy. A relative
entry requires an absolute effective Mods root. Operator-provided absolute paths
are also accepted; keep machine-specific paths out of committed task configs.
See the [configuration fragment](../config/ark_env/attempt_initial_images.example.yaml).

A path may reference an existing string in the current effective configuration
using `${section.key}` (additional dictionary-key levels are allowed). For
example, `all_tasks/YourTask/References/${task_params.reference_filename}` follows
an evaluation override of `task_params.reference_filename` automatically. The
reference is read afresh when each attempt's images are loaded. Missing keys,
non-string values, invalid dotted-key syntax, and values containing further
configuration references raise `ValueError`; references do not recurse or
evaluate expressions. Only dotted placeholders have this meaning. Plain paths,
`~`, `$NAME`, and `${NAME}` retain their existing literal-path behavior; this
feature does not expand environment variables. Source paths and configuration
values are not added to observation metadata.
Task-development tools can reuse
`agent_ark.ark_env.attempt_initial_images.resolve_attempt_initial_image_paths(env_cfg)`
to validate and resolve these strings without filesystem access or configuration
mutation; relative paths in its result still use the effective Mods root.

Supported files are static PNG, JPEG, and WEBP. Limits are 16 images, 16 MiB per
file, 64 MiB combined source bytes, 32 million pixels per image, and 128 million
combined pixels. Invalid configuration, unreadable files, unsupported or corrupt
images, animation, and exceeded limits raise `ValueError`; entries are not
silently skipped. File-specific diagnostics identify the zero-based config-list
entry. EXIF orientation is applied, with no resizing or cropping; orientation
may swap width and height. Modes unsupported by PNG transport, such as CMYK
JPEG, are converted to RGB/RGBA; PNG-compatible modes and transparency remain.

## Observation and message contract

On each attempt, images are loaded afresh and copied for each agent. A skipped
or absent reset observation defers attachment until that agent's first non-skipped,
nonterminal observation. Hidden initial-observation warmup does not consume the
attachment. Later observations do not acquire a fresh copy of the field.

The optional raw observation field is:

```python
"attempt_initial_images": [
    {"index": 1, "image": copied_pillow_image, "sha256": source_file_sha256}
]
```

Indices follow config order and start at one. SHA-256 covers the original file
bytes, before orientation and PNG transport encoding. Raw observations contain
pixels and hashes; message parts contain pixels and labels. Neither includes
source paths. The task prompt should explain each image's role using its visible
index.

Message rendering emits the neutral label followed by a native image part in
the selected `openai` or `raw_base64` format. `messages.include_images: false`
disables these message image parts; it does not remove raw images. Camera frame
selection limits and omission of unchanged workspace frames do not suppress
these attachments. Raw-observation API and Codex paths also support the field.
The HTTP observation codec transports this optional raw field with the existing
typed PNG encoding and restores Pillow images on the client, including when
there are no camera frames. Other protocol fields keep their existing format.

Each attachment is one logical observation event. A full-history request may
include that event again while it remains in retained history. With append-only
delta messages, it is sent in the attempt-start delta and retained by the
stateful conversation. This is not a guarantee of one provider wire upload or
one token charge: provider and agent context handling still apply.

## Records and replay

Trajectory JSONL recursively stores the raw image as a typed PNG payload and
preserves its index and source hash. Loading restores the Pillow image without
needing the source file. `eval.trajectory_save.include_images: false` replaces
typed image payloads with an omission marker (decoded as `None`) and preserves
index/hash metadata. This setting does not redact arbitrary legacy string or
data-URL fields elsewhere in a record.

Trajectory `history_snapshot` follows the existing history retention settings.
Trimming away the first transition also removes its additional images. For a
self-contained prefix replay, enable attempt history and retain the entire
attempt (`max_history_steps_per_attempt: 0`).

The evaluator emits its first `agent_request` hook event before the first action
and history trimming. Serialized raw observations expose separate
`attempt_initial_images` entries containing `index`, `sha256`, and a PNG data
`url`, even with messages disabled, independently of the camera-image cap.
Retaining those hook events preserves the initial pixels when history is later
trimmed. Normal result JSONL request summaries and sanitized raw request traces
retain labels/counts, rather than full image bytes. Hook events are not a new
durable log unless a configured consumer saves them.

Consumers that display only `vis` or the serialized camera `images` array must
add support for this separate field to display it. The existing human review
camera-frame extraction is unchanged. Task107 uses this option with two workspace
cameras and append-only delta messages; its task-local documentation describes
the companion files and text-only configuration.

See the [validation report](attempt-initial-images-validation.md) for automated
coverage and actual packaged Player comparisons with existing tasks.
