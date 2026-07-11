# Runtime Sandbox Migration Draft

English | [简体中文](runtime-sandbox-migration.zh-CN.md)

## Goal

Move from a shared Unity runtime directory to per-worker runtime sandboxes so that:

- each env instance has its own writable runtime state;
- parallel eval and RL workers do not race on shared Mods/config/bundle files;
- task assets can still be shared read-only to avoid duplicating thousands of mods.

This draft is intentionally scoped to the current local Unity execution model used by ArkSubEnv and ArkEnv.

## Platform Scope

The migration must support two runtime families:

- RL training: Linux only;
- evaluation: both Linux and Windows.

That means the sandbox architecture can be shared conceptually, but the concrete preparation and link implementation must be platform-aware.

Non-goal:

- one runtime pool shared across Linux and Windows.

Each platform should prepare and consume its own runtime template, pool root, and task-store mapping.

## Phase 1 Decisions

The current implementation direction is now fixed as follows:

1. Configuration lives under `env_cfg.runtime_sandbox`.
2. The current runtime sandbox uses full per-worker runtime copies and exposes a shared read-only `Mods/all_tasks` store inside each worker sandbox.
3. `worker_index` is the sandbox selector across demo, evaluation, and RL paths.
4. The low-level env constructors can resolve sandboxed `env_path` and `mod_path` as a safe fallback, but launcher-side ensure remains the preferred orchestration point.
5. The current template fingerprint strategy is a recursive hash over the runtime tree using relative path, file size, and file modification time metadata.
6. `runtime_sandbox.shared_task_store_path` is the host-side source of truth for the shared task repository.
7. Under `link_mode=auto`, Linux should realize `Mods/all_tasks` as a directory symlink and Windows should prefer a directory junction, with `copy` only as the final fallback.

These decisions define the current shared-task-store implementation.

Important current assumption:

- Task folders live in one shared task store and are exposed inside each worker as `Mods/all_tasks/<task>`.
- `mod_path` still means the writable worker `Mods/` root, not the shared task-store root.
- The main goal remains the same: isolate writable runtime state while keeping task selection deterministic across concurrent envs.

## Problem Summary

The current layout uses one shared runtime directory for all local env instances.

Observed failure mode:

- multiple Unity instances load the same shared Mods root;
- Unity-side ModManager replaces the active bundle in Mods root at runtime;
- other Unity instances may still have that bundle open;
- Addressables then reports CRC mismatch, invalid path, or archive modified while opened;
- Python may also rewrite shared config.yaml and config.json in the same Mods root.

This is a sandboxing problem, not a port allocation problem.

## Design Principles

1. Writable runtime state must be private per worker.
2. Large task assets should be shared read-only when possible.
3. Worker-to-runtime mapping should be stable across a run.
4. Runtime preparation should be explicit and scriptable.
5. The design must work for local demos, evaluation, and future RL training.

## Recommended Target Architecture

Split the current single `mod_path` concept into two roles:

- active runtime mods root: writable, private to one worker;
- task store: read-only shared repository of all task directories.

### Target Layout

Example host layout:

```text
unity_runtime_template/
  AgentArk.x86_64
  AgentArk_Data/
    Resources/
      Mods/
        config.yaml
        config.json
        defaultlocalgroup_assets_all_....bundle

unity_task_store/
  snake/
    ...
  marble/
    ...
  ...

unity_runtime_pool/
  worker_000/
    AgentArk.x86_64
    AgentArk_Data/
      Resources/
        Mods/
          config.yaml
          config.json
          defaultlocalgroup_assets_all_....bundle
          all_tasks -> /abs/path/to/unity_task_store
  worker_001/
    ...
  worker_002/
    ...
```

For Windows evaluation, the equivalent layout should exist under a Windows-native root, for example:

```text
<AGENTARK_RUNTIME_TEMPLATE_ROOT>\
<AGENTARK_TASK_STORE_PATH>\
<AGENTARK_RUNTIME_POOL_ROOT>\worker_000\
```

The key rule is that the active worker sandbox should live on the native filesystem of the platform that is running Unity.

### What Stays Private Per Worker

- the runtime executable directory;
- the active Mods root;
- config.yaml and config.json under Mods root;
- the currently active shared bundle file under Mods root;
- any future task-generated runtime files;
- worker-specific logs if later redirected.

### What Can Be Shared Read-Only

- all task source folders;
- task metadata files;
- task-specific source bundles copied into active Mods root;
- other static assets that are never modified at runtime.

## all_tasks Contract

The current low-risk Unity-side contract is:

- move all task folders out of Mods root into one shared `all_tasks` repository;
- expose that repository inside each worker runtime as `Mods/all_tasks` via a platform-appropriate link;
- keep Unity-side loading logic pointed at `Mods/all_tasks/<task>` as the source;
- continue copying the selected task's runtime bundle into the worker's private Mods root.

This preserves the existing requirement that Addressables load from a fixed Mods root, while removing shared mutation from that root.

### Link Strategy by Platform

The design should treat `Mods/all_tasks` as a logical link target, not require one specific OS primitive.

Linux:

- prefer directory symlink.

Windows:

- prefer directory junction or directory symlink;
- if symlink permissions are unavailable, fall back to junctions for directory links;
- only fall back to copying task directories if links are impossible in the target environment.

Implementation note:

- the pool preparation utility should own this decision so launchers do not care whether `all_tasks` was realized as a symlink, junction, or copied directory.

### Required Invariant

`all_tasks` must be strictly read-only at runtime.

If any task writes into its own task folder during execution, that task cannot safely live in the shared store without another layer of sandboxing.

## Preparation Strategy

Use a two-level preparation model.

### Primary Mechanism: Explicit Preparation Script

Provide a dedicated script that prepares or refreshes a runtime pool.

Suggested inputs:

- runtime template path;
- runtime pool root;
- shared task store path;
- pool size;
- target platform;
- optional force refresh flag;
- optional manifest/version stamp.

Suggested responsibilities:

- create `worker_000 ... worker_N` sandboxes;
- copy the runtime template into each worker sandbox;
- remove bundled task directories from per-worker Mods if they are no longer needed there;
- create `Mods/all_tasks` using the correct platform-specific link mode;
- validate that the sandbox has the expected private files and shared links.

The preparation implementation should be callable from both Linux and Windows environments. It should not assume WSL-only paths, `/mnt/...`, or Linux-only symlink semantics.

### Secondary Mechanism: Launcher-Side Ensure

Demo/eval/train entrypoints may call a lightweight ensure step before launching workers.

That ensure step should:

- verify the requested worker count exists;
- verify a version stamp matches the template;
- fail or rebuild before workers start;
- never allow each worker process to race while building its own sandbox.

This keeps ad hoc local runs convenient without making runtime construction part of reset-time logic.

For Windows evaluation, the same ensure contract should apply, but it should validate a Windows-native pool root and Windows runtime template rather than reusing Linux assumptions.

## Worker Lifecycle

### Local Demo / Eval

- each parallel eval or local worker chooses a unique `worker_index`;
- each `worker_index` maps to one sandbox path;
- the sandbox is reused for the full process lifetime;
- env reset only mutates that worker's private runtime.

This worker mapping must work the same way on Linux and Windows, so higher-level eval code should resolve a sandbox path through one platform-neutral helper instead of assembling paths inline.

### RL Training

- each long-lived rollout worker receives one sandbox assignment;
- the sandbox survives across episodes and attempts;
- if a worker corrupts its runtime, only that worker sandbox is rebuilt;
- pool sizing should match peak concurrent Unity instances, not episode count.

## Configuration Direction

The current `mod_path` field overloads two different concepts. The migration should gradually split configuration into:

- `runtime_path`: path to the worker-specific Unity executable or sandbox root;
- `active_mod_path`: path to the worker-specific writable Mods root;
- `task_store_path`: shared read-only repository of all tasks;
- `runtime_pool_root`: parent directory that holds all worker sandboxes;
- `runtime_platform`: `linux` or `windows`;
- `worker_index`: sandbox selector.

Short-term compatibility can keep `mod_path` as an alias for `active_mod_path` while Python and Unity are migrated.

For evaluation, configuration examples should eventually ship in both Linux-style and Windows-style path forms, rather than assuming one OS path convention.

## Suggested Migration Phases

### Phase 1: Full Sandbox Copy With Minimal Logic Change

Objective:

- eliminate shared writable runtime state first.

Approach:

- copy one full runtime directory per worker;
- keep current task layout inside each sandbox;
- switch launchers to choose a sandbox per worker.

Explicitly for the current runtime:

- task folders live in the shared task store and are exposed inside each worker as `Mods/all_tasks/<task_name>/...`;
- Python-side case discovery now resolves tasks through `Mods/all_tasks` while keeping root config writes under `Mods`;
- Unity-side task loading still needs to read from `Mods/all_tasks/<task_name>` consistently for full end-to-end validation.

Cross-platform note:

- do this first on Linux because training depends on it;
- keep the preparation interface identical on Windows so evaluation can later consume the same conceptual model with a Windows template.

Benefits:

- lowest implementation risk;
- fastest way to prove the CRC and bundle races are gone.

Cost:

- highest disk usage.

### Phase 2: Shared all_tasks Store

Objective:

- reduce storage cost without reintroducing bundle races.

Approach:

- move task directories to a single shared task store;
- expose that store as `Mods/all_tasks` in every sandbox;
- keep only writable active files in each private Mods root.

Benefits:

- much lower disk footprint when task count grows large;
- preserves existing Unity-side fixed Mods root idea.

Risk:

- requires a clean read-only contract for task folders.

### Phase 3: Config and Launcher Cleanup

Objective:

- make the new architecture explicit in code and config.

Approach:

- split `mod_path` into explicit runtime and task-store fields;
- centralize worker sandbox resolution;
- expose one shared pool preparation utility used by demo, eval, and RL launchers.

Benefits:

- less ambiguity in the Python codebase;
- easier long-term maintenance.

## Why This Should Solve the Current Bundle Error

The specific CRC mismatch and invalid path failures are caused by concurrent mutation of one shared bundle file in one shared Mods root.

With per-worker sandboxes:

- worker A modifies only `worker_A/Mods/...bundle`;
- worker B modifies only `worker_B/Mods/...bundle`;
- neither worker can invalidate the other worker's open archive handle.

That removes the observed root cause.

This does not guarantee that every future Unity failure disappears, but it should remove the current shared-bundle race.

## Path Placement Recommendation

Prefer storing the runtime pool on the Linux filesystem inside WSL rather than under `/mnt/c/...` when feasible.

Reasons:

- better small-file metadata performance;
- more predictable symlink behavior;
- fewer cross-filesystem quirks under heavy parallel file operations.

If the source runtime must remain on `/mnt/c/...`, the prepared worker sandboxes can still live on the Linux side.

For native Windows evaluation, use a Windows-native local path for the runtime pool rather than a WSL-mounted path.

Recommended rule:

- Linux Unity process uses a Linux-native pool root;
- Windows Unity process uses a Windows-native pool root.

Do not design the pool around WSL-specific path assumptions if evaluation must also run outside WSL.

## Cross-Platform Preparation Model

The same high-level pool model should be implemented on both platforms, but with different concrete templates.

Linux:

- template source example: `<AGENTARK_RUNTIME_TEMPLATE_ROOT>/...`;
- pool root example: `/workspace/runtime_pool/...` or another ext4-backed path;
- preferred link type: symlink.

Windows:

- template source example: `<AGENTARK_RUNTIME_TEMPLATE_ROOT>\...`;
- pool root example: `<AGENTARK_RUNTIME_POOL_ROOT>\...`;
- preferred link type: junction or directory symlink.

This implies two operational modes, not two different architectures.

## Launcher Compatibility Requirement

The runtime sandbox mechanism should be introduced below the level of demo/eval/RL business logic.

That means:

- local and parallel API evaluation should request a sandbox for `worker_index`;
- future RL workers should request a sandbox for `worker_index`;
- none of those entrypoints should directly care whether the sandbox was prepared on Linux or Windows.

A small resolver layer should map:

- runtime platform;
- runtime pool root;
- worker index;

to:

- executable path;
- active Mods root;
- task store mapping metadata.

## Phase 1 Concrete Interface Draft

This section narrows the runtime sandbox proposal into concrete contracts for the shared `all_tasks` implementation.

### Preparation Script Contract

Suggested entrypoint:

- `python -m agent_ark.tools.prepare_runtime_pool`

Suggested arguments:

- `--runtime-platform {linux,windows}`
- `--template-root <path>`
- `--shared-task-store-path <path>`
- `--pool-root <path>`
- `--pool-size <int>`
- `--force-refresh`
- `--clean-extra-workers`
- `--manifest-name runtime_pool_manifest.json`
- `--link-mode {auto,symlink,junction,copy}` for later Phase 2 use

Phase 1 behavior:

- validate the template root exists and contains a runnable Unity build;
- create worker directories `worker_000 ... worker_N` under the pool root;
- copy the full runtime template into each worker sandbox;
- write one manifest file at the pool root describing the prepared pool;
- optionally remove stale workers beyond the requested pool size.

Phase 1 outputs:

- one prepared runtime sandbox per worker index;
- one pool manifest file;
- non-zero exit code if preparation is incomplete or inconsistent.

Suggested manifest fields:

- `runtime_platform`
- `template_root`
- `pool_root`
- `pool_size`
- `template_fingerprint`
- `prepared_at`
- `worker_dirs`
- `layout_version`

The script should be idempotent: re-running it with the same template and pool size should either do nothing or refresh only what is outdated.

### Launcher-Side Ensure Contract

The launcher-facing API should be lighter than the preparation script.

Suggested behavior:

- read the manifest from the pool root;
- verify the requested worker index range exists;
- verify the template fingerprint still matches the expected template;
- fail fast with a clear error if the pool is missing or stale;
- optionally invoke the preparation script when `auto_prepare=true`.

Important constraint:

- this ensure step should run once in the launcher or coordinator process, not inside each worker thread or each env reset path.

### Resolver Contract

Suggested internal Python helper:

```python
resolve_worker_runtime(
    runtime_platform: str,
    pool_root: str,
    worker_index: int,
) -> dict
```

Suggested resolved fields:

- `worker_index`
- `worker_name`
- `runtime_root`
- `env_path`
- `active_mod_path`
- `pool_root`
- `runtime_platform`

Phase 1 resolution rules:

- `runtime_root = <pool_root>/worker_<NNN>`
- `env_path` is the Unity executable inside that worker sandbox;
- `active_mod_path` is the worker-specific `.../Resources/Mods` directory;
- resolution should not depend on task identity yet.

This resolver should become the only place that knows the worker sandbox layout.

## Minimum Config Draft

The smallest Phase 1 configuration surface should be enough for demos, evaluation, and RL workers.

Suggested fields:

```yaml
runtime_sandbox:
  enabled: true
  runtime_platform: linux
  template_root: /abs/path/to/runtime_template
  pool_root: /abs/path/to/runtime_pool
  auto_prepare: false
  pool_size: 8
```

Runtime use rule:

- when `runtime_sandbox.enabled=false`, current direct `env_path` + `mod_path` behavior remains unchanged;
- when `runtime_sandbox.enabled=true`, `worker_index` must resolve through the sandbox pool;
- resolved `env_path` and `active_mod_path` should override the raw configured paths at runtime.

For Windows evaluation, the same shape should be used with Windows-native paths, for example:

```yaml
runtime_sandbox:
  enabled: true
  runtime_platform: windows
  template_root: "${AGENTARK_RUNTIME_TEMPLATE_ROOT}"
  pool_root: "${AGENTARK_RUNTIME_POOL_ROOT}"
  auto_prepare: false
  pool_size: 8
```

## Phase 1 Directory Examples

### Linux Example

```text
/workspace/unity_runtime_template/
  AgentArk.x86_64
  AgentArk_Data/

/workspace/unity_runtime_pool/
  runtime_pool_manifest.json
  worker_000/
    AgentArk.x86_64
    AgentArk_Data/
      Resources/
        Mods/
  worker_001/
  ...
```

Resolved values for `worker_index=3`:

- `runtime_root=/workspace/unity_runtime_pool/worker_003`
- `env_path=/workspace/unity_runtime_pool/worker_003/AgentArk.x86_64`
- `active_mod_path=/workspace/unity_runtime_pool/worker_003/AgentArk_Data/Resources/Mods`

### Windows Example

```text
<AGENTARK_RUNTIME_TEMPLATE_ROOT>\
  AgentArk.exe
  AgentArk_Data\

<AGENTARK_RUNTIME_POOL_ROOT>\
  runtime_pool_manifest.json
  worker_000\
    AgentArk.exe
    AgentArk_Data\
      Resources\
        Mods\
  worker_001\
  ...
```

Resolved values for `worker_index=3`:

- `runtime_root=<AGENTARK_RUNTIME_POOL_ROOT>\worker_003`
- `env_path=<AGENTARK_RUNTIME_POOL_ROOT>\worker_003\AgentArk.exe`
- `active_mod_path=<AGENTARK_RUNTIME_POOL_ROOT>\worker_003\AgentArk_Data\Resources\Mods`

## Entry Point Integration Draft

The current runtime resolution should be inserted below entrypoint business logic.

Suggested integration order:

1. launcher reads config;
2. launcher decides target worker indices;
3. launcher ensures the pool exists for the required platform;
4. launcher resolves sandbox paths for each worker;
5. launcher injects resolved `env_path` and `active_mod_path` into env config before ArkSubEnv or ArkEnv is constructed.

This lets local eval, parallel eval, and RL launchers all share the same sandbox mechanism.

## Recommended First Code Slice

If implementation starts from this draft, the most constrained first slice should be:

1. add a pool preparation utility that only supports full per-worker copies;
2. add a worker runtime resolver;
3. wire eval launchers to use the resolver when sandboxing is enabled;
4. validate multi-env startup and second-episode stability on Linux;
5. after Linux validation, mirror the same prepare-and-resolve contract for native Windows evaluation.

## Validation Plan

### Functional Validation

- start 8 envs using 8 distinct sandboxes;
- verify no CRC mismatch or archive-modified errors occur;
- verify each worker still loads the intended task;
- verify second-episode mod switches still work.

### Isolation Validation

- print resolved sandbox path per worker at startup;
- confirm each worker sees a different active Mods root;
- confirm `Mods/all_tasks` points to the shared store;
- confirm bundle replacement occurs only inside the worker sandbox.

### Regression Validation

- re-run API evaluation;
- re-run single-env local demo;
- re-run RL bootstrap path once available;
- re-run native Windows evaluation once the Windows pool path is wired up;
- verify no change to single-env rollout semantics.

## Suggested First Implementation Slice

The lowest-risk first slice is:

1. add a standalone runtime-pool preparation script;
2. generate one full runtime sandbox per worker;
3. update eval launchers to select sandbox by `worker_index`;
4. verify the bundle race disappears before optimizing disk usage.

Only after that should the Unity-side loader be switched to consume `Mods/all_tasks/<task>` everywhere.

## Open Questions

1. Are task directories guaranteed read-only at runtime for all current and planned tasks?
2. Does Unity write any worker-specific cache or temp files outside the runtime tree that also need isolation?
3. Should the runtime pool be rebuilt eagerly per experiment or reused across many runs with a version stamp?
4. For RL training, should worker sandboxes be assigned statically or leased from a pool manager?
5. For Windows evaluation, is directory junction support acceptable as a standard dependency, or must the design support link-free fallback mode?

## Recommended Next Step

Complete the Python/runtime-sandbox rollout first, then validate multi-env startup and Unity-side loading against `Mods/all_tasks/<task>` on both Linux and Windows-native pools.
