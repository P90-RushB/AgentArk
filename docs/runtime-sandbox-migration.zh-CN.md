# 运行时沙箱迁移草案

[English](runtime-sandbox-migration.md) | 简体中文

## 目标

从共享 Unity 运行时目录迁移到每个 worker 独立的运行时沙箱，从而：

- 每个环境实例都有自己的可写运行时状态；
- 并行评测和 RL worker 不会争用共享的 Mods、配置和 bundle 文件；
- 任务资源仍可只读共享，避免复制数千个 Mod。

本草案有意限定于 ArkSubEnv 和 ArkEnv 当前使用的本地 Unity 执行模型。

## 平台范围

迁移必须支持两类运行时：

- RL 训练：仅 Linux；
- 评测：Linux 和 Windows。

因此，沙箱架构在概念上可以共用，但具体准备过程和链接实现必须感知平台。

非目标：

- 让 Linux 和 Windows 共用同一个运行时池。

每个平台都应准备并使用自己的运行时模板、pool root 和 task-store 映射。

## 第一阶段决策

当前实现方向确定如下：

1. 配置位于 `env_cfg.runtime_sandbox`。
2. 当前运行时沙箱为每个 worker 完整复制一份运行时，并在每个 worker 沙箱中暴露共享、
   只读的 `Mods/all_tasks` store。
3. demo、评测和 RL 流程统一使用 `worker_index` 选择沙箱。
4. 底层环境 constructor 可以把沙箱化的 `env_path` 和 `mod_path` 解析作为安全 fallback，
   但优先由 launcher 侧执行 ensure 编排。
5. 当前 template fingerprint 策略会递归遍历运行时树，根据相对路径、文件大小和修改时间
   元数据计算 hash。
6. `runtime_sandbox.shared_task_store_path` 是共享任务仓库在 host 侧的 source of truth。
7. `link_mode=auto` 时，Linux 应使用目录 symlink；Windows 应优先使用目录 junction；
   只有前两者不可用时才 fallback 到 `copy`。

这些决策定义了当前的共享 task-store 实现。

当前的重要假设：

- 任务目录存放在一个共享 task store 中，并在每个 worker 内暴露为
  `Mods/all_tasks/<task>`。
- `mod_path` 仍表示 worker 私有且可写的 `Mods/` root，而不是共享 task-store root。
- 核心目标仍是隔离可写运行时状态，同时让并发环境中的任务选择保持确定性。

## 问题概述

当前布局让所有本地环境实例共用同一个运行时目录。

已经观察到的故障模式：

- 多个 Unity 实例加载同一个共享 Mods root；
- Unity 侧 ModManager 在运行时替换 Mods root 中的活动 bundle；
- 其他 Unity 实例可能仍然打开着该 bundle；
- Addressables 随后报告 CRC mismatch、invalid path，或 archive 在打开期间被修改；
- Python 也可能同时重写同一个 Mods root 中的 `config.yaml` 和 `config.json`。

这是沙箱隔离问题，不是端口分配问题。

## 设计原则

1. 可写运行时状态必须为每个 worker 私有。
2. 体积较大的任务资源应尽可能只读共享。
3. 在一次运行中，worker 到运行时的映射应保持稳定。
4. 运行时准备过程应明确且可脚本化。
5. 设计必须适用于本地 demo、评测和未来的 RL 训练。

## 推荐目标架构

把当前单一的 `mod_path` 概念拆成两个角色：

- 活动运行时 Mods root：可写，每个 worker 私有；
- task store：包含所有任务目录的只读共享仓库。

### 目标布局

host 布局示例：

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

Windows 评测应在 Windows 原生 root 下使用等价布局，例如：

```text
<AGENTARK_RUNTIME_TEMPLATE_ROOT>\
<AGENTARK_TASK_STORE_PATH>\
<AGENTARK_RUNTIME_POOL_ROOT>\worker_000\
```

关键规则：活动 worker 沙箱必须位于运行 Unity 的平台原生文件系统上。

### 每个 Worker 私有的内容

- 运行时可执行文件目录；
- 活动 Mods root；
- Mods root 下的 `config.yaml` 和 `config.json`；
- Mods root 下当前活动的共享 bundle 文件；
- 未来由任务生成的运行时文件；
- 如果后续重定向，worker 专属日志。

### 可以只读共享的内容

- 所有任务源码目录；
- 任务 metadata 文件；
- 将被复制到活动 Mods root 的任务专属源 bundle；
- 其他在运行时不会修改的静态资源。

## `all_tasks` 约定

当前风险较低的 Unity 侧约定为：

- 把所有任务目录从 Mods root 移到一个共享 `all_tasks` 仓库；
- 通过适合平台的链接，在每个 worker 运行时中把该仓库暴露为 `Mods/all_tasks`；
- Unity 侧加载逻辑继续以 `Mods/all_tasks/<task>` 为 source；
- 继续把选中任务的运行时 bundle 复制到 worker 私有 Mods root。

这样既保留 Addressables 必须从固定 Mods root 加载的现有要求，又消除了对该 root 的
共享修改。

### 各平台链接策略

设计应把 `Mods/all_tasks` 视为逻辑链接目标，而不是强制使用某一种 OS primitive。

Linux：

- 优先使用目录 symlink。

Windows：

- 优先使用目录 junction 或目录 symlink；
- 没有 symlink 权限时，目录链接 fallback 到 junction；
- 只有目标环境无法创建链接时，才复制任务目录。

实现注意事项：

- 由 pool preparation utility 负责选择链接方式，launcher 无需关心 `all_tasks` 最终是
  symlink、junction 还是复制目录。

### 必须维持的不变量

运行时必须把 `all_tasks` 视为严格只读。

如果某个任务执行时会写入自己的任务目录，则在没有额外沙箱层时，该任务不能安全地
存放在共享 store 中。

## 准备策略

采用两层准备模型。

### 主要机制：显式准备脚本

提供专门脚本，用于准备或刷新运行时池。

建议输入：

- 运行时模板路径；
- runtime pool root；
- shared task store 路径；
- pool size；
- 目标平台；
- 可选的强制刷新 flag；
- 可选的 manifest/version stamp。

建议职责：

- 创建 `worker_000 ... worker_N` 沙箱；
- 把运行时模板复制到每个 worker 沙箱；
- 如果每个 worker 的 Mods 中不再需要打包任务目录，则移除这些目录；
- 根据平台选择正确的 link mode 创建 `Mods/all_tasks`；
- 验证沙箱是否包含预期的私有文件和共享链接。

准备实现必须能在 Linux 和 Windows 中调用，不能假设 WSL 专用路径、`/mnt/...` 或仅
Linux 可用的 symlink 语义。

### 次要机制：Launcher 侧 Ensure

demo/eval/train 入口可以在启动 worker 前执行轻量 ensure。

ensure 应当：

- 验证请求的 worker 数量已经存在；
- 验证 version stamp 与模板匹配；
- 在 worker 启动前失败或重建；
- 禁止每个 worker 进程并发构建自己的沙箱。

这样既方便临时本地运行，又不会把运行时构建放入 reset 路径。

Windows 评测使用相同 ensure 约定，但必须验证 Windows 原生 pool root 和 Windows
运行时模板，而不是复用 Linux 假设。

## Worker 生命周期

### 本地 Demo / 评测

- 每个并行评测或本地 worker 选择唯一的 `worker_index`；
- 每个 `worker_index` 映射到一条沙箱路径；
- 整个进程生命周期复用该沙箱；
- 环境 reset 只修改该 worker 的私有运行时。

worker 映射必须在 Linux 和 Windows 上一致。高层评测代码应通过平台无关的 helper
解析沙箱路径，而不是自行拼接路径。

### RL 训练

- 每个长期存活的 rollout worker 获得一个沙箱 assignment；
- 沙箱跨 episode 和 attempt 存活；
- 某个 worker 损坏运行时时，只重建该 worker 的沙箱；
- pool size 应匹配 Unity 实例的最大并发数，而不是 episode 数量。

## 配置方向

当前 `mod_path` 字段混合了两个概念。迁移应逐步拆分为：

- `runtime_path`：worker 专属 Unity 可执行文件或沙箱 root；
- `active_mod_path`：worker 专属的可写 Mods root；
- `task_store_path`：包含所有任务的共享只读仓库；
- `runtime_pool_root`：所有 worker 沙箱的父目录；
- `runtime_platform`：`linux` 或 `windows`；
- `worker_index`：沙箱 selector。

Python 和 Unity 迁移期间，为兼容现有配置，可以暂时保留 `mod_path` 作为
`active_mod_path` 的 alias。

评测配置示例最终应同时提供 Linux 和 Windows 风格路径，而不是假设只有一种 OS 路径
约定。

## 建议迁移阶段

### 第一阶段：最小逻辑改动的完整沙箱复制

目标：

- 先消除共享可写运行时状态。

方式：

- 每个 worker 复制一份完整运行时目录；
- 在每个沙箱中保留当前任务布局；
- launcher 改为按 worker 选择沙箱。

针对当前运行时的明确约定：

- 任务目录位于共享 task store，并在每个 worker 中暴露为
  `Mods/all_tasks/<task_name>/...`；
- Python 侧 case discovery 通过 `Mods/all_tasks` 解析任务，root 配置仍写入 `Mods`；
- 要完成端到端验证，Unity 侧任务加载必须始终从
  `Mods/all_tasks/<task_name>` 读取。

跨平台注意事项：

- 训练依赖 Linux，因此先在 Linux 上完成；
- Windows 使用相同的准备接口，让评测随后能使用 Windows 模板和同一概念模型。

收益：

- 实现风险最低；
- 能最快验证 CRC 和 bundle 竞态是否消失。

成本：

- 磁盘占用最高。

### 第二阶段：共享 `all_tasks` Store

目标：

- 降低存储成本，同时不重新引入 bundle 竞态。

方式：

- 把任务目录移到单一共享 task store；
- 在每个沙箱中把该 store 暴露为 `Mods/all_tasks`；
- 每个私有 Mods root 只保留可写活动文件。

收益：

- 当任务数量大幅增长时显著降低磁盘占用；
- 保留 Unity 侧固定 Mods root 的现有设计。

风险：

- 必须为任务目录建立明确的只读约定。

### 第三阶段：配置与 Launcher 清理

目标：

- 在代码和配置中显式表达新架构。

方式：

- 将 `mod_path` 拆成明确的运行时字段和 task-store 字段；
- 集中处理 worker 沙箱路径解析；
- 为 demo、eval 和 RL launcher 提供同一个 pool preparation utility。

收益：

- 减少 Python 代码中的歧义；
- 更易长期维护。

## 为什么能解决当前 Bundle 错误

具体的 CRC mismatch 和 invalid path 故障来自多个进程并发修改同一个共享 Mods root
中的同一个 bundle 文件。

使用每个 worker 独立的沙箱后：

- worker A 只修改 `worker_A/Mods/...bundle`；
- worker B 只修改 `worker_B/Mods/...bundle`；
- 任一 worker 都无法使另一个 worker 已打开的 archive handle 失效。

这消除了已经观察到的根因。它不能保证所有未来的 Unity 故障都会消失，但应当能消除
当前共享 bundle 的竞态。

## 路径放置建议

条件允许时，建议把运行时池放在 WSL 内的 Linux 文件系统，而不是 `/mnt/c/...`：

- 小文件 metadata 性能更好；
- symlink 行为更可预测；
- 高并发文件操作时更少遇到跨文件系统问题。

如果源运行时必须保留在 `/mnt/c/...`，准备后的 worker 沙箱仍可以放在 Linux 侧。

原生 Windows 评测应把运行时池放在 Windows 原生本地路径，而不是 WSL mount 路径。

推荐规则：

- Linux Unity 进程使用 Linux 原生 pool root；
- Windows Unity 进程使用 Windows 原生 pool root。

如果评测还需在 WSL 外运行，pool 设计就不能依赖 WSL 专属路径假设。

## 跨平台准备模型

两个平台使用同一种高层 pool 模型，但各自使用不同的具体模板。

Linux：

- 模板 source 示例：`<AGENTARK_RUNTIME_TEMPLATE_ROOT>/...`；
- pool root 示例：`/workspace/runtime_pool/...` 或其他 ext4 路径；
- 首选链接类型：symlink。

Windows：

- 模板 source 示例：`<AGENTARK_RUNTIME_TEMPLATE_ROOT>\...`；
- pool root 示例：`<AGENTARK_RUNTIME_POOL_ROOT>\...`；
- 首选链接类型：junction 或目录 symlink。

这是同一架构的两种操作模式，不是两套不同架构。

## Launcher 兼容要求

运行时沙箱机制应位于 demo/eval/RL 业务逻辑的下一层。

具体而言：

- 本地和并行 API 评测按 `worker_index` 请求沙箱；
- 未来 RL worker 按 `worker_index` 请求沙箱；
- 这些入口都无需关心沙箱是在 Linux 还是 Windows 上准备的。

一个小型 resolver 层把以下输入：

- runtime platform；
- runtime pool root；
- worker index；

映射为：

- executable path；
- active Mods root；
- task-store mapping metadata。

## 第一阶段具体接口草案

本节把运行时沙箱提案收敛成共享 `all_tasks` 实现的具体约定。

### 准备脚本约定

建议入口：

- `python -m agent_ark.tools.prepare_runtime_pool`

建议参数：

- `--runtime-platform {linux,windows}`
- `--template-root <path>`
- `--shared-task-store-path <path>`
- `--pool-root <path>`
- `--pool-size <int>`
- `--force-refresh`
- `--clean-extra-workers`
- `--manifest-name runtime_pool_manifest.json`
- `--link-mode {auto,symlink,junction,copy}`，供后续第二阶段使用

第一阶段行为：

- 验证 template root 存在并包含可运行的 Unity build；
- 在 pool root 下创建 `worker_000 ... worker_N`；
- 把完整运行时模板复制到每个 worker 沙箱；
- 在 pool root 写入一个描述已准备 pool 的 manifest；
- 可选地移除超出请求 pool size 的过期 worker。

第一阶段输出：

- 每个 worker index 对应一个已准备的运行时沙箱；
- 一个 pool manifest；
- 准备不完整或不一致时返回非零退出码。

建议 manifest 字段：

- `runtime_platform`
- `template_root`
- `pool_root`
- `pool_size`
- `template_fingerprint`
- `prepared_at`
- `worker_dirs`
- `layout_version`

脚本应当幂等：使用相同模板和 pool size 重复运行时，要么不执行任何修改，要么只刷新
已经过期的内容。

### Launcher 侧 Ensure 约定

面向 launcher 的 API 应比准备脚本更轻量。

建议行为：

- 从 pool root 读取 manifest；
- 验证请求的 worker index 范围存在；
- 验证 template fingerprint 仍与预期模板一致；
- pool 缺失或过期时快速失败并给出明确错误；
- `auto_prepare=true` 时可选择调用准备脚本。

重要约束：

- ensure 只在 launcher/coordinator 进程中执行一次，不能在每个 worker thread 或每次
  env reset 中运行。

### Resolver 约定

建议的内部 Python helper：

```python
resolve_worker_runtime(
    runtime_platform: str,
    pool_root: str,
    worker_index: int,
) -> dict
```

建议返回字段：

- `worker_index`
- `worker_name`
- `runtime_root`
- `env_path`
- `active_mod_path`
- `pool_root`
- `runtime_platform`

第一阶段解析规则：

- `runtime_root = <pool_root>/worker_<NNN>`；
- `env_path` 是该 worker 沙箱中的 Unity 可执行文件；
- `active_mod_path` 是 worker 专属的 `.../Resources/Mods` 目录；
- 解析逻辑暂不依赖任务 identity。

该 resolver 最终应成为唯一了解 worker 沙箱布局的代码位置。

## 最小配置草案

第一阶段的最小配置面应足以支持 demo、评测和 RL worker。

建议字段：

```yaml
runtime_sandbox:
  enabled: true
  runtime_platform: linux
  template_root: /abs/path/to/runtime_template
  pool_root: /abs/path/to/runtime_pool
  auto_prepare: false
  pool_size: 8
```

运行时使用规则：

- `runtime_sandbox.enabled=false` 时，保留当前直接使用 `env_path` + `mod_path` 的行为；
- `runtime_sandbox.enabled=true` 时，必须通过 sandbox pool 解析 `worker_index`；
- 解析出的 `env_path` 和 `active_mod_path` 应在运行时覆盖原始配置路径。

Windows 评测使用相同结构和 Windows 原生路径，例如：

```yaml
runtime_sandbox:
  enabled: true
  runtime_platform: windows
  template_root: "${AGENTARK_RUNTIME_TEMPLATE_ROOT}"
  pool_root: "${AGENTARK_RUNTIME_POOL_ROOT}"
  auto_prepare: false
  pool_size: 8
```

## 第一阶段目录示例

### Linux 示例

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

`worker_index=3` 的解析值：

- `runtime_root=/workspace/unity_runtime_pool/worker_003`
- `env_path=/workspace/unity_runtime_pool/worker_003/AgentArk.x86_64`
- `active_mod_path=/workspace/unity_runtime_pool/worker_003/AgentArk_Data/Resources/Mods`

### Windows 示例

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

`worker_index=3` 的解析值：

- `runtime_root=<AGENTARK_RUNTIME_POOL_ROOT>\worker_003`
- `env_path=<AGENTARK_RUNTIME_POOL_ROOT>\worker_003\AgentArk.exe`
- `active_mod_path=<AGENTARK_RUNTIME_POOL_ROOT>\worker_003\AgentArk_Data\Resources\Mods`

## 入口集成草案

当前运行时路径解析应插入在入口业务逻辑下方。

建议集成顺序：

1. launcher 读取配置；
2. launcher 决定目标 worker index；
3. launcher 确保目标平台的 pool 已存在；
4. launcher 为每个 worker 解析沙箱路径；
5. 构造 ArkSubEnv 或 ArkEnv 前，把解析出的 `env_path` 和 `active_mod_path` 注入
   env config。

这样本地评测、并行评测和 RL launcher 都能共用同一沙箱机制。

## 推荐的第一段代码实现

若从本草案开始实现，范围最小的第一步应为：

1. 增加只支持每个 worker 完整复制的 pool preparation utility；
2. 增加 worker runtime resolver；
3. 沙箱启用时，让评测 launcher 使用 resolver；
4. 在 Linux 上验证多环境启动和第二个 episode 的稳定性；
5. Linux 验证后，把同一 prepare-and-resolve 约定扩展到原生 Windows 评测。

## 验证计划

### 功能验证

- 使用 8 个独立沙箱启动 8 个环境；
- 验证没有 CRC mismatch 或 archive-modified 错误；
- 验证每个 worker 仍加载预期任务；
- 验证第二个 episode 切换 Mod 仍然正常。

### 隔离验证

- 启动时打印每个 worker 解析出的沙箱路径；
- 确认每个 worker 使用不同的 active Mods root；
- 确认 `Mods/all_tasks` 指向共享 store；
- 确认 bundle 替换只发生在 worker 沙箱内部。

### 回归验证

- 重新运行 API 评测；
- 重新运行单环境本地 demo；
- RL bootstrap 可用后重新运行；
- Windows pool 路径接入后重新运行原生 Windows 评测；
- 验证单环境 rollout 语义不变。

## 建议的首个实现切片

风险最低的首个切片是：

1. 增加独立的 runtime-pool preparation 脚本；
2. 为每个 worker 生成一份完整运行时沙箱；
3. 更新评测 launcher，让其按 `worker_index` 选择沙箱；
4. 在优化磁盘占用前，先验证 bundle 竞态已经消失。

之后再让 Unity 侧 loader 全面切换到 `Mods/all_tasks/<task>`。

## 待确认问题

1. 当前及规划中的所有任务目录是否都保证运行时只读？
2. Unity 是否会在运行时树外写入也需要隔离的 worker 专属 cache 或临时文件？
3. runtime pool 应在每次实验前主动重建，还是使用 version stamp 跨多次运行复用？
4. RL 训练中，worker 沙箱应静态分配，还是由 pool manager 租赁？
5. 原生 Windows 评测能否把目录 junction 作为标准依赖，还是必须支持完全不使用链接的
   fallback 模式？

## 建议下一步

先完成 Python/运行时沙箱 rollout，再针对 Linux 和 Windows 原生 pool 验证多环境启动，
以及 Unity 侧从 `Mods/all_tasks/<task>` 加载任务的行为。
