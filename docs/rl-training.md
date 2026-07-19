# RL Training: Architecture and Semantics

English | [简体中文](rl-training.zh-CN.md)

AgentArk exposes its Unity runtime pool as an HTTP environment service. Two
GRPO paths are maintained; use their runbooks for executable instructions:

- [ms-swift runbook (Chinese)](../integrations/ms_swift/README.md): the adapter
  lives in this repository, currently targets the validated ms-swift 4.4.1
  stack, and defaults to AgentArk HTTP protocol v2.
- [VERL integration guide](../integrations/verl/README.md): the AgentArk bridge
  lives here, while the trainer adapter lives in the public `agentark_rl` fork
  and currently uses protocol v1.

See the [RL integrations index](../integrations/README.md) for navigation. This
document explains shared architecture, synchronization, grouping, task
selection, and failure semantics without duplicating either runbook.

## 1. Shared architecture

```text
┌────────────── trainer Python environment ─────────────┐
│  dataset → rollout/model server → framework adapter  │
└───────────────────────┬───────────────────────────────┘
                        │ HTTP v1 or v2
┌───────────────────────▼───────────────────────────────┐
│ AgentArk Env Server (one service per host/port)       │
│ task/seed selector · session/lease · runtime pool     │
└───────────────────────┬───────────────────────────────┘
                        │
             multiple runtime sandboxes / Unity children
```

The AgentArk wrapper and Env Server use Python 3.10.12. ms-swift or VERL uses
its own Python environment and talks over HTTP without importing `agent_ark`.

“One Env Server process” means that one address must not have multiple server
workers with independent in-memory state. It does not limit Unity concurrency.
One `EnvSessionManager` allocates distinct worker indices and the existing
runtime sandbox mechanism isolates multiple Unity processes.

Protocol v1 and v2 use isolated namespaces. Within a namespace, a runtime can
only be reused for an equivalent semantic `env_cfg`, so warmup must use the
configuration that the selected adapter will actually send.

## 2. What asynchronous environment interaction means

Both integrations concurrently await resets and steps for multiple trajectories
inside one rollout/generation batch. A slow Unity task does not prevent Python
from initiating other environment requests. The ms-swift `AgentArkScheduler`
uses coroutines; the VERL agent loop also schedules trajectories asynchronously
inside its batch workers.

This is not a continuous pipeline across optimizer updates:

```text
collect every trajectory in batch N
                 ↓ barrier
update the policy with batch N
                 ↓
collect batch N+1
```

For the current ms-swift path, “async” therefore means within-batch environment
I/O concurrency. Rollout batches and optimizer updates still alternate
synchronously; the ms-swift 4.4.1 multi-turn Gym scheduler does not use an
`async_generate` cross-batch pipeline.

## 3. GRPO groups, tasks, and seeds

GRPO samples sibling trajectories for one prompt and computes relative
advantages from their rewards. AgentArk must reset siblings to the same task
and seed while leasing a different Unity runtime for each trajectory.

### ms-swift

- A static dataset ticket stores `group_uid`, and Swift repeats the row
  `G=num_generations` times.
- Siblings share `group_uid`, while Swift gives each real trajectory a distinct
  request UUID. The former selects the same task/seed; the latter identifies an
  independent lease and action stream.
- When the same ticket appears in another epoch, its `group_uid` remains stable,
  so the current selector maps it to the same task/seed. Generating enough
  tickets for the intended run avoids repeated rows; future curriculum policy
  can still deliberately change this behavior.

### VERL

- The trainer creates a `uid` each time a prompt group is consumed and copies
  it to that occurrence's n siblings.
- The current server-managed dataset also stores an explicit `group_seed`.
  Thus the uid selects the task and the row selects the seed; they are not both
  derived solely from uid.
- Consuming the row in another epoch creates a new uid, so its task may change
  while its row seed remains stable.

With a pinned `task_name`, both adapters forward the requested task. Keeping
task/curriculum selection on the AgentArk side lets that policy evolve without
embedding it in framework-specific trainer code.

## 4. Policy-loss scope

The ms-swift adapter exposes two choices:

- `all_turns` (default): every assistant turn contributes to policy loss;
  environment-observation tokens do not.
- `last_round`: intermediate assistant turns are masked and only the last
  assistant turn is retained.

See the [ms-swift runbook](../integrations/ms_swift/README.md) and
[architecture note (Chinese)](../integrations/ms_swift/ARCHITECTURE.zh-CN.md)
for the exact settings and driver masks. The current VERL recipe uses its own
agent-loop response masks for assistant output and masks environment
observations; it does not consume the Swift-specific loss-scope switch.

## 5. Integration comparison

| Dimension | ms-swift | VERL |
| --- | --- | --- |
| Adapter location | `integrations/ms_swift` in this repository | External fork; local bridge/preflight |
| Environment abstraction | Swift Gym Env + multi-turn scheduler | VERL async agent loop |
| Current protocol | v2 | v1 |
| Sibling identity | Static ticket `group_uid` + distinct request UUID | Per-occurrence `uid` |
| Cross-epoch default | Ticket task/seed stays stable | Task can change with uid; row seed stays stable |
| Assistant loss | `all_turns` or `last_round` | Current recipe's assistant response mask |
| Safe retries | Operation IDs and generation fencing | Transport/5xx retry, no end-to-end exactly-once proof |
| Abandoned leases | Heartbeat + TTL reclamation | Usually restart Server and rebuild v1 pool after a hard kill |
| Current topology | Validated colocate path | Current single-node FSDP2/vLLM recipe |

Protocol v1/v2 is an AgentArk Server capability, not inherently tied to an RL
framework. The table describes today's adapters; VERL can migrate to v2 later.

## 6. Mixing non-AgentArk data or environments

ms-swift can register multiple Gym environments, but the current AgentArk
launcher globally selects:

```text
--multi_turn_scheduler agentark_scheduler
--gym_env agentark
```

`AgentArkScheduler` consequently interprets every request in its batch as an
AgentArk ticket. The current path naturally mixes many AgentArk tasks in one
AgentArk dataset, but it does not yet provide an in-job dispatcher for ordinary
single-turn data or another Gym environment.

That form of mixed training needs a composite scheduler/dispatcher that routes
on row-level `env_config.name`, plus explicit reward scaling, group batching,
and loss-mask rules for every source. Concatenating datasets alone is not
sufficient. Separate jobs need no such router. The current VERL recipe is also
an AgentArk-specific dataset/agent-loop path.

## 7. Capacity and scaling

Increasing environment count normally requires no Env Server core-code change:
increase the final runtime configuration and warmup count for the selected
namespace. Also account for peak rollout leases, sandbox disk usage, Unity
CPU/RAM and Xvfb capacity, model context and visual-token memory, and long-tail
step latency across heterogeneous tasks.

The ms-swift runbook gives its exact generation-batch and ticket-capacity
formula. The VERL guide gives its peak-lease formula and exact-config warmup.

## 8. Failure and recovery boundary

The Server applies hard timeouts to blocking Unity reset/step/close calls,
discards broken runtimes, and can recycle long-lived runtimes after a configured
number of interactions.

Protocol v2 additionally has stable operation IDs, lease generations,
idempotent replay, heartbeats, and TTL. A client can recover a lost response
without blindly repeating an action, and a late request cannot operate a
re-leased runtime. A v1 client can retry transport/5xx failures but lacks those
identities, so a lost step response has no end-to-end exactly-once proof.

Normal trainer errors can rely on adapter cleanup. After `SIGKILL`, host failure,
or OOM, ms-swift/v2 can wait for TTL and inspect active leases. For the current
VERL/v1 path, restart the Server, warm the exact config again, and resume the
checkpoint.

For the full ms-swift data flow, script responsibilities, masks, and v2
lifecycle, see the
[ms-swift architecture note (Chinese)](../integrations/ms_swift/ARCHITECTURE.zh-CN.md).
