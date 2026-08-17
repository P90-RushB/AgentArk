import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_ark.ark_env.direct_env import EnvInfoManager, EnvWrapper  # noqa: E402


AUDIT_PATH = ROOT.parent / ".agents" / "skills" / "agentark-task-config" / "scripts" / "audit_task_config.py"
_audit_spec = importlib.util.spec_from_file_location("agentark_task_config_audit", AUDIT_PATH)
assert _audit_spec is not None and _audit_spec.loader is not None
audit_task_config = importlib.util.module_from_spec(_audit_spec)
_audit_spec.loader.exec_module(audit_task_config)


def _valid_code_runtime(*, policy="stop_at_step_boundary", live_actions=1):
    return {
        "previous_action_policy": policy,
        "max_live_actions": live_actions,
        "max_source_bytes": 32768,
        "checkpoint_budget_per_callback": 10000,
    }


class CodeRuntimeConfigTest(unittest.TestCase):
    def test_raw_exact_code_requires_task_local_block_and_policy_fields(self):
        with self.assertRaisesRegex(ValueError, "task-local code_runtime mapping"):
            EnvInfoManager._validate_raw_code_runtime_config(
                {"action_mode": "code"}, context="raw task"
            )

        with self.assertRaisesRegex(ValueError, "previous_action_policy"):
            EnvInfoManager._validate_raw_code_runtime_config(
                {"action_mode": " code ", "code_runtime": {"max_live_actions": 1}},
                context="raw task",
            )

        # Budget limits are inherited at raw-task validation time, but policy
        # decisions must never be inherited from the root config.
        EnvInfoManager._validate_raw_code_runtime_config(
            {
                "action_mode": "CODE",
                "code_runtime": {
                    "previous_action_policy": "stop_at_step_boundary",
                    "max_live_actions": 1,
                },
            },
            context="raw task",
        )

    def test_raw_exact_code_rejects_invalid_policy_and_stop_live_limit(self):
        with self.assertRaisesRegex(ValueError, "previous_action_policy"):
            EnvInfoManager._validate_raw_code_runtime_config(
                {
                    "action_mode": "code",
                    "code_runtime": _valid_code_runtime(policy="stop"),
                },
                context="raw task",
            )

    def test_raw_exact_code_enforces_all_authored_limit_bounds(self):
        for key, value in (
            ("max_live_actions", 17),
            ("max_live_actions", 0),
            ("max_source_bytes", 255),
            ("max_source_bytes", 65537),
            ("checkpoint_budget_per_callback", 0),
            ("checkpoint_budget_per_callback", 1_000_001),
        ):
            cfg = {"action_mode": "code", "code_runtime": _valid_code_runtime()}
            cfg["code_runtime"][key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(ValueError, key):
                    EnvInfoManager._validate_raw_code_runtime_config(cfg, context="raw task")

        # Raw authored budget fields are optional because the immutable base
        # template can supply them, but any authored value is still bounded.
        EnvInfoManager._validate_raw_code_runtime_config(
            {
                "action_mode": "code",
                "code_runtime": {
                    "previous_action_policy": "stop_at_step_boundary",
                    "max_live_actions": 1,
                },
            },
            context="raw task",
        )
        with self.assertRaisesRegex(ValueError, "requires max_live_actions=1"):
            EnvInfoManager._validate_raw_code_runtime_config(
                {
                    "action_mode": "code",
                    "code_runtime": _valid_code_runtime(live_actions=2),
                },
                context="raw task",
            )

    def test_effective_exact_code_requires_positive_integer_budgets(self):
        for key, value in (
            ("max_source_bytes", 255),
            ("max_source_bytes", True),
            ("checkpoint_budget_per_callback", 0),
            ("checkpoint_budget_per_callback", "10"),
        ):
            cfg = {"action_mode": "code", "code_runtime": _valid_code_runtime()}
            cfg["code_runtime"][key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(ValueError, key):
                    EnvInfoManager._validate_effective_code_runtime_config(
                        cfg, context="effective task"
                    )

        EnvInfoManager._validate_effective_code_runtime_config(
            {"action_mode": "code", "code_runtime": _valid_code_runtime()},
            context="effective task",
        )

    def test_effective_exact_code_enforces_upper_bounds_and_all_fields(self):
        for key, value in (
            ("max_live_actions", 17),
            ("max_source_bytes", 65537),
            ("checkpoint_budget_per_callback", 1_000_001),
        ):
            cfg = {"action_mode": "code", "code_runtime": _valid_code_runtime()}
            cfg["code_runtime"][key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(ValueError, key):
                    EnvInfoManager._validate_effective_code_runtime_config(
                        cfg, context="effective task"
                    )

        for missing in ("max_source_bytes", "checkpoint_budget_per_callback"):
            cfg = {"action_mode": "code", "code_runtime": _valid_code_runtime()}
            del cfg["code_runtime"][missing]
            with self.subTest(missing=missing):
                with self.assertRaisesRegex(ValueError, missing):
                    EnvInfoManager._validate_effective_code_runtime_config(
                        cfg, context="effective task"
                    )

    def test_non_exact_code_modes_bypass_code_runtime_validation(self):
        malformed = {"code_runtime": {"previous_action_policy": "bad", "max_live_actions": 0}}
        for mode in ("func", "tool", "", None, "CodeMode"):
            cfg = dict(malformed)
            if mode is not None:
                cfg["action_mode"] = mode
            with self.subTest(mode=mode):
                EnvInfoManager._validate_raw_code_runtime_config(cfg, context="raw task")
                EnvInfoManager._validate_effective_code_runtime_config(cfg, context="effective task")

    def test_unity_payload_preserves_code_runtime_without_mutating_input(self):
        env_cfg = {
            "action_mode": "code",
            "max_steps": 3,
            "code_runtime": _valid_code_runtime(
                policy="keep_until_exit", live_actions=2
            ),
            "task_params": {"private": True},
            "env_wrapper_cfg": {},
        }
        before = copy.deepcopy(env_cfg)

        payload = EnvWrapper._build_unity_env_params_payload(env_cfg)

        self.assertEqual(env_cfg, before)
        self.assertEqual(payload["code_runtime"], before["code_runtime"])
        self.assertNotIn("task_params", payload)
        self.assertNotIn("env_wrapper_cfg", payload)

    def test_system_prompt_has_exact_code_func_and_legacy_branches(self):
        info_dir = ROOT / "src" / "agent_ark" / "ark_env" / "info"
        expected = {
            "code": (info_dir / "system_prompt_code.txt").read_text(encoding="utf-8"),
            "func": (info_dir / "system_prompt_func.txt").read_text(encoding="utf-8"),
            "legacy": (info_dir / "system_prompt.txt").read_text(encoding="utf-8"),
        }
        manager = EnvInfoManager.__new__(EnvInfoManager)

        self.assertEqual(manager._get_system_prompt(action_mode=" code "), expected["code"])
        self.assertEqual(manager._get_system_prompt(action_mode="FUNC"), expected["func"])
        for mode in (None, "", "tool", "CodeMode"):
            with self.subTest(mode=mode):
                self.assertEqual(manager._get_system_prompt(action_mode=mode), expected["legacy"])

    def test_exact_code_action_envelope_is_stripped_without_touching_raw_source(self):
        wrapped = (
            "<think>bounded action</think>\n<code>\n"
            "public sealed class AgentAction : CodeAction { "
            "public override void Start() { Context.CompleteStep(); } }\n"
            "</code>"
        )
        raw = "public sealed class AgentAction : CodeAction { public override void Tick() { } }"
        explicit_sources = (
            ({"action_mode": " code "}, {}),
            ({}, {"action_mode": "CODE"}),
        )
        for env_config, cfg in explicit_sources:
            with self.subTest(env_config=env_config, cfg=cfg):
                env = object.__new__(EnvWrapper)
                env.env_info_mgr = SimpleNamespace(env_config=env_config)
                env.cfg = cfg

                rendered, errors = env._render_func_code_actions({1: wrapped, 2: raw})

                self.assertEqual(errors, {})
                self.assertEqual(
                    rendered[1],
                    "public sealed class AgentAction : CodeAction { "
                    "public override void Start() { Context.CompleteStep(); } }",
                )
                self.assertEqual(rendered[2], raw)

    def test_legacy_action_modes_do_not_enable_exact_code_envelope_stripping(self):
        wrapped = (
            "<think>legacy payload must remain byte-identical</think>\n"
            "<code>public sealed class LegacyAction { }</code>"
        )
        cases = (
            ("missing", {}, {}, "code"),
            ("empty", {"action_mode": ""}, {"action_mode": "code"}, "code"),
            ("tool", {"action_mode": "tool"}, {}, "tool"),
            ("unknown", {"action_mode": "unknown"}, {}, "unknown"),
        )
        for name, env_config, cfg, historical_mode in cases:
            with self.subTest(name=name):
                env = object.__new__(EnvWrapper)
                env.env_info_mgr = SimpleNamespace(env_config=env_config)
                env.cfg = cfg
                actions = {7: wrapped}

                self.assertEqual(env._get_action_mode(), historical_mode)
                rendered, errors = env._render_func_code_actions(actions)

                self.assertIs(rendered, actions)
                self.assertEqual(errors, {})
                self.assertEqual(rendered[7].encode("utf-8"), wrapped.encode("utf-8"))

    def test_func_action_rendering_is_unchanged_by_exact_code_detection(self):
        env = object.__new__(EnvWrapper)
        env.env_info_mgr = SimpleNamespace(env_config={"action_mode": "func"})
        env.cfg = {}
        env.ml_unity_id_map = {3: 11}
        env._render_func_wrapper = lambda unity_id, payload: f"{unity_id}|{payload}"
        actions = {3: '<tool_call>{"name":"Move","arguments":{}}</tool_call>'}

        rendered, errors = env._render_func_code_actions(actions)

        self.assertEqual(errors, {})
        self.assertEqual(rendered, {3: f"11|{actions[3]}"})

    def test_audit_enforces_authored_code_runtime_upper_bounds_only_for_exact_code(self):
        task_cfg = {
            "action_mode": "code",
            "code_runtime": _valid_code_runtime(live_actions=17),
        }
        audit = audit_task_config.audit_config(
            Path("Task/task_config.yaml"), task_cfg, None, None, None, None
        )
        self.assertTrue(
            any(issue["code"] == "code-runtime-live-actions" for issue in audit.issues)
        )

        legacy_cfg = {
            "action_mode": "tool",
            "code_runtime": {
                "previous_action_policy": "bad",
                "max_live_actions": 999999,
                "max_source_bytes": -1,
            },
        }
        legacy_audit = audit_task_config.audit_config(
            Path("Task/task_config.yaml"), legacy_cfg, None, None, None, None
        )
        self.assertFalse(
            any(issue["code"].startswith("code-runtime-") and issue["severity"] == "ERROR" for issue in legacy_audit.issues)
        )

    def test_audit_requires_all_effective_code_runtime_fields_when_base_is_available(self):
        task_cfg = {
            "action_mode": "code",
            "code_runtime": {
                "previous_action_policy": "stop_at_step_boundary",
                "max_live_actions": 1,
            },
        }
        base_cfg = {
            "action_mode": "code",
            "code_runtime": {
                "previous_action_policy": "stop_at_step_boundary",
                "max_live_actions": 1,
            },
        }
        audit = audit_task_config.audit_config(
            Path("Task/task_config.yaml"), task_cfg, Path("Mods/config.yaml.bak"), base_cfg, None, None
        )
        missing_effective = [
            issue for issue in audit.issues if issue["code"] == "code-runtime-effective-field"
        ]
        self.assertEqual(len(missing_effective), 2)


if __name__ == "__main__":
    unittest.main()
