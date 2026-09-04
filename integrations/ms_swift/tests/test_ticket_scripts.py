from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
GENERATE = INTEGRATION_ROOT / "scripts" / "generate_tickets.py"
CHECK = INTEGRATION_ROOT / "scripts" / "check_ticket_capacity.py"
LAUNCHER = INTEGRATION_ROOT / "scripts" / "run_agentark_grpo.sh"
SERVER_LAUNCHER = INTEGRATION_ROOT / "scripts" / "run_agentark_server.sh"


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        check=check,
        capture_output=True,
        text=True,
    )


class TicketScriptTests(unittest.TestCase):

    def _run_launcher_with_fake_swift(
        self,
        temp_path: Path,
        *,
        builtin: bool,
        version: str,
        integration: str = "auto",
        plugin_path: str | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        python_wrapper = temp_path / "python-wrapper.sh"
        swift_wrapper = temp_path / "swift-wrapper.sh"
        captured_args = temp_path / "swift-args.txt"
        runtime_config = temp_path / "runtime.yaml"
        dataset = temp_path / "tickets.jsonl"
        output_dir = temp_path / "output"
        compat_dir = temp_path / "compat"

        runtime_config.write_text("env_cfg: {}\n", encoding="utf-8")
        compat_dir.mkdir()
        _run(
            str(GENERATE),
            "--output",
            str(dataset),
            "--run-id",
            "launcher-mode-test",
            "--count",
            "2",
        )
        python_wrapper.write_text(
            """#!/usr/bin/env bash
set -eu
if [[ "${1:-}" == "-c" && "${2:-}" == *"importlib.metadata"* ]]; then
  printf '%s\n' "$FAKE_SWIFT_VERSION"
  exit 0
fi
if [[ "${1:-}" == "-c" && "${2:-}" == *"agentark_scheduler"* ]]; then
  exit "$FAKE_BUILTIN_EXIT"
fi
if [[ "${1:-}" == *"check_agentark_server.py" ]]; then
  exit 0
fi
exec "$REAL_PYTHON" "$@"
""",
            encoding="utf-8",
        )
        python_wrapper.chmod(0o755)
        swift_wrapper.write_text(
            """#!/usr/bin/env bash
set -eu
printf '%s\n' "$@" > "$CAPTURE_FILE"
""",
            encoding="utf-8",
        )
        swift_wrapper.chmod(0o755)
        env = {
            **os.environ,
            "AGENTARK_SWIFT_PYTHON": str(python_wrapper),
            "AGENTARK_SWIFT_BIN": str(swift_wrapper),
            "AGENTARK_SWIFT_INTEGRATION": integration,
            "AGENTARK_SWIFT_COMPAT_DIR": str(compat_dir),
            "AGENTARK_MODEL": "org/example-model",
            "AGENTARK_RUNTIME_CONFIG": str(runtime_config),
            "AGENTARK_TICKET_DATASET": str(dataset),
            "AGENTARK_OUTPUT_DIR": str(output_dir),
            "AGENTARK_USE_VLLM": "false",
            "FAKE_SWIFT_VERSION": version,
            "FAKE_BUILTIN_EXIT": "0" if builtin else "1",
            "REAL_PYTHON": sys.executable,
            "CAPTURE_FILE": str(captured_args),
        }
        if plugin_path is not None:
            env["AGENTARK_SWIFT_PLUGIN"] = plugin_path

        result = subprocess.run(
            ["bash", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        args = captured_args.read_text(encoding="utf-8").splitlines() if captured_args.exists() else []
        return result, args

    def test_server_launcher_derives_bind_from_server_url(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            python_wrapper = temp_path / "python-wrapper.sh"
            captured_args = temp_path / "server-args.txt"
            python_wrapper.write_text(
                """#!/usr/bin/env bash
set -eu
if [[ "${1:-}" == "-" ]]; then
  exec "$REAL_PYTHON" "$@"
fi
printf '%s\n' "$@" > "$CAPTURE_FILE"
""",
                encoding="utf-8",
            )
            python_wrapper.chmod(0o755)
            env = {
                **os.environ,
                "AGENTARK_PYTHON_BIN": str(python_wrapper),
                "AGENTARK_SERVER_URL": "http://127.0.0.1:19001",
                "REAL_PYTHON": sys.executable,
                "CAPTURE_FILE": str(captured_args),
            }

            result = subprocess.run(
                ["bash", str(SERVER_LAUNCHER)],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            args = captured_args.read_text(encoding="utf-8").splitlines()
            self.assertIn("--host", args)
            self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")
            self.assertIn("--port", args)
            self.assertEqual(args[args.index("--port") + 1], "19001")

    def test_server_launcher_rejects_non_http_url(self):
        env = {
            **os.environ,
            "AGENTARK_PYTHON_BIN": sys.executable,
            "AGENTARK_SERVER_URL": "https://127.0.0.1:19001",
        }
        result = subprocess.run(
            ["bash", str(SERVER_LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("AGENTARK_SERVER_URL must be an HTTP URL", result.stderr)

    def test_launcher_is_model_and_tuner_generic(self):
        launcher = LAUNCHER.read_text(encoding="utf-8")
        self.assertNotIn("Qwen3.5-0.8B", launcher)
        self.assertNotIn('! -d "$MODEL_DIR"', launcher)
        self.assertIn('TUNER_TYPE="${AGENTARK_TUNER_TYPE:-lora}"', launcher)
        self.assertIn('if [[ "$TUNER_TYPE" == "lora" ]]', launcher)
        self.assertIn('SWIFT_TUNER_ARGS=(--tuner_type "$TUNER_TYPE")', launcher)

    def test_launcher_rejects_invalid_tuner_before_training(self):
        env = {
            **os.environ,
            "AGENTARK_SWIFT_PYTHON": sys.executable,
            "AGENTARK_SWIFT_BIN": shutil.which("true") or "/bin/true",
            "AGENTARK_MODEL": "org/example-model",
            "AGENTARK_TUNER_TYPE": "unsupported",
        }
        result = subprocess.run(
            ["bash", str(LAUNCHER)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("AGENTARK_TUNER_TYPE must be lora or full", result.stderr)

    def test_launcher_rejects_trailing_capacity_override_before_preflight(self):
        result = subprocess.run(
            ["bash", str(LAUNCHER), "--generation_batch_size=2"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Do not override --generation_batch_size", result.stderr)

    def test_launcher_prefers_builtin_agentark_without_external_plugin(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result, args = self._run_launcher_with_fake_swift(
                Path(temp_dir),
                builtin=True,
                version="9.9.0",
                plugin_path=str(Path(temp_dir) / "missing-plugin.py"),
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("agentark_integration=builtin", result.stdout)
        self.assertNotIn("--external_plugins", args)
        self.assertIn("--gym_env", args)
        self.assertEqual(args[args.index("--gym_env") + 1], "agentark")

    def test_launcher_falls_back_to_supported_external_adapter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result, args = self._run_launcher_with_fake_swift(
                Path(temp_dir),
                builtin=False,
                version="4.6.0.dev0",
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("agentark_integration=external", result.stdout)
        self.assertIn("--external_plugins", args)
        plugin = args[args.index("--external_plugins") + 1]
        self.assertTrue(plugin.endswith("src/agentark_swift/plugin.py"))

    def test_launcher_builtin_mode_fails_if_swift_lacks_agentark(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result, args = self._run_launcher_with_fake_swift(
                Path(temp_dir),
                builtin=False,
                version="4.6.0.dev0",
                integration="builtin",
            )

        self.assertEqual(result.returncode, 2)
        self.assertFalse(args)
        self.assertIn("does not provide the built-in AgentArk", result.stderr)

    def test_pinned_task_derives_stable_distinct_group_seeds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "tickets.jsonl"
            command = (
                str(GENERATE),
                "--output",
                str(dataset),
                "--run-id",
                "stable-seed-test",
                "--count",
                "3",
                "--task-name",
                "Pushbox",
            )
            _run(*command)
            first_bytes = dataset.read_bytes()
            rows = [json.loads(line) for line in dataset.read_text().splitlines()]

            self.assertEqual(len(rows), 3)
            self.assertEqual(len({row["env_config"]["group_seed"] for row in rows}), 3)
            for index, row in enumerate(rows):
                uid = f"stable-seed-test:{index:08d}"
                self.assertEqual(row["env_config"]["group_uid"], uid)
                self.assertEqual(row["env_config"]["task_name"], "Pushbox")
                self.assertEqual(row["messages"][0]["content"], f"<agentark-ticket:{uid}>")

            _run(*command, "--force")
            self.assertEqual(dataset.read_bytes(), first_bytes)

    def test_capacity_is_aligned_to_complete_prompt_group_batches(self):
        result = _run(
            str(CHECK),
            "--max-steps",
            "3",
            "--per-device-train-batch-size",
            "4",
            "--world-size",
            "2",
            "--gradient-accumulation-steps",
            "1",
            "--num-generations",
            "4",
            "--reserve-percent",
            "10",
            "--print-required",
        )
        # 8 trajectories / 4 generations = 2 groups per batch. Three steps
        # need 6 rows, 10% reserve rounds to 7, then batch alignment gives 8.
        self.assertEqual(result.stdout.strip(), "8")

    def test_explicit_small_generation_batch_counts_every_rollout_call(self):
        base_args = (
            str(CHECK),
            "--max-steps",
            "4",
            "--per-device-train-batch-size",
            "2",
            "--world-size",
            "1",
            "--gradient-accumulation-steps",
            "4",
            "--num-generations",
            "2",
            "--generation-batch-size",
            "2",
            "--print-required",
        )
        # Swift derives steps_per_generation=1. Four optimizer steps contain
        # 16 micro-steps, so K=1 triggers 16 separate one-group rollouts.
        self.assertEqual(_run(*base_args).stdout.strip(), "16")
        # K=2 reuses each generated group for two micro-steps.
        with_iterations = (*base_args[:-1], "--num-iterations", "2", base_args[-1])
        self.assertEqual(_run(*with_iterations).stdout.strip(), "8")

    def test_capacity_checker_rejects_duplicate_group_uid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "tickets.jsonl"
            _run(
                str(GENERATE),
                "--output",
                str(dataset),
                "--run-id",
                "duplicate-test",
                "--count",
                "2",
            )
            rows = dataset.read_text().splitlines()
            dataset.write_text(f"{rows[0]}\n{rows[0]}\n", encoding="utf-8")

            result = _run(
                str(CHECK),
                "--dataset",
                str(dataset),
                "--max-steps",
                "1",
                "--per-device-train-batch-size",
                "2",
                "--world-size",
                "1",
                "--gradient-accumulation-steps",
                "1",
                "--num-generations",
                "2",
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            payload = json.loads(result.stdout)
            self.assertFalse(payload["ok"])
            self.assertTrue(any("duplicate group_uid" in error for error in payload["errors"]))

    def test_capacity_checker_rejects_pinned_task_without_group_seed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "unsafe-pinned-ticket.jsonl"
            uid = "unsafe-pinned:00000000"
            dataset.write_text(
                json.dumps(
                    {
                        "messages": [{"role": "user", "content": f"<agentark-ticket:{uid}>"}],
                        "env_config": {
                            "name": "agentark",
                            "group_uid": uid,
                            "task_name": "Pushbox",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = _run(
                str(CHECK),
                "--dataset",
                str(dataset),
                "--max-steps",
                "1",
                "--per-device-train-batch-size",
                "2",
                "--world-size",
                "1",
                "--gradient-accumulation-steps",
                "1",
                "--num-generations",
                "2",
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            errors = json.loads(result.stdout)["errors"]
            self.assertTrue(any("pinned task_name requires group_seed" in error for error in errors))

    def test_capacity_checker_rejects_pinned_env_and_top_level_media(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir) / "unsafe-media-ticket.jsonl"
            uid = "unsafe-media:00000000"
            dataset.write_text(
                json.dumps(
                    {
                        "messages": [{"role": "user", "content": f"<agentark-ticket:{uid}>"}],
                        "images": ["/tmp/stale.png"],
                        "env_config": {
                            "name": "agentark",
                            "group_uid": uid,
                            "env_id": "one-runtime-for-all-g",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = _run(
                str(CHECK),
                "--dataset",
                str(dataset),
                "--max-steps",
                "1",
                "--per-device-train-batch-size",
                "2",
                "--world-size",
                "1",
                "--gradient-accumulation-steps",
                "1",
                "--num-generations",
                "2",
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            errors = json.loads(result.stdout)["errors"]
            self.assertTrue(any("env_id must not be pinned" in error for error in errors))
            self.assertTrue(any("top-level images is not supported" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
