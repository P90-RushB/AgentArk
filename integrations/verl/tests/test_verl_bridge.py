from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
CHECKER = INTEGRATION_ROOT / "check_compatibility.py"
RENDERER = INTEGRATION_ROOT / "render_runtime_config.py"


def _run(*args: str, cwd: Path | None = None, env: dict[str, str] | None = None):
    return subprocess.run(
        list(args),
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _git(repo: Path, *args: str) -> str:
    result = _run("git", *args, cwd=repo)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _commit_all(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _make_checkout(tmp_path: Path) -> tuple[Path, Path, str]:
    repo = tmp_path / "verl"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "agentark_rl")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "AgentArk tests")

    recipe = repo / "agentark_recipe" / "agentark_env_agent"
    recipe.mkdir(parents=True)
    (recipe / "env_client.py").write_text(
        'ACQUIRE = "/v1/envs/acquire_start"\n'
        'STEP = "/v1/envs/{env_id}/step"\n'
        'RELEASE = "/v1/envs/{env_id}/release"\n',
        encoding="utf-8",
    )
    (recipe / "agentark_env_agent_loop.py").write_text(
        'group_uid = kwargs.get("uid")\ncall(uid=group_uid)\n', encoding="utf-8"
    )
    watch = repo / "verl" / "trainer.py"
    watch.parent.mkdir(parents=True)
    watch.write_text("BASELINE = True\n", encoding="utf-8")
    baseline = _commit_all(repo, "baseline")
    _git(repo, "remote", "add", "origin", "git@github.com:P90-RushB/verl.git")

    manifest = tmp_path / "compatibility.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "integration": "test",
                "repository": {
                    "canonical_url": "https://github.com/P90-RushB/verl.git",
                    "canonical_slug": "github.com/p90-rushb/verl",
                    "recommended_branch": "agentark_rl",
                    "minimum_compatible_commit": baseline,
                    "commit_policy": "ancestor_of_head",
                },
                "required_paths": [
                    "agentark_recipe/agentark_env_agent/env_client.py",
                    "agentark_recipe/agentark_env_agent/agentark_env_agent_loop.py",
                ],
                "protocol": {
                    "version": "v1",
                    "required_literals": [
                        {
                            "path": "agentark_recipe/agentark_env_agent/env_client.py",
                            "literals": [
                                "/v1/envs/acquire_start",
                                "/v1/envs/{env_id}/step",
                                "/v1/envs/{env_id}/release",
                            ],
                        },
                        {
                            "path": "agentark_recipe/agentark_env_agent/agentark_env_agent_loop.py",
                            "literals": ['group_uid = kwargs.get("uid")', "uid=group_uid"],
                        },
                    ],
                },
                "watch_paths": ["verl/trainer.py"],
            }
        ),
        encoding="utf-8",
    )
    return repo, manifest, baseline


def _check(repo: Path, manifest: Path, *extra: str):
    return _run(
        sys.executable,
        os.fspath(CHECKER),
        "--checkout",
        os.fspath(repo),
        "--manifest",
        os.fspath(manifest),
        "--format",
        "json",
        *extra,
    )


def _write_external_env_config(checkout: Path) -> None:
    config_dir = checkout / "agentark_recipe" / "agentark_env_agent" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "env_cfg.yaml").write_text(
        """\
server:
  host: http://127.0.0.1
  port: 18080
  timeout: 1200
env_cfg:
  env_path: ${oc.env:TEST_AGENTARK_ENV_PATH}
  mod_path: ${oc.env:TEST_AGENTARK_MOD_PATH}
  task_type: RLTask
  reset_timeout_s: 37
  runtime_sandbox:
    enabled: true
    pool_size: 3
    auto_prepare: true
""",
        encoding="utf-8",
    )


class CompatibilityTests(unittest.TestCase):
    def test_exact_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, manifest, _ = _make_checkout(Path(tmp))
            result = _check(repo, manifest)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "compatible")
            self.assertEqual(payload["warnings"], [])

    def test_descendant_is_accepted_and_review_path_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, manifest, _ = _make_checkout(Path(tmp))
            (repo / "verl" / "trainer.py").write_text("BASELINE = False\n", encoding="utf-8")
            _commit_all(repo, "change watched code")

            result = _check(repo, manifest)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "compatible")
            self.assertTrue(
                any("review-sensitive paths changed" in item for item in payload["warnings"])
            )

            strict = _check(repo, manifest, "--strict")
            self.assertEqual(strict.returncode, 1)
            self.assertEqual(json.loads(strict.stdout)["status"], "incompatible")

    def test_missing_protocol_literal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, manifest, _ = _make_checkout(Path(tmp))
            client = repo / "agentark_recipe" / "agentark_env_agent" / "env_client.py"
            client.write_text('ACQUIRE = "/v2/envs/acquire-start"\n', encoding="utf-8")
            _commit_all(repo, "remove v1 contract")

            result = _check(repo, manifest)
            self.assertEqual(result.returncode, 1)
            payload = json.loads(result.stdout)
            self.assertTrue(
                any("protocol contract literal missing" in item for item in payload["errors"])
            )

    def test_unavailable_baseline_is_indeterminate(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, manifest, _ = _make_checkout(Path(tmp))
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data["repository"]["minimum_compatible_commit"] = "0" * 40
            manifest.write_text(json.dumps(data), encoding="utf-8")

            result = _check(repo, manifest)
            self.assertEqual(result.returncode, 2)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "indeterminate")
            self.assertTrue(payload["indeterminate"])

    def test_required_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, manifest, _ = _make_checkout(Path(tmp))
            client = repo / "agentark_recipe" / "agentark_env_agent" / "env_client.py"
            client.unlink()
            client.symlink_to("agentark_env_agent_loop.py")
            _commit_all(repo, "replace required file with symlink")

            result = _check(repo, manifest)
            self.assertEqual(result.returncode, 1)
            self.assertTrue(
                any(
                    "tracked regular file" in item or "missing or unsafe" in item
                    for item in json.loads(result.stdout)["errors"]
                )
            )


class RuntimeConfigRendererTests(unittest.TestCase):
    def test_uses_resolved_external_config_and_pool_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkout = tmp_path / "verl"
            _write_external_env_config(checkout)
            output = tmp_path / "rendered.json"
            env = os.environ.copy()
            env.update(
                TEST_AGENTARK_ENV_PATH="/runtime/AgentArk.x86_64",
                TEST_AGENTARK_MOD_PATH="/runtime/mod.zip",
            )
            result = _run(
                sys.executable,
                os.fspath(RENDERER),
                "--verl-root",
                os.fspath(checkout),
                "--output",
                os.fspath(output),
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rendered = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(rendered["warmup"]["num_envs"], 3)
            self.assertEqual(rendered["env_cfg"]["env_path"], "/runtime/AgentArk.x86_64")
            self.assertEqual(rendered["env_cfg"]["reset_timeout_s"], 37)
            self.assertEqual(rendered["_agentark_verl_bridge"]["protocol_version"], "v1")

    def test_capacity_above_declared_pool_preserves_external_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkout = tmp_path / "verl"
            _write_external_env_config(checkout)
            env = os.environ.copy()
            env.update(TEST_AGENTARK_ENV_PATH="/env", TEST_AGENTARK_MOD_PATH="/mod")
            output = tmp_path / "rendered.json"
            result = _run(
                sys.executable,
                os.fspath(RENDERER),
                "--verl-root",
                os.fspath(checkout),
                "--num-envs",
                "4",
                "--output",
                os.fspath(output),
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rendered = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(rendered["warmup"]["num_envs"], 4)
            self.assertEqual(rendered["env_cfg"]["runtime_sandbox"]["pool_size"], 3)
            self.assertTrue(
                rendered["_agentark_verl_bridge"]["sandbox_auto_expands_for_worker_index"]
            )


if __name__ == "__main__":
    unittest.main()
