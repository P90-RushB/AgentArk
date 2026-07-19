#!/usr/bin/env python3
"""Offline compatibility check for the external AgentArk VERL recipe.

The checker never imports or executes code from the checkout and never fetches
or changes Git state.  It verifies the reviewed history baseline, required
files, and the HTTP v1 contract literals used by the current integration.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import urlparse


EXIT_COMPATIBLE = 0
EXIT_INCOMPATIBLE = 1
EXIT_INDETERMINATE = 2
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass
class Result:
    checkout: str = ""
    head: str = ""
    branch: str | None = None
    minimum_compatible_commit: str = ""
    protocol_version: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    indeterminate: list[str] = field(default_factory=list)

    def exit_code(self, *, strict: bool) -> int:
        if self.errors or (strict and self.warnings):
            return EXIT_INCOMPATIBLE
        if self.indeterminate:
            return EXIT_INDETERMINATE
        return EXIT_COMPATIBLE

    def as_dict(self, *, strict: bool) -> dict[str, Any]:
        code = self.exit_code(strict=strict)
        if code == EXIT_COMPATIBLE:
            status = "compatible"
        elif code == EXIT_INDETERMINATE:
            status = "indeterminate"
        else:
            status = "incompatible"
        return {
            "status": status,
            "ok": code == EXIT_COMPATIBLE,
            "strict": strict,
            "checkout": self.checkout,
            "head": self.head,
            "branch": self.branch,
            "minimum_compatible_commit": self.minimum_compatible_commit,
            "protocol_version": self.protocol_version,
            "errors": self.errors,
            "warnings": self.warnings,
            "indeterminate": self.indeterminate,
            "info": self.info,
        }


def _run_git(checkout: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", os.fspath(checkout), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _safe_relative_path(raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw:
        return None
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        return None
    return path.as_posix()


def _normalise_remote(url: str) -> str:
    value = url.strip()
    if value.startswith("git@") and ":" in value:
        host, path = value[4:].split(":", 1)
        value = f"{host}/{path}"
    elif "://" in value:
        parsed = urlparse(value)
        value = f"{parsed.hostname or ''}{parsed.path}"
    value = value.rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    return value.lower()


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    return data


def _validate_required_file(checkout: Path, rel: str, result: Result) -> None:
    tree = _run_git(checkout, "ls-tree", "HEAD", "--", rel)
    if tree.returncode != 0 or not tree.stdout.strip():
        result.errors.append(f"required path is not tracked at HEAD: {rel}")
        return
    first = tree.stdout.splitlines()[0]
    metadata = first.split("\t", 1)[0].split()
    if len(metadata) < 2 or metadata[1] != "blob" or metadata[0] == "120000":
        result.errors.append(f"required path must be a tracked regular file: {rel}")
        return

    candidate = checkout / rel
    try:
        if candidate.is_symlink() or not candidate.is_file():
            result.errors.append(f"required working-tree file is missing or unsafe: {rel}")
            return
        candidate.resolve(strict=True).relative_to(checkout.resolve(strict=True))
    except (OSError, ValueError):
        result.errors.append(f"required path escapes the checkout: {rel}")


def _changed_paths(checkout: Path, baseline: str, paths: Iterable[str]) -> list[str]:
    unique_paths = sorted(set(paths))
    if not unique_paths:
        return []
    proc = _run_git(checkout, "diff", "--name-only", f"{baseline}..HEAD", "--", *unique_paths)
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line]


def check(checkout_arg: Path, manifest: dict[str, Any]) -> Result:
    result = Result()
    repository = manifest.get("repository")
    protocol = manifest.get("protocol")
    if not isinstance(repository, dict) or not isinstance(protocol, dict):
        result.errors.append("manifest must contain repository and protocol mappings")
        return result

    baseline = repository.get("minimum_compatible_commit")
    if not isinstance(baseline, str) or not _COMMIT_RE.fullmatch(baseline):
        result.errors.append("minimum_compatible_commit must be a full 40-character SHA")
        return result
    result.minimum_compatible_commit = baseline.lower()
    result.protocol_version = str(protocol.get("version", ""))

    checkout_arg = checkout_arg.expanduser().resolve()
    top = _run_git(checkout_arg, "rev-parse", "--show-toplevel")
    if top.returncode != 0:
        result.errors.append(f"not a Git checkout: {checkout_arg}")
        return result
    checkout = Path(top.stdout.strip()).resolve()
    result.checkout = os.fspath(checkout)

    head = _run_git(checkout, "rev-parse", "HEAD")
    if head.returncode != 0 or not _COMMIT_RE.fullmatch(head.stdout.strip()):
        result.errors.append("could not resolve checkout HEAD")
        return result
    result.head = head.stdout.strip().lower()

    branch = _run_git(checkout, "symbolic-ref", "--quiet", "--short", "HEAD")
    result.branch = branch.stdout.strip() if branch.returncode == 0 else None
    recommended_branch = repository.get("recommended_branch")
    if result.branch is None:
        result.warnings.append("HEAD is detached; provenance is valid but no branch name can be checked")
    elif isinstance(recommended_branch, str) and result.branch != recommended_branch:
        result.warnings.append(
            f"current branch is {result.branch!r}; recommended branch is {recommended_branch!r}"
        )

    baseline_object = _run_git(checkout, "cat-file", "-e", f"{baseline}^{{commit}}")
    ancestry_known = baseline_object.returncode == 0
    if not ancestry_known:
        result.indeterminate.append(
            "the reviewed baseline commit is unavailable locally; fetch the agentark_rl branch "
            "or unshallow the checkout, then run this check again"
        )
    else:
        ancestor = _run_git(checkout, "merge-base", "--is-ancestor", baseline, "HEAD")
        if ancestor.returncode == 1:
            result.errors.append(
                f"reviewed baseline {baseline} is not an ancestor of HEAD {result.head}"
            )
        elif ancestor.returncode != 0:
            result.indeterminate.append("Git could not determine baseline ancestry")
        else:
            result.info.append("reviewed baseline is an ancestor of HEAD")

    required_raw = manifest.get("required_paths")
    if not isinstance(required_raw, list) or not required_raw:
        result.errors.append("manifest required_paths must be a non-empty list")
        required_paths: list[str] = []
    else:
        required_paths = []
        for raw in required_raw:
            rel = _safe_relative_path(raw)
            if rel is None:
                result.errors.append(f"unsafe required path in manifest: {raw!r}")
                continue
            required_paths.append(rel)
            _validate_required_file(checkout, rel, result)

    marker_specs = protocol.get("required_literals", [])
    if not isinstance(marker_specs, list):
        result.errors.append("protocol.required_literals must be a list")
    else:
        for spec in marker_specs:
            if not isinstance(spec, dict):
                result.errors.append("each required_literals entry must be a mapping")
                continue
            rel = _safe_relative_path(spec.get("path"))
            literals = spec.get("literals")
            if rel is None or not isinstance(literals, list) or not all(
                isinstance(item, str) and item for item in literals
            ):
                result.errors.append(f"invalid required_literals entry: {spec!r}")
                continue
            candidate = checkout / rel
            try:
                if candidate.is_symlink() or not candidate.is_file():
                    raise ValueError("not a regular working-tree file")
                candidate.resolve(strict=True).relative_to(checkout.resolve(strict=True))
                contents = candidate.read_text(encoding="utf-8")
            except (OSError, UnicodeError, ValueError) as exc:
                result.errors.append(f"cannot safely read protocol contract file {rel}: {exc}")
                continue
            for literal in literals:
                if literal not in contents:
                    result.errors.append(f"protocol contract literal missing from {rel}: {literal!r}")

    canonical_slug = repository.get("canonical_slug")
    remote_names_proc = _run_git(checkout, "remote")
    remote_names = [line.strip() for line in remote_names_proc.stdout.splitlines() if line.strip()]
    remote_urls: list[str] = []
    for remote_name in remote_names:
        remote_proc = _run_git(checkout, "remote", "get-url", "--all", remote_name)
        remote_urls.extend(line.strip() for line in remote_proc.stdout.splitlines() if line.strip())
    if not remote_urls:
        result.warnings.append("Git remotes are absent; canonical repository provenance was not checked")
    elif isinstance(canonical_slug, str) and canonical_slug.lower() not in {
        _normalise_remote(url) for url in remote_urls
    }:
        result.warnings.append(
            f"no Git remote matches canonical repository {repository.get('canonical_url')!r}"
        )

    watch_raw = manifest.get("watch_paths", [])
    watch_paths = [
        rel for raw in watch_raw if (rel := _safe_relative_path(raw)) is not None
    ] if isinstance(watch_raw, list) else []
    review_paths = sorted(set(required_paths + watch_paths))
    if ancestry_known and result.head != baseline.lower():
        changed = _changed_paths(checkout, baseline, review_paths)
        if changed:
            result.warnings.append(
                "review-sensitive paths changed after the baseline; run the real Unity smoke: "
                + ", ".join(changed)
            )

    dirty = _run_git(checkout, "status", "--porcelain", "--untracked-files=no", "--", *review_paths)
    if dirty.returncode == 0 and dirty.stdout.strip():
        changed = sorted({line[3:] for line in dirty.stdout.splitlines() if len(line) > 3})
        result.warnings.append(
            "review-sensitive working-tree changes are present: " + ", ".join(changed)
        )

    result.info.append(
        f"verified {len(required_paths)} required files and AgentArk HTTP {result.protocol_version} literals"
    )
    return result


def _print_text(payload: dict[str, Any]) -> None:
    status = str(payload["status"]).upper()
    print(f"[{status}] AgentArk VERL checkout: {payload['checkout'] or '(unresolved)'}")
    if payload.get("head"):
        print(f"  HEAD: {payload['head']} ({payload.get('branch') or 'detached'})")
    for label, key in (
        ("ERROR", "errors"),
        ("WARN", "warnings"),
        ("UNKNOWN", "indeterminate"),
        ("INFO", "info"),
    ):
        for message in payload[key]:
            print(f"  [{label}] {message}")


def _build_parser() -> argparse.ArgumentParser:
    default_manifest = Path(__file__).with_name("compatibility.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkout",
        type=Path,
        default=Path(os.environ["VERL_ROOT"]) if os.environ.get("VERL_ROOT") else None,
        help="VERL checkout root (or set VERL_ROOT)",
    )
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat provenance, branch, dirty-tree, and review-path warnings as incompatible",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.checkout is None:
        parser.error("--checkout is required when VERL_ROOT is not set")
    try:
        manifest = _load_manifest(args.manifest.expanduser().resolve())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ERROR] cannot load compatibility manifest: {exc}", file=sys.stderr)
        return EXIT_INDETERMINATE

    result = check(args.checkout, manifest)
    payload = result.as_dict(strict=args.strict)
    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_text(payload)
    return result.exit_code(strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
