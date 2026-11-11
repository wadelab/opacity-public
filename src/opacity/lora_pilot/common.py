"""Small, CPU-only identity, configuration and append-only I/O helpers."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from . import MODEL, REVISION, SCHEMA


class Refusal(RuntimeError):
    """A failed instrument gate, rather than a model result."""


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise Refusal(reason)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path: Path) -> Any:
    with path.open() as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: a partial file is evidence, never a success marker.
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_event(path: Path, value: Any) -> None:
    with path.open("a") as stream:
        stream.write(canonical(value).decode() + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def safe_name(value: str) -> str:
    require(bool(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", value)), "invalid run ID")
    return value


RECIPE = {
    "rank": 8, "alpha": 16, "dropout": 0.05,
    "learning_rate": 5e-5, "weight_decay": 0.0,
    "betas": [0.9, 0.999], "epsilon": 1e-8,
    "clip_norm": 1.0, "accumulation": 4, "seed": 271028,
    "max_tokens": 512, "max_new_tokens": 128,
    "memory_fraction": 0.85, "profile_lengths": [512, 1024, 2048],
    "logit_abs_floor": 1e-5, "logit_rel_floor": 1e-6,
    "output_limit_bytes": 8589934592,
}

# Lengths for the "profile" stage. The study's own training examples (the 27B's
# rules-prompt reports, scripts/study_lengths.py, 27 October) run to 3,927 tokens,
# and 20% exceed the pilot's longest profile, 2,048. Not part of the v1 recipe.
STUDY_PROFILE_LENGTHS = [2048, 3072, 4096]


def config(path: Path) -> dict:
    import yaml

    value = yaml.safe_load(path.read_text())
    require(isinstance(value, dict), "config must be a mapping")
    require(set(value) == {"schema", "model", "revision", "recipe", "site"}, "unknown/missing config keys")
    require(value["schema"] == SCHEMA and value["model"] == MODEL and value["revision"] == REVISION,
            "unregistered model, revision or schema")
    require(value["recipe"] == RECIPE, "recipe differs from honest pilot v1; register a new version")
    site = value["site"]
    expected = {"snapshot", "probe_record", "revision_record", "runtime_image", "runtime_sha256",
                "runtime_lock", "study_root", "backup_root", "egress_ip", "egress_port", "egress_dns"}
    require(isinstance(site, dict) and set(site) == expected, "unknown/missing site settings")
    require(all(v is not None and v != "" for v in site.values()), "fill every site setting before preparation")
    require(bool(re.fullmatch(r"[0-9a-f]{64}", str(site["runtime_sha256"]))), "invalid runtime image digest")
    require(isinstance(site["egress_port"], int) and 0 < site["egress_port"] < 65536, "invalid probe port")
    # The two network targets must be infrastructure-owner approved; no default public probe.
    import ipaddress
    ipaddress.ip_address(site["egress_ip"])
    for key in expected - {"runtime_sha256", "egress_ip", "egress_port", "egress_dns"}:
        require(Path(site[key]).is_absolute(), f"site.{key} must be absolute")
        require(not any(c in site[key] for c in ":,\n"), f"unsupported bind-path character: {key}")
    return value


def tree_manifest(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"symlink in frozen tree: {path.name}")
        if path.is_file():
            result[str(path.relative_to(root))] = digest(path)
    return result


def verify_tree(root: Path, expected: dict[str, str], *, exclude: tuple[str, ...] = ()) -> None:
    actual = {name: sha for name, sha in tree_manifest(root).items() if name not in exclude}
    require(actual == expected, "frozen tree differs from its manifest")


def bundle_manifest(bundle: Path, *, verify: bool = True) -> dict:
    manifest = read_json(bundle / "manifest.json")
    require(manifest["schema"] == SCHEMA, "wrong bundle schema")
    require(manifest["identity"] == identity(manifest["files"]), "invalid bundle identity")
    if verify:
        verify_tree(bundle, manifest["files"], exclude=("manifest.json",))
    return manifest


def predecessor(path: Path, bundle_id: str, stage: str) -> dict:
    result = read_json(path / "result.json")
    require(result.get("schema") == SCHEMA and result.get("stage") == stage, "wrong predecessor stage")
    require(result.get("bundle_id") == bundle_id and result.get("status") == "passed",
            "failed, incomplete or different-bundle predecessor")
    artifacts = result.get("artifacts")
    require(isinstance(artifacts, dict) and {"checks.json", "preflight.json", "events.jsonl"} <= set(artifacts),
            "predecessor evidence incomplete")
    require(artifacts == tree_manifest_subset(path, artifacts),
            "predecessor artifact changed")
    return result


def tree_manifest_subset(root: Path, expected: dict[str, str]) -> dict[str, str]:
    result = {}
    for name in expected:
        path = root / name
        require(path.resolve().is_relative_to(root.resolve()) and path.is_file() and not path.is_symlink(),
                "invalid artifact path")
        result[name] = digest(path)
    return result
