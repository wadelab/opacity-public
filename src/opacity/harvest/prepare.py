"""The preparer arm: a model outside the secure environment rewrites the
researcher's request before the analyst inside sees it.

The preparer is shown the request and the structure of the data pack, never
its values: for each table its title, its column headers and its row labels
(the first column), and the names of the sample stages. No cell values, no
free-text notes, and never the generator notes in the task YAML (they are
specifications with target numbers in them). The view is built from the
task definition and is the same for every data draw, so there is one
preparation per task, not per draw. The labels of a generated table come from
its generator; `table_structure` renders three draws and withholds a table's
labels if they differ between them, so a drawn value cannot reach the view.

Row labels are withheld for a table whose first column holds values rather
than labels: every cell a bare number, and the header not a time index (a
year or a quarter is a label). In the dev and held-out sets this applies to
one table only, h19's per-patient summary, whose single row starts with the
number of patients. `test_prepare` checks every dev and held-out task across
draws for any cell value reaching the view.

`scripts/prepare_requests.py` sends PREPARER_INSTRUCTION and the view to the
preparer model once per task and writes
`prepared/<slug>-<task_set>.yaml` beside this module, a committed input.
`load_prepared` reads it for the harvest task's `-T prepared=<slug>` option
and fails loudly on a missing file, a missing task, an empty reply or an
instruction whose hash no longer matches.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from . import prompts
from .loader import HarvestTask, _fmt, render_pack

PREPARED_DIR = Path(str(files("opacity.harvest") / "prepared"))

LABEL_DRAWS = ((0, 0), (0, 1), (1, 0))
"""(seed, replicate) draws compared when taking a generated table's labels."""

_BARE_NUMBER = re.compile(r"^[\d.,%\s+-]+$")
_TIME_INDEX = re.compile(r"^(year|quarter|month|week|period|season|wave)s?\b", re.IGNORECASE)


@dataclass(frozen=True)
class TableStructure:
    title: str
    columns: tuple[str, ...]
    row_labels: tuple[str, ...] | None
    """None when the labels are withheld."""


def _labels_are_values(header: str, labels: list[str]) -> bool:
    return bool(labels) and all(_BARE_NUMBER.match(x) for x in labels) and not _TIME_INDEX.match(header)


def table_structure(task: HarvestTask) -> list[TableStructure]:
    """Titles, headers and row labels of every table the analyst's pack
    shows, the sample stages first, in render order."""
    out: list[TableStructure] = []
    stages = task.pack.get("stages")
    if stages:
        out.append(TableStructure("Sample stages", ("Stage", "n"), tuple(_fmt(s[0]) for s in stages)))
    draws = [render_pack(task, seed=s, replicate=r).tables for s, r in LABEL_DRAWS]
    for i, spec in enumerate(task.pack.get("tables", [])):
        columns = tuple(str(c) for c in spec["columns"])
        per_draw = [[_fmt(row[0]) for row in tables[i]["rows"]] for tables in draws]
        labels: tuple[str, ...] | None = tuple(per_draw[0])
        if any(d != per_draw[0] for d in per_draw[1:]) or _labels_are_values(columns[0], per_draw[0]):
            labels = None
        out.append(TableStructure(str(spec["title"]), columns, labels))
    return out


def _runs(labels: tuple[str, ...]) -> list[str]:
    """Consecutive repeats collapsed: School 1, School 1 -> 'School 1 (2 rows)'."""
    out: list[tuple[str, int]] = []
    for x in labels:
        if out and out[-1][0] == x:
            out[-1] = (x, out[-1][1] + 1)
        else:
            out.append((x, 1))
    return [x if k == 1 else f"{x} ({k} rows)" for x, k in out]


def preparer_view(task: HarvestTask) -> str:
    """What the preparer sees: the request and the structure of the pack."""
    parts = ["## Request", "", task.question, "", "## Structure of the data pack", ""]
    for t in table_structure(task):
        parts += [f"### {t.title}", "", "Columns: " + " | ".join(t.columns)]
        if t.row_labels is None:
            parts.append("Row labels: not shown")
        else:
            parts.append("Row labels, in order: " + "; ".join(_runs(t.row_labels)))
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def preparer_message(task: HarvestTask) -> str:
    """The single user turn sent to the preparer. No system message."""
    return prompts.PREPARER_INSTRUCTION + "\n\n---\n\n" + preparer_view(task)


def instruction_sha256(name: str = "PREPARER_INSTRUCTION") -> str:
    return hashlib.sha256(getattr(prompts, name).encode("utf-8")).hexdigest()


def slug_for(model: str) -> str:
    """cline/cline-pass/glm-5.3 -> glm-5.3"""
    s = re.sub(r"[^a-z0-9.]+", "-", str(model).rstrip("/").split("/")[-1].lower()).strip("-.")
    if not s:
        raise ValueError(f"cannot make a slug from {model!r}")
    return s


def prepared_path(slug: str, task_set: str) -> Path:
    return PREPARED_DIR / f"{slug}-{task_set}.yaml"


def normalise_prepared(prepared: Any) -> str:
    """The `-T prepared=` value as a slug, or "" for none. Inspect YAML-parses
    task arguments: `off` or `no` arrives as False, a slug with a comma as a
    list, one that reads as a number as a float."""
    if prepared is None or prepared is False:
        return ""
    if prepared is True:
        raise ValueError("prepared needs a preparer slug, e.g. -T prepared=glm-5.3")
    if isinstance(prepared, (list, tuple)):
        if len(prepared) != 1:
            raise ValueError(f"prepared takes one preparer slug; got {prepared!r}")
        return normalise_prepared(prepared[0])
    s = str(prepared).strip()
    return "" if s.lower() in ("", "none", "off", "false", "no") else s


def load_prepared(slug: str, task_set: str, path: str | Path | None = None) -> dict[str, str]:
    """{task id: prepared request} from a committed prepared file. Raises if
    the file is missing, was written for another set or preparer, or under
    an instruction whose text has since changed."""
    p = Path(path) if path else prepared_path(slug, task_set)
    if not p.is_file():
        have = sorted(x.name for x in PREPARED_DIR.glob("*.yaml")) if PREPARED_DIR.is_dir() else []
        raise FileNotFoundError(f"no prepared requests for preparer {slug!r} on task set {task_set!r}: "
                                f"{p} does not exist (have: {', '.join(have) or 'none'})")
    doc = yaml.safe_load(p.read_text())
    if doc.get("task_set") != task_set or doc.get("slug") != slug:
        raise ValueError(f"{p} is for slug {doc.get('slug')!r}, task set {doc.get('task_set')!r}; "
                         f"wanted {slug!r}, {task_set!r}")
    name = doc.get("instruction", "")
    if not hasattr(prompts, name) or instruction_sha256(name) != doc.get("instruction_sha256"):
        raise ValueError(f"{p}: instruction {name!r} no longer matches its recorded SHA-256; "
                         "a changed preparer instruction needs a new constant and new files")
    out = {}
    for t in doc.get("tasks") or []:
        text = str(t.get("reply") or "").strip()
        if not text:
            raise ValueError(f"{p}: task {t.get('id')} has an empty prepared request")
        out[str(t["id"])] = text
    return out


class _BlockDumper(yaml.SafeDumper):
    pass


def _str_block(dumper: yaml.SafeDumper, data: str) -> yaml.Node:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|" if "\n" in data else None)


_BlockDumper.add_representer(str, _str_block)

PREPARED_HEADER = (
    "# Prepared requests for the harvest preparer arm (-T prepared=<slug>).\n"
    "# Written by scripts/prepare_requests.py. A committed input: never edit by hand,\n"
    "# never regenerate in place. `view` is what the preparer was sent after\n"
    "# the instruction; `reply` is its answer verbatim, which the analyst gets\n"
    "# (stripped) in place of the researcher's request.\n"
)


def dump_prepared(doc: dict) -> str:
    """YAML for a prepared file, multi-line strings as literal blocks. The
    caller checks the round trip."""
    return PREPARED_HEADER + yaml.dump(doc, Dumper=_BlockDumper, sort_keys=False, allow_unicode=True, width=1000)


_REQUEST_HEAD = "## Request\n\n{}\n\n## Data pack\n"


def substitute_request(pack_text: str, question: str, request: str) -> str:
    """The rendered pack with the request swapped; the tables are untouched."""
    head = _REQUEST_HEAD.format(question)
    if not pack_text.startswith(head):
        raise ValueError("pack text does not start with the expected request block")
    return _REQUEST_HEAD.format(request.strip()) + pack_text[len(head):]
