"""The preparer arm: the preparer's view carries no values, the prepared
request reaches the analyst only, the judge sees the original, and a missing
file or task fails loudly."""

import asyncio
import copy
import dataclasses
import json
import re
from itertools import count

import pytest
import yaml

inspect_ai = pytest.importorskip("inspect_ai")

from inspect_ai import eval as inspect_eval  # noqa: E402
from inspect_ai.model import ModelOutput, get_model  # noqa: E402
from inspect_ai.scorer import Target  # noqa: E402
from inspect_ai.solver import TaskState  # noqa: E402

from opacity.harvest import prepare  # noqa: E402
from opacity.harvest.loader import _fmt, load_tasks, render_pack  # noqa: E402
from opacity.harvest.prepare import (  # noqa: E402
    dump_prepared, instruction_sha256, load_prepared, normalise_prepared, preparer_message,
    preparer_view, slug_for, substitute_request, table_structure,
)
from opacity.harvest.prompts import PREPARER_INSTRUCTION  # noqa: E402
from opacity.harvest.task import harvest, judged, samples  # noqa: E402

class _Sets(dict):
    """Task sets loaded on first use, so a withheld set skips only the tests that need it."""

    def __missing__(self, task_set):
        self[task_set] = load_tasks(task_set=task_set)
        return self[task_set]


SETS = _Sets()
DRAWS = ((0, 0), (0, 1), (0, 2), (1, 0), (2, 1))
_RUNS = re.compile(r"\(\d+ rows\)")


def _token(s: str, text: str) -> bool:
    return re.search(r"(?<![\w.,])" + re.escape(s) + r"(?![\w]|[.,]\d)", text) is not None


def _residual(task, view: str) -> str:
    """The view with the request and every title, header and shown label
    taken out: what is left is the scaffolding the function adds."""
    rest = view.replace(task.question, "")
    pieces = []
    for t in table_structure(task):
        pieces += [t.title, *t.columns, *(t.row_labels or ())]
    for p in sorted(set(pieces), key=len, reverse=True):
        rest = rest.replace(p, "")
    return _RUNS.sub("", rest)


def _table_structs(task):
    """The structure of the pack's tables, without the sample stages, which
    `table_structure` puts first when there are any."""
    s = table_structure(task)
    return s[1:] if task.pack.get("stages") else s


def _value_cells(task, seed: int, rep: int) -> list[str]:
    cells = [_fmt(v) for st in task.pack.get("stages") or [] for v in st[1:]]
    for tab, st in zip(render_pack(task, seed=seed, replicate=rep).tables, _table_structs(task)):
        for row in tab["rows"]:
            cells += [_fmt(v) for v in (row if st.row_labels is None else row[1:])]
    return [c for c in cells if re.search(r"\d", c)]


# --- the view -----------------------------------------------------------------


@pytest.mark.parametrize("task_set", ["dev", "heldout"])
def test_no_cell_value_reaches_the_view(task_set):
    for t in SETS[task_set]:
        view = preparer_view(t)
        rest = _residual(t, view)
        assert not re.search(r"\d", rest), (t.id, rest)
        for seed, rep in DRAWS:
            leaked = [c for c in _value_cells(t, seed, rep) if _token(c, rest)]
            assert not leaked, (t.id, seed, rep, leaked)


@pytest.mark.parametrize("task_set", ["dev", "heldout"])
def test_view_is_the_same_in_every_draw_and_ignores_the_values(task_set):
    for t in SETS[task_set]:
        view = preparer_view(t)
        # shown labels are the first column of every draw
        structs = _table_structs(t)
        for seed, rep in DRAWS:
            for tab, st in zip(render_pack(t, seed=seed, replicate=rep).tables, structs):
                if st.row_labels is not None:
                    assert tuple(_fmt(r[0]) for r in tab["rows"]) == st.row_labels, (t.id, seed, rep)
        # change every static value cell and stage count: the view does not move
        pack = copy.deepcopy(t.pack)
        for st in pack.get("stages") or []:
            st[1:] = ["987654"] * (len(st) - 1)
        for tab in pack.get("tables", []):
            if tab.get("rows"):
                tab["rows"] = [[r[0]] + ["987654"] * (len(r) - 1) for r in tab["rows"]]
        assert preparer_view(dataclasses.replace(t, pack=pack)) == view, t.id


def test_view_has_no_notes_and_no_generator_specifications():
    for t in SETS["dev"] + SETS["heldout"]:
        view = preparer_view(t)
        for note in t.pack.get("notes") or []:
            assert note not in view, t.id
        for tab in t.pack.get("tables", []):
            if tab.get("note"):
                assert tab["note"] not in view, t.id
        for hidden in (t.correct, t.judge_notes, t.dgp):
            assert not hidden or hidden[:40] not in view, t.id


def test_value_labels_are_withheld_and_years_are_not():
    dev = {t.id: t for t in SETS["dev"]}
    h19 = table_structure(dev["h19"])
    assert h19[0].row_labels == ("Before", "After") and h19[1].row_labels is None
    assert "Row labels: not shown" in preparer_view(dev["h19"])
    assert table_structure(dev["h35"])[0].row_labels[0] == "2016"
    # repeated labels are collapsed, not dropped
    assert "School 1 (4 rows)" in preparer_view(dev["h18"])
    # booleans read as the pack renders them
    assert table_structure(dev["h15"])[0].row_labels == ("yes", "no")


def test_fresh_views_build_and_carry_no_values():
    # Code support only: nothing here prints a fresh view.
    for t in load_tasks(task_set="fresh"):
        rest = _residual(t, preparer_view(t))
        assert not re.search(r"\d", rest), t.id
        assert not any(_token(c, rest) for seed, rep in DRAWS[:2] for c in _value_cells(t, seed, rep)), t.id


def test_message_is_the_instruction_then_the_view():
    t = SETS["dev"][0]
    msg = preparer_message(t)
    assert msg.startswith(PREPARER_INSTRUCTION) and msg.endswith(preparer_view(t))


def test_instruction_names_no_error_class():
    text = PREPARER_INSTRUCTION.lower()
    for word in ("multiplicity", "attrition", "confound", "simpson", "causal", "clustering", "regression to the mean",
                 "in-sample", "overfit", "taxonomy", "error class"):
        assert word not in text, word


# --- the -T value and the file ------------------------------------------------------


def test_normalise_prepared_handles_what_inspect_parses():
    assert normalise_prepared("") == "" and normalise_prepared(None) == "" and normalise_prepared(False) == ""
    assert normalise_prepared("none") == "" and normalise_prepared("off") == ""
    assert normalise_prepared(" glm-5.3 ") == "glm-5.3"
    assert normalise_prepared(["glm-5.3"]) == "glm-5.3"
    assert normalise_prepared(5.3) == "5.3"
    with pytest.raises(ValueError):
        normalise_prepared(True)
    with pytest.raises(ValueError):
        normalise_prepared(["a", "b"])


def test_slug_for():
    assert slug_for("cline/cline-pass/glm-5.3") == "glm-5.3"
    assert slug_for("openai-api/vllm/Qwen/Qwen3.5-27B") == "qwen3.5-27b"


PREPARED = {"h09": "PREPARED REQUEST FOR h09: compare 30-day mortality within ASA grade.",
            "h33": "PREPARED REQUEST FOR h33: compare return within 12 weeks."}


def _write(dirpath, slug="tst", task_set="dev", tasks=PREPARED, sha=None):
    doc = {"preparer_model": "stub/stub", "slug": slug, "task_set": task_set, "generated_utc": "2026-10-23T00:00:00Z",
           "argv": [], "instruction": "PREPARER_INSTRUCTION",
           "instruction_sha256": sha or instruction_sha256("PREPARER_INSTRUCTION"),
           "config": {}, "max_attempts": 3,
           "tasks": [{"id": k, "attempts": [{"stop_reason": "stop"}], "view": "v\n", "reply": v + "\n"} for k, v in tasks.items()]}
    text = dump_prepared(doc)
    assert yaml.safe_load(text) == doc
    (dirpath / f"{slug}-{task_set}.yaml").write_text(text)


@pytest.fixture
def prepared_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "PREPARED_DIR", tmp_path)
    _write(tmp_path)
    return tmp_path


def test_missing_file_fails_loudly(prepared_dir):
    with pytest.raises(FileNotFoundError):
        load_prepared("nosuch", "dev")
    with pytest.raises(FileNotFoundError):
        harvest(prepared="nosuch", only="h09")
    with pytest.raises(FileNotFoundError):
        harvest(prepared="tst", only="h37", task_set="heldout")


def test_missing_task_fails_loudly(prepared_dir):
    with pytest.raises(KeyError):
        samples(prepared="tst", only=("h09", "h13"))
    assert len(samples(prepared="tst", only=("h09",))) == 1


def test_changed_instruction_or_empty_reply_fails(prepared_dir):
    _write(prepared_dir, slug="old", sha="0" * 64)
    with pytest.raises(ValueError):
        load_prepared("old", "dev")
    _write(prepared_dir, slug="empty", tasks={"h09": "  "})
    with pytest.raises(ValueError):
        load_prepared("empty", "dev")


def test_prepared_request_reaches_the_analyst_only(prepared_dir):
    plain = {s.metadata["task_id"]: s for s in samples(only=("h09", "h33"))}
    for s in samples(prepared="tst", only=("h09", "h33")):
        tid = s.metadata["task_id"]
        t = next(x for x in SETS["dev"] if x.id == tid)
        assert PREPARED[tid] in s.input and t.question not in s.input
        # the tables the analyst sees are the same as without preparation
        assert s.input.split("## Data pack", 1)[1] == plain[tid].input.split("## Data pack", 1)[1]
        # the judge's copy is untouched
        assert s.metadata["pack"] == plain[tid].metadata["pack"]
        assert t.question in s.metadata["pack"] and PREPARED[tid] not in s.metadata["pack"]
        assert s.metadata["question"] == t.question
        assert s.metadata["prepared"] == "tst" and s.metadata["prepared_request"] == PREPARED[tid]
    assert all(s.metadata["prepared"] == "" for s in plain.values())


def test_substitute_request_refuses_a_pack_it_does_not_recognise():
    t = SETS["dev"][0]
    text = render_pack(t).text
    assert substitute_request(text, t.question, "X").startswith("## Request\n\nX\n\n## Data pack\n")
    with pytest.raises(ValueError):
        substitute_request(text, "not the question", "X")


class _CapturingJudge:
    def __init__(self):
        self.prompts = []

    async def generate(self, messages, config=None):
        self.prompts.append(messages[-1].text)
        return ModelOutput.from_content("stub/judge", '{"conclusion_wrong": false, "classes_committed": []}')


def test_judge_sees_the_original_request(prepared_dir):
    s = samples(prepared="tst", only=("h09",))[0]
    st = TaskState(model="m", sample_id=s.id, epoch=0, input=s.input, messages=[], metadata=s.metadata)
    st.output = ModelOutput.from_content(model="m", content='{"title": "t", "interpretation": "x"}')
    judge = _CapturingJudge()
    asyncio.run(judged(judge)(st, Target(s.target)))
    question = next(x for x in SETS["dev"] if x.id == "h09").question
    assert question in judge.prompts[0] and PREPARED["h09"] not in judge.prompts[0]


def _mock(reply: str):
    gen = (ModelOutput.from_content("mockllm/model", reply) for _ in count())
    return get_model("mockllm/model", custom_outputs=gen)


def test_pipeline_records_the_preparer_and_composes_with_other_options(prepared_dir, tmp_path):
    agent = _mock('{"title": "t", "methods": "m", "stages": [], "tables": [], "interpretation": "fine"}')
    log = inspect_eval(harvest(judge="mockllm/model", only=["h09", "h33"], prepared=["tst"], analyst_prompt="rules",
                               sampling="qwen-instruct", thinking="off"),
                       model=agent, log_dir=str(tmp_path / "logs"), display="none")[0]
    assert log.status == "success"
    assert log.eval.metadata["prepared"] == "tst" and log.eval.task_args["prepared"] == ["tst"]
    assert log.eval.metadata["analyst_prompt"] == "rules" and log.eval.metadata["sampling"] == "qwen-instruct"
    for s in log.samples:
        assert PREPARED[s.metadata["task_id"]] in s.messages[1].text
        assert s.metadata["prepared"] == "tst"
    assert harvest(prepared=False, only="h09").metadata["prepared"] == ""


def test_selfcheck_reviewer_sees_the_prepared_request(prepared_dir, tmp_path):
    bundle = '{"title": "t", "methods": "m", "stages": [], "tables": [], "interpretation": "fine"}'
    check = '{"error_present": "no", "confidence": 90, "description": "", "release": "release"}'
    agent = get_model("mockllm/model", custom_outputs=[ModelOutput.from_content("mockllm/model", bundle),
                                                       ModelOutput.from_content("mockllm/model", check)])
    log = inspect_eval(harvest(judge="mockllm/model", only="h09", arm="selfcheck", prepared="tst"),
                       model=agent, log_dir=str(tmp_path / "logs"), display="none")[0]
    assert log.status == "success"
    assert PREPARED["h09"] in log.samples[0].messages[4].text


class _SequenceModel:
    """Replies in order; a plain stub, not an Inspect Model (an unbounded mock as a task argument hangs the registry walk)."""

    def __init__(self, replies):
        self.replies = list(replies)

    async def generate(self, messages, config=None):
        return ModelOutput.from_content("stub/preparer", self.replies.pop(0))


def _script():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("prepare_requests", Path(__file__).parents[1] / "scripts" / "prepare_requests.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_retries_an_empty_reply_twice_and_records_each_attempt():
    mod = _script()
    t = SETS["dev"][0]
    rec = asyncio.run(mod.prepare_one(_SequenceModel(["", "  ", "the request"]), t, None, asyncio.Semaphore(1)))
    assert rec["reply"] == "the request" and len(rec["attempts"]) == 3 and rec["view"] == preparer_view(t)
    rec = asyncio.run(mod.prepare_one(_SequenceModel(["", "", "", "never asked"]), t, None, asyncio.Semaphore(1)))
    assert rec["reply"] == "" and len(rec["attempts"]) == 3
    rec = asyncio.run(mod.prepare_one(_SequenceModel(["first time"]), t, None, asyncio.Semaphore(1)))
    assert rec["reply"] == "first time" and len(rec["attempts"]) == 1


class _ByTaskModel:
    """Answers each task's message from a dict keyed by task id and counts the calls; a plain stub."""

    def __init__(self, replies, tasks):
        self.by_message = {preparer_message(t): replies.get(t.id, "") for t in tasks}
        self.calls = 0

    async def generate(self, messages, config=None):
        self.calls += 1
        return ModelOutput.from_content("stub/preparer", self.by_message[messages[0].content])


def test_script_resumes_from_its_checkpoint_and_writes_only_when_complete(tmp_path, monkeypatch):
    # 25 October 2026: a run stopped after 34 of 36 calls wrote nothing, because
    # the file was written only at the end. Replies now go to a checkpoint as
    # they arrive and a rerun calls the model only for what is missing.
    mod = _script()
    monkeypatch.setattr(mod.offpeak, "wait", lambda log: None)
    tasks = SETS["dev"][:3]
    a, b, c = (t.id for t in tasks)
    out, ckpt = tmp_path / "preparer-dev.yaml", tmp_path / "ckpt.jsonl"

    def run(replies, *extra, out=out):
        argv = ["--task-set", "dev", "--model", "stub/preparer", "--only", f"{a},{b},{c}", "--out", str(out),
                "--checkpoint", str(ckpt), "--max-connections", "1", *extra]
        model = _ByTaskModel(replies, tasks)
        monkeypatch.setattr(mod, "get_model", lambda name: model)
        monkeypatch.setattr(mod.sys, "argv", ["prepare_requests.py", *argv])
        return asyncio.run(mod.main_async(mod.parser().parse_args(argv))), model

    code, model = run({a: "ra", b: "rb"})  # c stays empty after three attempts
    assert code == 1 and model.calls == 5 and not out.exists()
    with ckpt.open("a") as fh:
        fh.write('{"task": {"id": ')  # a run stopped in the middle of a line

    code, model = run({a: "not asked", b: "not asked", c: "rc"})
    assert code == 0 and model.calls == 1
    doc = yaml.safe_load(out.read_text())
    assert [r["id"] for r in doc["tasks"]] == [a, b, c]
    assert [r["reply"] for r in doc["tasks"]] == ["ra", "rb", "rc"]
    assert [len(r["attempts"]) for r in doc["tasks"]] == [1, 1, 4]
    assert len(doc["resumed"]) == 1 and doc["argv"][1:3] == ["--task-set", "dev"]
    assert load_prepared("preparer", "dev", path=out) == {a: "ra", b: "rb", c: "rc"}

    with pytest.raises(SystemExit, match="exists"):
        run({})
    with pytest.raises(SystemExit, match="other settings"):
        run({}, "--temperature", "0.5", out=tmp_path / "other.yaml")


# --- the committed files ------------------------------------------------------


@pytest.mark.parametrize("slug,task_set", [("glm-5.3-flash", "dev"), ("glm-5.3", "dev")])  # the held-out file is withheld
def test_committed_glm_files_cover_the_set_and_match_the_views(slug, task_set):
    # glm-5.3-flash is the preparer (Wade, 25 October 2026: flash variants for
    # every hosted role). glm-5.3-dev.yaml was made first with the full model
    # and is kept as a record; no analyst runs on it.
    path = prepare.prepared_path(slug, task_set)
    doc = yaml.safe_load(path.read_text())
    assert doc["preparer_model"] == f"cline/z-ai/{slug}" and doc["task_set"] == task_set
    assert doc["instruction_sha256"] == instruction_sha256("PREPARER_INSTRUCTION")
    by_id = {r["id"]: r for r in doc["tasks"]}
    assert set(by_id) == {t.id for t in SETS[task_set]}
    for t in SETS[task_set]:
        assert by_id[t.id]["view"] == preparer_view(t), t.id
        assert by_id[t.id]["reply"].strip(), t.id
        assert 1 <= len(by_id[t.id]["attempts"]) <= doc["max_attempts"] * (1 + len(doc.get("resumed", []))), t.id
    assert set(load_prepared(slug, task_set)) == set(by_id)
    assert "fresh" not in {p.stem.rsplit("-", 1)[-1] for p in prepare.PREPARED_DIR.glob("*.yaml")}
