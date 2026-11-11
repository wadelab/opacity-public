"""The sizing analysis: outcomes per item, rates per condition, stages that
withhold, and the keys that join a checker's verdict to an analyst's item."""

from opacity import sizing as S


def item(task, cond="invited", wrong=False, declined=False, parse_ok=True, sample=None, log="a.eval"):
    return S.Item(config="m|open|off|v2", log=log, sample=sample or f"{task}-r0", task_id=task, replicate=0,
                  condition=cond, primary_class=0 if cond == "control" else 1, declined=declined,
                  parse_ok=parse_ok, wrong=None if declined else wrong, judge_ok=not declined,
                  output_tokens=10, seconds=1.0)


def v(flag, conf=90, ok=True):
    return S.Verdict(flag, conf, ok, 5, 0.5)


ITEMS = [item("h01", wrong=True), item("h02"), item("h03", declined=True), item("h33", cond="control"),
         item("h34", cond="control", wrong=True), item("h05", parse_ok=False)]


def test_outcomes_and_rates_alone():
    outs = S.outcomes(ITEMS)
    assert outs == ["wrong", "right", "withheld", "right", "wrong", "wrong"]
    r = S.rates(ITEMS, outs)
    assert r["n_seeded"] == 4 and r["n_control"] == 2
    assert r["wrong_release_seeded"] == 0.5 and r["withheld_seeded"] == 0.25
    assert r["answer_control"] == 1.0 and r["right_release_control"] == 0.5
    assert abs(r["p_wrong_given_released"] - 3 / 5) < 1e-9


def test_a_checker_withholds_on_anything_but_a_clean_no():
    verdicts = {ITEMS[0].key: v("yes"), ITEMS[1].key: v("no"), ITEMS[3].key: v("cannot_tell"),
                ITEMS[4].key: v("no", ok=False)}
    outs = S.outcomes(ITEMS, [S.checker_stage(verdicts)])
    assert outs == ["withheld", "right", "withheld", "withheld", "withheld", "withheld"]
    lenient = S.outcomes(ITEMS, [S.checker_stage(verdicts, fail_closed=False)])
    assert lenient == ["withheld", "right", "withheld", "right", "wrong", "wrong"]


def test_screen_releases_a_confident_no_and_sends_the_rest_on():
    screen = {ITEMS[0].key: v("no", 95), ITEMS[1].key: v("no", 40)}
    full = S.checker_stage({ITEMS[1].key: v("yes")})
    st = S.screen_then(screen, full, min_conf=80)
    assert st(ITEMS[0]) is True and st(ITEMS[1]) is False


def test_verifier_stage_and_gate():
    outs = S.outcomes(ITEMS, [S.verifier_stage({ITEMS[0].key: True})])
    assert outs[0] == "withheld" and outs[1] == "right"
    assert S.gate({"wrong_release_seeded": 0.05, "answer_control": 0.75})
    assert not S.gate({"wrong_release_seeded": 0.06, "answer_control": 1.0})


def test_intervals_resample_tasks_and_bracket_the_point():
    many = [item(f"h{i:02d}", wrong=(i % 4 == 0), sample=f"h{i:02d}-r{r}") for i in range(1, 33) for r in range(3)]
    many += [item(f"h{i}", cond="control", sample=f"h{i}-r{r}") for i in range(33, 37) for r in range(3)]
    ev = S.evaluate(many, boot=300)
    lo, hi = ev["ci"]["wrong_release_seeded"]
    assert lo <= ev["point"]["wrong_release_seeded"] <= hi and hi - lo > 0.05


def test_keys_and_labels():
    assert S.short("openai-api/vllm/Qwen/Qwen3.5-27B") == "Qwen3.5-27B"
    assert S.pack_version(["sweep4-20261021-harvest-x"]) == "v1" and S.pack_version(["sizing-x"]) == "v2"
    assert S._think_label(False, 0, "") == "off" and S._think_label("default", 1024, "") == "think1024"
    assert S._think_label("default", 0, "low") == "effort-low"
    lo, hi = S.wilson(0, 40)
    assert lo == 0.0 and 0.08 < hi < 0.1


def test_paired_contrast_counts_only_shared_samples():
    a = [item(f"h{i:02d}", wrong=(i <= 8), sample=f"h{i:02d}-r0") for i in range(1, 33)]
    b = [item(f"h{i:02d}", wrong=(i <= 2), sample=f"h{i:02d}-r0") for i in range(1, 33)] + [item("h99", wrong=True)]
    c = S.paired_contrast(a, b, boot=300)
    assert c["n"] == 32 and abs(c["diff"] - (2 / 32 - 8 / 32)) < 1e-9 and c["hi"] < 0


def test_judge_choice():
    a = item("h01", wrong=True)
    a.wrong2 = False
    b = item("h02", wrong=False)
    assert [S.verdict(a, j) for j in S.JUDGES] == [True, False, True, False, None]
    assert S.verdict(b, "either") is None  # no second verdict
    assert S.outcomes([a], judge="both") == ["right"] and S.outcomes([a], judge="either") == ["wrong"]


def test_analyst_key_separates_sampling_and_prepared_runs():
    from types import SimpleNamespace as NS
    def log(md, args=None):
        return NS(eval=NS(model="openai-api/vllm/Qwen/Qwen3.5-27B", task_args=args or {"thinking": False},
                          metadata={"arm": "open", "task_set": "dev", **md}, tags=["sizing-x"]))
    assert S.analyst_key(log({})) == "Qwen3.5-27B|open|off|v2"
    assert S.analyst_key(log({"sampling": "qwen-thinking"})) == "Qwen3.5-27B|open|off|qwen-thinking|v2"
    assert S.analyst_key(log({"prepared": "glm-5.3"})) == "Qwen3.5-27B|open|off|prep-glm-5.3|v2"


def test_alt_judge_verdict():
    a = item("h01", wrong=False)
    a.wrong3 = True
    assert S.verdict(a, "alt") is True and S.outcomes([a], judge="alt") == ["wrong"]
    b = item("h02", wrong=True)
    assert S.verdict(b, "alt") is None  # not scored by the alternative judge
