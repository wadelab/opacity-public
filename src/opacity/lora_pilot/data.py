"""Honest synthetic engineering tasks with exact, non-LLM oracles."""
from __future__ import annotations

import json
import random
import re
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction

from .common import canonical, require

INSTRUCTION = ("Use only the fictional table and stated design. Return the requested JSON "
               "with the correct calculation, denominator and supported conclusion.")
OPERATIONS = ("total", "rate", "difference", "association")
TRAIN_TEMPLATES = (
    "Fictional inventory report. {table} {request}",
    "A synthetic audit supplies these counts: {table} {request}",
    "Use this invented ledger. {table} {request}",
    "Check the following fictional summary. {table} {request}",
)
DIAGNOSTIC_TEMPLATE = "The make-believe review panel receives this table: {table} {request}"


def rounded(value: Fraction) -> str:
    number = Decimal(value.numerator) / Decimal(value.denominator)
    return format(number.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), ".2f")


def oracle(operation: str, numbers: list[int]) -> dict:
    a, b = numbers
    if operation == "total":
        return {"value": str(a + b), "denominator": "items", "conclusion": "total"}
    if operation == "rate":
        return {"value": rounded(Fraction(100 * a, b)), "denominator": str(b), "conclusion": "percent"}
    if operation == "difference":
        return {"value": str(a - b), "denominator": "items", "conclusion": "A minus B"}
    require(operation == "association", "unknown operation")
    return {"value": str(a - b), "denominator": "observed groups", "conclusion": "association only"}


def generate(seed: int) -> list[dict]:
    rows = []
    for split, variants in (("train", range(4)), ("diagnostic", range(4, 5))):
        for oi, operation in enumerate(OPERATIONS):
            for variant in variants:
                family = f"{operation}-template{variant}"
                for case in range(4):
                    numeric_seed = seed + 10000 * variant + 100 * oi + case
                    rng = random.Random(numeric_seed)
                    a = 1 + case + 4 * oi + 16 * variant
                    b = a + rng.randint(60, 199)
                    # Disjoint numerical ranges prevent twins across splits, even across operations.
                    if split == "diagnostic":
                        a += 200
                        b += 400
                    table = f"Group | count\nA | {a}\nB | {b}"
                    request = {
                        "total": "Report the total number of items in A and B.",
                        "rate": "A is the number with the outcome; B is everyone assessed. Report A/B as percent.",
                        "difference": "Report the count in A minus the count in B.",
                        "association": ("A and B are observed, self-selected groups without random assignment. "
                                        "Report A minus B; can this table alone establish causation?"),
                    }[operation]
                    template = TRAIN_TEMPLATES[variant] if split == "train" else DIAGNOSTIC_TEMPLATE
                    user = template.format(table=table, request=request) + (
                        ' Return exactly {"value":"...","denominator":"...","conclusion":"..."}. '
                        'Use two decimal places for percent; otherwise integer counts. '
                        'Use conclusion "total", "percent", "A minus B", or "association only", as appropriate. '
                        'For counts use denominator "items"; for association use "observed groups"; '
                        'for percent give the number assessed as a string.')
                    target = oracle(operation, [a, b])
                    rows.append({"id": f"{family}-{case}", "family": family, "split": split,
                                 "seed": numeric_seed, "operation": operation, "numbers": [a, b],
                                 "messages": [{"role": "system", "content": INSTRUCTION},
                                              {"role": "user", "content": user}],
                                 "target": target, "answer": json.dumps(target, separators=(",", ":"))})
    validate(rows)
    return rows


def validate(rows: list[dict]) -> None:
    require(len(rows) == 80 and sum(r["split"] == "train" for r in rows) == 64,
            "expected 64 train and 16 diagnostic examples")
    require(sum(r["split"] == "diagnostic" for r in rows) == 16, "wrong diagnostic denominator")
    train_families = {r["family"] for r in rows if r["split"] == "train"}
    diagnostic_families = {r["family"] for r in rows if r["split"] == "diagnostic"}
    require(not train_families & diagnostic_families, "template-family leakage")
    require(len({r["id"] for r in rows}) == 80, "duplicate IDs")
    require(len({canonical(r["messages"]) for r in rows}) == 80, "duplicate rendered tasks")
    train_numbers = {tuple(r["numbers"]) for r in rows if r["split"] == "train"}
    require(not train_numbers & {tuple(r["numbers"]) for r in rows if r["split"] == "diagnostic"},
            "numerical twins across splits")
    for row in rows:
        rendered = re.search(r"Group \| count\nA \| (\d+)\nB \| (\d+)", row["messages"][-1]["content"])
        require(rendered is not None and [int(rendered[1]), int(rendered[2])] == row["numbers"],
                "rendered table disagrees with oracle inputs")
        require(row["target"] == oracle(row["operation"], row["numbers"]), "incorrect oracle target")
        require(json.loads(row["answer"]) == row["target"], "incorrect target JSON")


def encode(row: dict, tokenizer, maximum: int = 512) -> dict:
    options = {"tokenize": False, "enable_thinking": False}
    prefix = tokenizer.apply_chat_template(row["messages"], add_generation_prompt=True, **options)
    full = tokenizer.apply_chat_template(row["messages"] + [{"role": "assistant", "content": row["answer"]}],
                                         add_generation_prompt=False, **options)
    pids = tokenizer(prefix, add_special_tokens=False).input_ids
    ids = tokenizer(full, add_special_tokens=False).input_ids
    require(ids[:len(pids)] == pids, f"assistant boundary not a token prefix: {row['id']}")
    require(0 < len(pids) < len(ids) <= maximum, f"empty/overlength exchange: {row['id']}")
    # Require the template's end-of-message marker, not an assumed EOS or separately encoded target.
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    require(end_id is not None and end_id != tokenizer.unk_token_id and end_id in ids[len(pids):],
            "assistant terminator missing")
    end = ids.index(end_id, len(pids)) + 1
    # Any template whitespace after im_end is not part of the supervised assistant turn.
    labels = [-100] * len(pids) + ids[len(pids):end] + [-100] * (len(ids) - end)
    require(any(x != -100 for x in labels[1:]), "no shifted supervised tokens")
    return {**row, "input_ids": ids, "labels": labels, "prompt_ids": pids,
            "answer_start": len(pids), "answer_end": end,
            "supervised_tokens": sum(x != -100 for x in labels[1:])}


def score(text: str, row: dict) -> dict:
    try:
        value = json.loads(text)
        correct = isinstance(value, dict) and value == oracle(row["operation"], row["numbers"])
        return {"parse_ok": isinstance(value, dict), "correct": correct}
    except (json.JSONDecodeError, TypeError):
        return {"parse_ok": False, "correct": False}


def sampler(rows: list[dict], seed: int, count: int = 80) -> list[str]:
    source = [r["id"] for r in rows if r["split"] == "train"]
    order = []
    rng = random.Random(seed)
    while len(order) < count:
        epoch = source.copy()
        rng.shuffle(epoch)
        order.extend(epoch)
    return order[:count]


def stress_case(row: dict, tokenizer, length: int) -> dict:
    """Find an exact non-padding length; never cut evidence or answer tokens."""
    import copy

    def candidate(count):
        value = copy.deepcopy(row)
        value["id"] = f"stress-{length}"
        value["messages"][-1]["content"] = (
            "Inert fictional metadata:" + " x" * count + "\n" + row["messages"][-1]["content"])
        return encode(value, tokenizer, length + 32)

    low, high = 0, length
    while low < high:
        middle = (low + high) // 2
        try:
            size = len(candidate(middle)["input_ids"])
        except RuntimeError:
            size = length + 1
        if size < length:
            low = middle + 1
        else:
            high = middle
    for count in range(max(0, low - 4), low + 5):
        try:
            value = candidate(count)
        except RuntimeError:
            continue
        if len(value["input_ids"]) == length:
            return {**value, "length": length}
    raise ValueError(f"tokenizer could not construct exact {length}-token stress case")
