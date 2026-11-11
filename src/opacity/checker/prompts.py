"""The reviewer instruction.

Identical across every arm -- human, LLM, and (as a null) mechanical. It asks
about correctness, not disclosure risk, because the study is about whether
analytic error is visible, and it never names an error class, because the
free-text description is coded to class afterwards by someone blind to seeding.

Do not edit this between arms or between phases. It is part of the
pre-registration.
"""

REVIEWER_INSTRUCTION = """\
You are reviewing the output of a data analysis before it is released.

Read the analysis below and decide whether the stated interpretation is \
supported by the methods and results shown. You are judging correctness, not \
confidentiality: assume disclosure control has already been applied and some \
values may be withheld or rounded. Where that leaves you unable to judge, say so.

Answer with a JSON object and nothing else, in this form:

{
  "error_present": "yes" | "no" | "cannot_tell",
  "confidence": <integer 0-100>,
  "description": "<if yes: what is wrong, in one to three sentences. \
if cannot_tell: what you would need to see. if no: leave empty>",
  "release": "release" | "query" | "block"
}

"error_present" is "yes" if the interpretation makes a claim the analysis does \
not support, or the analysis is wrong in a way that would change the \
conclusion. It is "no" if the interpretation is a fair reading of the results. \
It is "cannot_tell" if the information needed to judge has been withheld.

"release" is your decision as a reviewer: release as is, return with a query, \
or block.
"""

SYSTEM_PREAMBLE = (
    "You are a careful methodological reviewer. You respond only with the JSON "
    "object requested."
)


TYPED_QUESTIONS = {
    "error_present": {
        "type": "choice",
        "instructions": (
            "Is the stated interpretation supported by the methods and results "
            "shown? You are judging correctness, not confidentiality: assume "
            "disclosure control has already been applied and some values may be "
            "withheld or rounded."
        ),
        "criteria": {
            "yes": (
                "the interpretation makes a claim the analysis does not support, "
                "or the analysis is wrong in a way that would change the conclusion"
            ),
            "no": "the interpretation is a fair reading of the results",
            "cannot_tell": "the information needed to judge has been withheld",
        },
    },
    "release": {
        "type": "choice",
        "instructions": "Your decision as a reviewer.",
        "criteria": {
            "release": "release as is",
            "query": "return with a query",
            "block": "block",
        },
    },
}
"""The typed arm's questions for a decision server (openjev). The criteria
restate the definitions in REVIEWER_INSTRUCTION word for word. Part of the
pre-registration, like the instruction: do not edit between arms or phases."""
