# How small can a safe AI analyst be? A plain-language summary

*OPACITY model-sizing study, 22 to 25 October 2026. Full results: the capabilities preprint,
[`docs/build/opacity-preprint.pdf`](build/opacity-preprint.pdf).*

## The question

A trusted research environment (TRE) holds sensitive data, such as health
records, that cannot leave the building. safe-tre-agent puts an AI assistant
inside the TRE to run statistical analyses for researchers who never see the
raw data. Because the data cannot leave, the assistant has to be a model that
runs on the TRE's own computers, and smaller models are cheaper and easier to
run there. The risk is that a small model hands a researcher a confident but
wrong conclusion. We asked how small the model can be before that happens too
often.

## How we tested it

We wrote 108 realistic analysis requests, each with a small table of data.
Most contain one of eight traps that catch real analysts:

- testing twenty subgroups and reporting the one that looks different
- ignoring people who dropped out
- comparing groups that differ in make-up
- using the wrong denominator for a rate
- treating patients in the same clinic as independent
- mistaking a natural fall back from an extreme value for a treatment effect
- claiming cause from observational data
- judging a prediction model on the data it was built from

Some requests have no trap, to check that the assistant still answers honest
questions.

Each model answered every request three times, with fresh numbers each time.
A separate AI model, from a family not being tested, compared each answer
with a written key and judged whether its conclusion was wrong. We counted
how often a wrong conclusion would have reached the researcher. Our target
was no more than 1 in 20 on trap questions, while still answering at least
three in four of the honest ones.

The requests came in three sets. We developed our methods on the first, tested
them on the second, and kept the third sealed: its results were not looked at
until every choice had been written down and committed. We then wrote a
fourth set of 40 harder requests, also sealed. In most of them the trap shows
only in how the study was run, and every honest question carries a real
effect, so explaining that effect away also counts as a wrong conclusion.

## What we found

**The instructions mattered more than the size of the model.** Asking the
model to write down the design of the study before interpreting it (who was
compared with whom, how many comparisons were made, who is missing, what each
rate is out of) and then to apply ten short rules cut its wrong conclusions to
between a third and a tenth of what they had been, depending on the model and
the set of requests.

**With those instructions, mid-sized models reached the target on the sealed
set.** A 27-billion-parameter model got 2 of 96 trap questions wrong, and a
35-billion-parameter model that uses only 3 billion parameters at a time
(it fits on two ordinary data-centre graphics cards) got 4 of 96 wrong. Both
still released 95 to 98 per cent of the correct answers. A model of 9 billion
parameters stayed at about 1 in 10 on the same set.

**Going much larger did not help further.** A 122-billion-parameter model did
no better than the 35-billion one. Several of the large models first looked
worse than they were because of how we ran them (settings their makers advise
against, and a fault in our own code that marked some correct answers as
unreadable). Once fixed, they joined the others rather than pulling ahead.

**A second model as a checker adds a little safety at a clear cost.** Passing
each answer through a second, different model before release removed the
last few wrong conclusions, but it also held back between one honest answer
in twelve and one in six.

**The harder sealed set confirmed the 27-billion model with a checker.** It
got 3 of 96 trap questions wrong and answered five in six honest ones. On its
own the 27-billion model got 5 of 96 wrong and the 35-billion model 4 (one
of those four was a mistake in our answer key, below). Whether the models on
their own also pass depends on how strictly a judge reads caution on the
honest questions. The first judge failed both of them there. A second judge
from another family (DeepSeek) passed the 27-billion model on its own: the
three honest answers the first judge failed read to it as reasonable
caveats, which is how we read them too. For the 35-billion model the second
judge could not score two of the answers that matter, so that case is open.
Human statisticians will decide which reading is right.

**Some configurations that looked best on one set fell back on the next.**
That is why the sealed set mattered: the three configurations that passed
were the ones that had been steady across all three sets, not the ones with
the single best score.

## What did not help

**A calculator.** Most of the arithmetic in the answers was done "in the
head", and some of it was wrong, so we gave the model statistical tools. The
slips went, but the tools made it easy to put a precise number on a shaky
comparison. A second version with safeguards removed those mistakes, and its
new tools caused others. Neither version beat the written instructions with
the sums done by hand.

**Planning outside the TRE.** A strong online model wrote a plan for each
request from the shape of the data alone, never seeing a number, and the
model inside did the work. With plain instructions the plan helped. With the
written instructions it made things worse, because a plan written before
anyone sees the data has to choose the main comparison before anyone knows it
is the misleading one. It would also make the TRE depend on an online model.

**A stronger checker.** Scored over every possible threshold, the checkers
separate real wrong answers from right ones only a little better than chance
(0.58 to 0.72, where 0.5 is chance and 1.0 is perfect). The one we use scores
0.63, against 0.92 on mistakes we planted ourselves. A checker is a dial
between caution and cost, not a guarantee.

## Mistakes we found in our own measuring

Two of our 64 sealed trap questions had an answer key that was wrong on one
of their three versions: a result really did survive the allowance for
testing many groups, and models that said so were marked wrong. We fixed the
program that makes the data so that it cannot happen again, and corrected
the counts. No conclusion changed. The judges are not perfect either: one
marked the same claim right for one model and wrong for another.

## What we cannot yet say

Three AI judges from three different companies have now marked the first
sealed set. Each pair agrees on about 94 answers in 100, and the three
configurations pass under all three. Two judges marked the harder set. A
check by human statisticians is still to come. Each result rests on 96
attempts, so a measured 2 in 96 is consistent with a true rate of up to about
1 in 20. The tasks were written by an AI and are tidier than real requests. Although the
instructions prevent errors, they can also make a model explain away a real
effect, and judges disagree about where caution ends and explaining away
begins. In the full working system, run end to end on a separate bank of nine
questions, the 35-billion model mostly erred on the side of declining. When
that system began to require a statistical test behind every claim, the
model at first took a significant but unadjusted test as proof of a link that
was really explained by differences in income. Once the system offered a test
that adjusts for income, the model called that question correctly in all six
runs, although in those runs it also chose a different view of the data, so
the rerun does not show which change fixed it.

## What this means

A competent statistical assistant inside a TRE does not need a very large or
expensive model. A mid-sized model, run on modest hardware with careful
instructions, kept wrong conclusions near or below 1 in 20 on tasks it had
never seen. The set-up that passed both sealed sets, under every judge, is the
27-billion model with the written instructions and a checker from another
family. The 35-billion model is cheaper and passed the first sealed set, but
with the checker it held back too many honest answers on the harder one.
Adding tools or an outside planner did not improve on careful instructions.
The next steps are review by human statisticians, a run of the recommended
set-up in the full working system, and then, with a TRE's agreement, a test on
real data.
