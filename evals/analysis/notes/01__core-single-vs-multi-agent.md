# 01__core-single-vs-multi-agent (and runs 02, 03)

Question is explicitly corpus-grounded ("Based on the ingested survey"),
`expects.must_cite: true`.

## What the runs did

| run | verdict | revisions | latency | cost | knowledge_search | web_search | read_url |
|---|---|---|---|---|---|---|---|
| 1 | APPROVE | 0 | 130s | $0.0158 | 8 | 2 | 3 |
| 2 | APPROVE | 0 | 137s | $0.0156 | 10 | 1 | 2 |
| 3 | REVISE | 2 | 309s | $0.0437 | 8 | 6 | 7 |

Tool *selection* is right: `knowledge_search` dominates and every hit comes
from the corpus PDF (`2404.11584v1.pdf`). No wrong-tool-selection here in
the plain sense.

## Signals worth naming

**Redundant retrieval.** Each run fires 8-10 `knowledge_search` calls that
are near-identical paraphrases of one another ("distinction between
single-agent and multi-agent AI architectures" / "Differences between
single-agent and multi-agent AI architectures" / "single-agent versus
multi-agent AI architectures"), and several return the *same* page-2 chunk.
The retrieval is correct but not converging -- the agent re-asks rather than
concluding it already has the answer.

**Fetching a source it already has locally.** Every run tries
`read_url("https://arxiv.org/pdf/2404.11584v1.pdf")` -- the PDF of the very
paper already ingested into the knowledge base -- and it fails every time
("no readable text extracted", trafilatura on a PDF). Run 2 also passes a
bare filename as a URL (`read_url("2404.11584v1.pdf")` -> scheme error).
This is a distinct shape from classic wrong-tool-selection: not "used the
web where the corpus had it", but "used the web to re-fetch what the corpus
already gave it".

**Run-to-run cost/latency spread.** 2.8x cost and 2.4x latency between run 2
and run 3 on the *same question and the same configuration*. Run 3's
`web_search` also hit two live errors (rate-limit shaped). Any single-run
cost figure for this system would be misleading; the spread is the finding.

**Scope drift under revision.** Run 3, the one that went through 2 REVISE
rounds, is the one that pulled in Microsoft Learn and Redis blog pages for a
question that said "based on the ingested survey". The Critic's pressure to
improve pushed the answer *away* from the grounding the question demanded.

## The instrument defect this case exposed (the important one)

Reading the pack, I concluded "the report contains no citations at all" --
and run 2's final answer claims it does. That looked like a clean
unsupported-citation finding.

It was wrong, and the reason matters: **`save_report`'s `content` is
truncated at 2000 characters in the evidence pack** (`input_truncated:
true`), and this report's only citation is its *last line*:

> *This summary is based on the survey in arXiv:2404.11584v1 (pages 1-2,
> 6-9).*

Checked against the real file in `runs/sweep-01/output/` -- the citation is
there, and the pages it names (1-2, 6-9) are pages the run actually
retrieved. Citation supported.

Measured across the whole sweep: **31 of 35 saved reports are truncated in
the packs** (median report 3493 bytes, max 7622, against a 2000-char cap).
So for 31 of 36 traces the judge cannot see the end of the report -- which
is exactly where this system puts its citations.

Consequence: `citation_reliability` **cannot be scored from the packs as
they stand**. Any judge run now would fail most cases for missing evidence
the agent actually produced. This is spec risk #2 ("the evidence pack could
itself be the defect") realised, and it is the reason the hand-read of one
pack is a go/no-go item rather than a nicety.

Candidate category: **new -- instrument defect, not a system failure.** Must
be fixed before `evals.score` is run, or its citation numbers are noise.
