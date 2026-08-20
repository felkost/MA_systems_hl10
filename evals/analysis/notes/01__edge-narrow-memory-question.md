# 01__edge-narrow-memory-question (and runs 02, 03) — clean, and it tests something specific

A compound question: *does* the survey's memory discussion differ between
single- and multi-agent architectures, **and if so, how**. The case exists
to catch a system that silently answers one half — the failure mode
`RESEARCH_INPUT_TEMPLATE`/`CRITIQUE_INPUT_TEMPLATE` were built to prevent
at the coordination layer (hl8 measured a coordinator dropping context in
five of six paraphrases).

| run | verdict | rev | knowledge_search | web_search | read_url | both parts answered |
|---|---|---|---|---|---|---|
| 1 | APPROVE | 0 | 7 | 1 | 1 | yes |
| 2 | APPROVE | 1 | 17 | 2 | 2 | yes |
| 3 | APPROVE | 1 | 14 | 2 | 6 | yes |

**All three runs answer both halves** — each states that the discussion does
differ, then characterises how (internal/centralised memory in single-agent
vs communicative/shared memory across agents in multi-agent). No sub-question
was dropped in any run.

This is a genuine negative result and worth recording as one: the
context-forwarding mechanism the coordination layer was built around holds
on the case designed to break it. A taxonomy that only lists failures would
lose this.

## Minor observations

**Redundant retrieval at its most extreme.** Run 2 issues **17**
`knowledge_search` calls for a question the corpus answers in two passages.
This is the same non-convergence seen in `core-single-vs-multi-agent`, at
roughly double the volume.

**Citations are the strongest of any case.** Run 1: `**Source:**
2404.11584v1.pdf, pages 1-2, 7`. Run 3: `Ingrested survey document
[2404.11584v1.pdf], pages 1-2, 7-8` (note the typo "Ingrested" — cosmetic).
Both name file and pages, both checkable, neither invents a title. Whatever
drives the title fabrication elsewhere did not fire here.

Candidate category: none — clean. Recorded as a zero-instance data point for
**dropped sub-question**, and as the counter-example that keeps
"redundant retrieval" from being read as a correctness problem.
