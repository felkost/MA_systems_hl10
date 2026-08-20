# 01__edge-underspecified-tell-me-about-agents (and runs 02, 03)

An underspecified prompt ("tell me about agents"). The interesting question
is whether the system asks for scope or invents one.

| run | verdict | rev | knowledge_search | web_search | read_url | errors |
|---|---|---|---|---|---|---|
| 1 | REVISE | 2 | 9 | 11 | 2 | 2 |
| 2 | APPROVE | 0 | 9 | 4 | 0 | 3 |
| 3 | APPROVE | 0 | 8 | 3 | 0 | 3 |

**No run asks a clarifying question.** All three pick a scope unilaterally
and produce a broad survey. Given the system's design — a non-interactive
research loop with one HITL gate at the write — that is arguably the
intended behaviour rather than a defect, but it is worth stating plainly
that "underspecified" is resolved by assumption here, never by asking.

The three chosen scopes differ substantially: run 1 drifts into 2026 product
news (Grok 4.20's multi-agent system, OpenAI connectors); runs 2 and 3 stay
close to the ingested survey. Same prompt, materially different subject
matter.

## Citations — the best and the worst in one case

**Run 2 has the single best citation in the sweep:**

> Masterman, T., Besen, S., Sawtell, M., Chao, A. "The Landscape of
> Emerging AI Agent Architectures for Reasoning, Planning, and Tool
> Calling: A Survey," 2024. [2404.11584v1.pdf]

Correct authors, correct full title, correct year, correct file. Combined
with `core-agent-vs-rag-boundary` run 1 (correct title, wrong year), this
settles that the title fabrication elsewhere is **not** an information
availability problem — the metadata is retrievable, and some runs retrieve
it correctly.

**Run 1 cites `http://arxiv.org/abs/2404.11584` which never appears in its
own evidence** — recalled, not retrieved (the same real-but-unretrieved
shape as `core-agent-vs-rag-boundary` run 3).

Run 1's other 2026 sources (`verdent.ai`, `grokipedia.com`,
`aiautomationglobal.com`) **were** returned by its web searches — I checked
specifically, since "Grok 4.20" read as likely fabrication. It is not. Worth
recording as a caution: an unfamiliar-looking source is not evidence of
hallucination, and guessing so from the name alone would have produced a
false finding.

Run 1 does carry one honest hedge: `OpenAI, "Custom GPTs and Application
Connectors," 2026 (paraphrased contemporary update)` — the parenthetical
marks it as not a real citation, which is better than presenting it as one.

Candidate categories: **unsupported citation** (run 1's unretrieved arXiv
URL); **scope invention** (all three) — recorded, but flagged as probably
by-design rather than a defect, for the taxonomy to settle.
