# 01__core-tool-calling-role (and runs 02, 03) — largely clean

Corpus-grounded question about a third section of the same paper.
`expects.must_cite: true`.

| run | verdict | rev | knowledge_search | web_search | read_url | errors |
|---|---|---|---|---|---|---|
| 1 | APPROVE | 0 | 7 | 0 | 0 | 0 |
| 2 | APPROVE | 0 | 8 | 0 | 1 | 1 |
| 3 | REVISE | 3 | 3 | 10 | 5 | 8 |

**Run 1 is the cleanest trace in the entire sweep**: 7 `knowledge_search`
calls, zero web searches, zero errors, correct answer, APPROVE on first
pass. When the question is squarely inside the corpus and the search
backend is not consulted, the system behaves exactly as designed.

**Run 3 is the same question going the other way** — only 3 corpus queries
against 10 web searches (8 of which errored), 3 REVISE rounds. The content
is still correct; the route to it was three times as expensive and mostly
unnecessary.

## Citations

- run 1: `*This report is based on insights extracted from the ingested
  survey document.*` — no identifier, no page. Weak: technically a source
  attribution, but nothing a reader can check.
- run 2: `Survey on AI agent implementations focusing on reasoning,
  planning, and tool calling, pages 1-3, 10 [2404.11584v1.pdf].` — a
  paraphrase of the real title rather than the title, but the filename and
  pages are right and checkable.
- run 3: `Survey on AI Agent Architectures, arXiv:2404.11584v1,
  https://arxiv.org/abs/2404.11584v1` — shortened title, correct id and URL.

Same **correct identifier, wrong title** shape as `core-agent-persona`, and
run 1 adds a weaker variant: an attribution with no identifier at all.

Candidate categories: **unsupported citation (weak form)** for run 1's
identifier-free attribution; wrong-title sub-shape for runs 2 and 3; no
correctness failure in any run.
