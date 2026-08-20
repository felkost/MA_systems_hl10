# 01__core-agent-vs-rag-boundary (and runs 02, 03)

Sits deliberately on the corpus/web boundary — the survey frames agents as
the next generation beyond RAG chat, but "why might a team choose one over
the other" reaches past it. `expects.must_cite: true`.

| run | verdict | rev | knowledge_search | web_search | read_url | errors |
|---|---|---|---|---|---|---|
| 1 | REVISE | 1 | 6 | 6 | 4 | 3 |
| 2 | APPROVE | 0 | 6 | 5 | 1 | 1 |
| 3 | REVISE | 2 | 9 | 18 | 8 | 10 |

The mixed corpus/web tool usage is **appropriate here** — unlike
`core-agent-persona`, this question genuinely spans both. Run 3's 18 web
searches (10 errored) are excessive, but not wrong in kind.

## Citation quality splits three ways

- **run 1 — the best citation in the sweep:** `The Landscape of Emerging AI
  Agent Architectures for Reasoning, Planning, and Tool Calling: A Survey,
  arXiv:2404.11584, 2025. http://arxiv.org/abs/2404.11584` — the **correct
  full title**, correct id, working link. Only the year is wrong (2025; the
  paper is April 2024). So the title fabrication seen elsewhere is not a
  constant: this run got it right, which means the information was
  available to the others.

- **run 2 — non-citations:** `- Research papers on ReAct and interactive
  reasoning methods. | - Discussions on decision criteria for selecting RAG
  vs AI agent architectures.` Two bullet points naming *topics*, not
  sources. Nothing here identifies a document; nothing can be checked. This
  is the weakest form of the unsupported-citation shape — the report has a
  References heading and no references under it.

- **run 3 — real but unretrieved:** cites `arXiv:2403.12482` (Embodied LLM
  Agents Learn to Cooperate in Organized Teams), `arXiv:2307.05300`
  (Unleashing the Emergent Cognitive Synergy), `arXiv:2310.01557`
  (SmartPlay). These are **real papers**, and they are references from the
  survey's own bibliography — but the run never retrieved or read any of
  them. The claims attached to them may well be true; the run has no
  evidence for them.

That last shape deserves its own name. Spec §13.3's standard is "a plausible
claim without evidence is a FAIL" — by that standard run 3's citations fail
*even though they are factually correct*, because the run is passing off its
model's parametric recall as retrieved evidence. It is the hardest
citation defect to catch, because every individual reference checks out.

Candidate categories: **unsupported citation**, three distinct sub-shapes
(topic-only non-citation; real-but-unretrieved; correct-title-wrong-year).
