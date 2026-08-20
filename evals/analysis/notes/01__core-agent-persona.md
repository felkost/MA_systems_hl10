# 01__core-agent-persona (and runs 02, 03)

Question asks for a term "in the ingested survey's terminology" --
`expects.must_cite: true`. The survey defines it verbatim on page 1, so this
is a precise-retrieval case with one right source.

| run | verdict | revisions | latency | cost | knowledge_search | web_search | read_url |
|---|---|---|---|---|---|---|---|
| 1 | APPROVE | 1 | 231s | $0.0223 | 10 | 4 (1 err) | 2 (2 err) |
| 2 | REVISE | 2 | 264s | $0.0331 | 7 | 7 (4 err) | 4 (2 err) |
| 3 | APPROVE | 1 | 247s | $0.0258 | 8 | 6 (1 err) | 5 (3 err) |

## Run 1 is the correct behaviour, and shows what the others missed

Run 1's answer quotes the survey verbatim with a page number:

> "Agent Persona. An agent persona describes the role or personality that
> the agent should take on, including any other instructions specific to
> that agent. Personas also contain descriptions of any tools the agent has
> access to." [2404.11584v1, p.1]

That is the survey's own definition, retrieved from the corpus, quoted
rather than paraphrased. Exactly what the question asked for.

## Wrong tool selection — runs 2 and 3

The corpus answers this question verbatim on page 1. Runs 2 and 3 still
spent 6-7 `web_search` calls each on it and read third-party pages:

- run 2: `arxiv.org/html/2601.01743v1` (**a different paper entirely** --
  2601.01743, not 2404.11584), `thenewstack.io`, `medium.com`
- run 3: `azilen.com/learning/agent-persona/`,
  `github.com/mthalman/agent-personas`

The question said "in the ingested survey's terminology". Pulling
third-party definitions of the same phrase is precisely the
wrong-tool-selection shape the 2026-08-15 plan note names: each individual
call looks locally plausible, and only the question's own wording makes it
wrong.

## Fabricated source titles — 2 of 3 runs (the strongest finding here)

The real title, checked in `index/chunks.json` chunk 0:
**"The Landscape of Emerging AI Agent Architectures for Reasoning, Planning,
and Tool Calling: A Survey"** (Masterman, Besen, Sawtell et al.).

What the saved reports cite:

- **run 1:** `2404.11584v1, "Brain, Perception, and Action," pp. 1, 9.` --
  "Brain, Perception, and Action" is not the title; it is a quoted phrase
  from the body of page 1 (`"brain, perception, and action" [31]`), lifted
  out of a sentence and presented as the paper's name.
- **run 2:** `"Survey on Architectures of Language Model Powered Agents,"
  arXiv:2404.11584v1, 2024.` -- **an invented title.** Plausible, wrong, and
  attached to a correct arXiv id, which makes it harder to catch than an
  invented id would be.
- **run 3:** `*Source: Ingested AI agent architecture survey document, page
  1.*` -- vague but not false.

So the arXiv identifier is right every time while the human-readable title
is fabricated in 2 of 3 runs. A reader checking the id lands on the right
paper; a reader searching the quoted title finds nothing.

Candidate category: **unsupported citation** -- and a sharper sub-shape
worth naming separately: *correct identifier, fabricated title*. The claim
is not unsupported in the sense of pointing at nothing; it points at the
right thing under a wrong name.

## Other observations

**Redundant retrieval, again.** Run 1 issues the identical query
`knowledge_search("agent persona")` four times and gets the identical
page-1 chunk back each time.

**`web_search` is operationally flaky.** Across these three runs: 6 of 17
web searches failed outright ("error sending request", "No results found"),
and 7 of 11 `read_url` calls errored -- almost all of them the same
`https://arxiv.org/pdf/2404.11584v1.pdf` PDF that trafilatura cannot
extract. The agents keep retrying a fetch that has never once succeeded in
this sweep.

**Citation lives in the truncated tail again.** All three citations above
are the last line of their reports, i.e. invisible in the packs (F1).
