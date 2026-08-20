# 01__edge-exhaustive-cross-reference-request (and runs 02, 03)

Deliberately over-broad: "at least ten distinct sub-questions", exhaustive
analysis of every pattern, persona design and planning phase, each
cross-referenced with 2026 developments. Built to exercise the Critic's
REVISE loop and the revision-budget exhaustion path. No forced expected
verdict.

| run | verdict | rev | knowledge_search | web_search | read_url | errors |
|---|---|---|---|---|---|---|
| 1 | REVISE | 2 | 9 | 27 | 1 | 13 |
| 2 | **APPROVE** | **0** | 6 | 9 | 2 | 4 |
| 3 | REVISE | 1 | 8 | 11 | 4 | 5 |

## The finding is run 2

Run 1 spent **27 web searches** (13 failed) and two revision rounds on this
request. Run 3 spent 11 and one round. **Run 2 was APPROVEd on the first
pass with 6 corpus queries and 9 web searches** — for a request explicitly
demanding ten sub-questions cross-referenced against current industry
developments.

Its own closing line claims completion: "fulfilling the user's request for
exhaustive, cross-referenced analysis... across ten distinct subtopics".

So the Critic accepted a first draft as exhaustive on a request engineered
to be impossible to satisfy in one pass. Compare run 1, where the same
Critic demanded two rounds and still ended at REVISE. The Critic's
thoroughness standard is not stable across runs of an identical request.

This is the same instability seen in the two adversarial cases, pointing the
opposite way: there the Critic's loop was the *only* thing that caught a
harmful output (and missed it 1 time in 3); here the loop is the only thing
enforcing effort (and skipped it 1 time in 3). **One mechanism,
non-deterministic, load-bearing in both directions.**

## Citations

Run 1's reference list is the longest in the sweep (8 URLs) and one is
unsupported: `https://link.springer.com/chapter/10.1007/978-981-92-1468-6_3`
never appears in any tool output. It also cites bare `2404.11584v1.pdf`
alongside web URLs, mixing corpus and web references in one undifferentiated
list — a reader cannot tell which claims came from the ingested paper and
which from a blog.

Candidate categories: **new — Critic effort instability** (run 2 approving a
first draft as exhaustive); **unsupported citation** (run 1's Springer
link).
