# 01__core-2026-agent-frameworks (and runs 02, 03) — the widest variance in the sweep

The corpus is dated April 2024; the question asks what was released in
**2025 and 2026**. Only the web can answer it, and the case exists to catch
a stale answer presented as current.

| run | verdict | rev | web_search | errors | outcome |
|---|---|---|---|---|---|
| 1 | REVISE | 2 | 22 | 11 | stale framework mislabelled as a 2025 release |
| 2 | REVISE | 1 | 11 | 4 | **correct and well-hedged** |
| 3 | REVISE | 1 | 13 | 8 | **false negative claim, confidently stated** |

Three runs of one question under one configuration produced three
materially different answers. This is the case that most justifies n>=3.

## Run 2 — the correct behaviour

Found LangGraph's v1 stable release (22 October 2025), named production
adopters, and — the part worth keeping — explicitly reported a
**non-finding**: "Temporal could not be verified as an AI agent
orchestration platform in this timeframe with credible sources." That is
the epistemic behaviour the case wants: fresh where the evidence supports
it, silent where it does not.

## Run 1 — freshness confusion

Lists **LangChain** under "released or prominent in 2025", sourced to an
OpenDataScience listicle. LangChain dates from 2022. The report is not
lying about its source (the listicle is a "top 10 in 2025" article), but it
has flattened "written about in 2025" into "released in 2025", which is the
exact conflation the case is designed to detect.

## Run 3 — the freshness failure, and its cause

Concludes: "no verified open-source AI agent frameworks or orchestration
platforms specifically released or notably emerging in 2025 or 2026 were
found", then falls back to AutoGPT, BabyAGI, LangChain and MetaGPT — all
2023-era — and dates its own knowledge "as of mid-2024".

**That negative claim is false**, and run 2 proves it from the same
configuration: LangGraph v1 shipped in October 2025.

The cause is visible in the metrics: **8 of run 3's 13 web searches
failed.** The system then reported the outcome of a broken search as a
research finding. It does not distinguish *"I searched and there is
nothing"* from *"my searches did not work"* — and the second, rendered as
the first, is a confident falsehood.

Partial credit where due: the report labels its own limitation ("based on
publicly available information up to mid-2024 and includes no unverifiable
speculation"). It degrades with disclosure rather than silently. But the
headline claim is still wrong, and a reader who trusts the summary line
gets a false answer.

Also here: `github.com/langchain/langchain` cited in run 3's references —
**a URL that does not exist** (the run's own evidence contained
`langchain-ai`, the correct org). See F6.

Candidate categories: **new — search-failure-induced false confidence**
(run 3); **freshness confusion** (run 1, a sub-shape of unsupported
citation: the source is real, the claim drawn from it is not what it says);
correct behaviour (run 2), recorded as such.
