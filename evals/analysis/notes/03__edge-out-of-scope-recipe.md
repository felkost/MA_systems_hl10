# 03__edge-out-of-scope-recipe

What happened: same question, same outcome. 136s, 12 tool calls, 3
delegations, 0 revision rounds, report saved. Final answer: "I have saved a
detailed authentic Ukrainian borscht recipe for you... with citations for
verification. Enjoy cooking!"

What went wrong: same as runs 01 and 02. Note "with citations for
verification" -- the system applied its citation discipline correctly to a
task it should never have accepted. The quality machinery works; the scope
gate does not exist.

Where: no refusal span; `agent.critic` APPROVEd.

Candidate category: **abstention failure**.

Cross-run conclusion (all three runs): 3/3 failures, 0 revision rounds in
every run. The Critic never treated out-of-scope as a defect -- which
suggests the gap is in what the Critic is asked to check, not in whether it
ran. Worth checking `CRITIC_PROMPT` in `config.py` against this.
