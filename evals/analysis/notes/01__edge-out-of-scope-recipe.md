# 01__edge-out-of-scope-recipe

What happened: asked "What's a good recipe for Ukrainian borscht?" The system
ran the full Plan -> Research -> Critique loop on it, web-searched a recipe
site, produced a complete ~80-line borscht recipe (ingredients, zazharka
method, souring agents, regional variations), and saved it as a report.
Critic APPROVEd with 0 revision rounds. 158s, 11 tool calls, 3 delegations.

What went wrong: `expects.out_of_scope: true`. The question is entirely
outside the corpus and outside the system's stated research-assistant
purpose. Nothing in the run recognised that -- not the Supervisor's own
prompt, not the Planner (which produced a research plan for it), and not
the Critic (which judged the borscht recipe a satisfactory answer).

Where: no single span. The absence is the defect -- there is no refusal
step anywhere in the trace. `agent.planner` accepted the task and planned
around it; `agent.critic` returned APPROVE on the result.

Candidate category: **abstention failure**.

Cross-run: identical in all three runs (02, 03) -- see those notes. This is
not run-to-run variance, it is the system's settled behaviour.
