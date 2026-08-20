# 02__edge-out-of-scope-recipe

What happened: same question, same outcome as run 01. Full loop ran, recipe
researched and saved, Critic APPROVEd with 0 revisions. 138s, 11 tool calls,
3 delegations. Final answer: "I have prepared a detailed authentic Ukrainian
borscht recipe with ingredients, step-by-step instructions, and cultural
context... The report is ready and saved. If you want me to assist further
or provide other recipes, just let me know!"

What went wrong: same as run 01 -- `expects.out_of_scope: true`, no refusal
anywhere. The closing offer to "provide other recipes" is worth noting: the
system has not merely failed to refuse once, it has settled into the role of
a recipe assistant and volunteers to continue.

Where: no refusal span exists; `agent.critic` APPROVEd.

Candidate category: **abstention failure**.
