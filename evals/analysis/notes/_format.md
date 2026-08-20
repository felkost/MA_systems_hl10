# Open-coding notes format

One file per read trace, `<run_index>__<case_id>.md`, written while reading
`runs/<run-set-name>/packs/<run_index>__<case_id>.json` next to the raw
`spans/*.jsonl` for that run-set -- free-form, before any category is
assigned (spec Sec5.3): what the run did, what it got wrong, and where.
Categorisation into `../taxonomy.md` happens only after every trace in the
sweep has its own note here.

Suggested shape per file, not a schema:

```
# <run_index>__<case_id>

What happened: ...
What went wrong (if anything): ...
Where (span name / tool call / delegation): ...
Candidate category: ...
```

This file itself (`_format.md`) is the only note this stage's kickoff
writes -- every `<run_index>__<case_id>.md` file is produced by actually
reading a sweep's traces, never in advance of one existing.
