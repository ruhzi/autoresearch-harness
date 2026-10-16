# The vanilla arm

[pi](https://github.com/badlogic/pi-mono) 1.1.0 (`@earendil-works/pi-coding-agent@1.1.0`)
as shipped, with its stock prompts, no extensions, and no settings beyond the model.

Each run was started once, non-interactively, in a fresh git repository containing
`BRIEF.md` and a minimal starting `agent/`:

```
pi -p --model <provider>/z-ai/glm-5.3-flash "$(cat PROMPT.md)"
```

and resumed with `pi -p -c ...` whenever pi exited before the budget was spent or `DONE`
was written. The budget is the only clock: a call has no turn limit and no wall-clock limit.

The custom arm was started with exactly the same line plus `PI_CODING_AGENT_DIR` pointing at
`harness/`.

## The first prompt

```
Read `BRIEF.md` in this repository. It describes the job, the conditions, and
your budget. Then work on the job.

Conventions of this repository:
- `agent/` is the deliverable. `agent/README.md` is its contract.
- `results.tsv` is your log: one line per experiment with what you changed,
  how you measured it, the result, and whether you kept it. Keep it current;
  it is how you remember what you did if your context is reset.
- `practice/` is yours for anything you build to test with.
- The model endpoint is `$INNER_API_BASE`, model `$INNER_MODEL`, key
  `$INNER_API_KEY`. There is no other network.
- Write a file named `DONE` in the repository root when you judge that no
  further experiment is worth its cost.

Your work is saved after each step. If you stop before you are done, you will
be asked to continue.
```

## The resume prompt

```
Continue the job. Re-read `results.tsv` and `BRIEF.md` if your context was reset.
Write `DONE` in the repository root when no further experiment is worth its cost.
```

## The starting `agent/`

Both arms started from the same minimal agent: the contract in `agent/README.md` (the
first section of every delivered agent's README is that contract, unchanged) and a
`run.py` that is the smallest loop that works, with read, bash, edit and write tools and
no limits of any kind. The brief says it is a starting point, not a recommendation.
