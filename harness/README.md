# The custom autoresearcher

A pi 1.1.0 agent directory: point `PI_CODING_AGENT_DIR` at this folder and run pi as usual.
It is vanilla pi plus two extensions:

1. **subagent**: pi's own example extension, copied unchanged, with one `worker` agent.
2. **autoresearch**: the pipeline. Once, before the researcher's first turn:
   - **identify**: a root pass breaks the task (the researcher's own first prompt) into load-bearing
     threads; one pass per thread breaks it into sub-threads (depth 2). Each pass quotes the task's own
     words and says why the thread bears load.
   - **order**: the root conversation gets one more turn: put the threads in the order they should be
     built, by dependency (the canon). On failure the order is T1, T2, ...
   - **implement**: in a git worktree of the workspace (under the pending directory, removed afterwards), each
     first-level thread's conversation gets one more turn: implement its sub-threads and report. One
     thread at a time, in canon order; each turn is given the reports of the turns before it and may
     change earlier work if it names the change and the reason. After any successful edit, if that
     conversation is at `stop_percent` of its context window, it is steered to stop and report (it is not
     told this in advance). The worktree's changes are saved as `implement.diff` (a `DONE` file an
     implementer may have created is left out: ending the job is the researcher's call).
   - **think**: the researcher starts with the plain task plus one instruction: look around, change
     nothing yet, reason out loud in your reply about what you would build, how you would test it and what
     could go wrong. Edits and writes are refused during that turn.
   - **hand over**: when the researcher replies without tool calls, `implement.diff` is applied to the
     workspace and the next turn tells it what was built, in what order, and that it is the final
     reviewer and integrator: hold the build to its own plan, change what conflicts, and treat ending
     with no changes as a finding to justify. (Fallback: after 40 tool results with no plan, the
     hand-over is delivered anyway.)

Until the hand-over, everything the pipeline produces waits in a pending directory under the OS temp dir
(`autoresearch-pending-<hash of the workspace path>`), outside the workspace, so the planning turn cannot reach
it by any path, including a recursive grep or find; only `state.json` is in `<workspace>/.harness/` then, and the
planning guard also refuses any tool call naming `.harness`, the pending directory, or the temp dir at all. At
the hand-over the pending directory is moved into `<workspace>/.harness/`.

The implementation turn tells each implementer that earlier threads' work is a draft it has the same authority
over as its own, that its own reading of the brief governs, and that for every earlier decision it touches it
must report whether it kept or changed it and why. The hand-over asks the researcher to reconcile its blind plan
in two passes, against the thread tree and against the code as built, and tells it that nothing in either is
settled and that the implementers' own tests passing is not evidence.

A model call that fails outright (rate limit, outage, pi exiting without an answer) is retried after waits of
15, 30, 60 and 120 seconds before the pass, ordering turn or implementation turn is given up. If the root
pass still fails, the pipeline is marked failed and the researcher runs plain: no planning steer, no hand-over.

## Layout

```
settings.json              default provider/model; enables both extensions
models.json                the provider pi calls (OpenRouter here; set OPENROUTER_API_KEY)
autoresearch.json          pipeline settings: auto_pipeline, max_depth, max_threads, stop_percent
extensions/subagent/       pi 1.1.0's examples/extensions/subagent, unchanged (SOURCE.md)
extensions/autoresearch/   the pipeline extension
agents/worker.md           the one subagent
lib/threads.ts             identification, ordering, implementation turns, the planning steer and
                           the researcher section (pure Node); every prompt the pipeline adds is here
lib/implementer.ts         loaded only into implementation turns: the stop_percent steer
lib/threads.test.ts        logic tests, model faked (node --test lib/threads.test.ts)
```

## Settings we used

`autoresearch.json`: `max_depth` 2, `max_threads` 6, `stop_percent` 60, `auto_pipeline` true.

## Outputs in `<workspace>/.harness/`

| File | What |
|---|---|
| `threads.json`, `threads.md` | The thread tree, with quotes and reasons, and the canon (`order`) |
| `implement.json`, `implement.md` | Each implementer's edits and report, in canon order |
| `implement.diff` | Every change the implementers made, against the commit they started from |
| `researcher_section.md` | Exactly what the researcher was shown at hand-over |
| `sessions/` | Every kept conversation (root continued into ordering, thread passes into implementation) |
| `state.json`, `pipeline.log` | Pipeline status (`phase`: planning or handed_over, `applied`) and its log |

Every pi process the pipeline starts carries `AUTORESEARCH_CHILD=1`; the extension does nothing in such a
process, so passes and implementers never run the pipeline themselves.

## Run it

```
npm install -g @earendil-works/pi-coding-agent@1.1.0
export OPENROUTER_API_KEY=...
cd <workspace>     # a git repository; the pipeline needs a commit to diff against
PI_CODING_AGENT_DIR=/path/to/harness pi -p --model openrouter/z-ai/glm-5.3-flash "$(cat PROMPT.md)"
```
