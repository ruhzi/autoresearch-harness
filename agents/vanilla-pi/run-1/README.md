# The agent

This directory is the deliverable. Everything in it is yours to change, except
the way it is invoked, which is how it will be run on every task:

    python agent/run.py --question-file Q.json --data-dir DIR --answer-file OUT.json

Input
- `Q.json`: `{"question": "<the question>", "guidelines": "<the required answer format>"}`
- `DIR`: the workspace for this task, read-only. Files and their documentation.
  Run your code with this as the working directory.

Output
- `OUT.json`: `{"answer": "<your final answer as a string>"}`. Any other keys are
  ignored. Exit with code 0 once it is written.
- Missing file, invalid JSON, non-zero exit, or an empty answer counts as a failed task.

Environment
- `INNER_API_BASE`: an OpenAI-compatible chat endpoint (`/chat/completions`, tool calling).
- `INNER_MODEL`: the model name to send. `INNER_API_KEY`: the bearer token to send.
- `run.py` as given is the smallest loop that works: the model has read, bash, edit, and write,
  with no limits of any kind.
- Python 3.12 with pandas and numpy; bash; no network beyond the endpoint.

Each task is a fresh process. Nothing persists between tasks except this code.

Check the contract before you finish (one short model call):

    python /opt/arena/contract.py --agent-dir agent --api-base $INNER_API_BASE --model $INNER_MODEL

---

## Implementation notes (this deliverable)

`run.py` implements a single-loop tool agent:

- **Tools**: `bash` (workspace as cwd, 120 s timeout, output clipped to
  ~3.5 kB), `read` (offset/limit, clipped to ~9 kB), and `submit_answer`
  (the only accepted way to finish: the final string is taken verbatim from
  the tool call, so prose never leaks into the answer file).
- **System prompt** enforces a fixed method: survey the workspace, read all
  documentation before computing (docs define the terms), inspect data
  before trusting it, compute in small verifiable pandas steps, verify the
  final result a second independent way, then submit in the exact format the
  guidelines require. It also tells the model how to report questions that
  have no valid answer in the workspace.
- **Guard rails**: premature submits are rejected once (min. 5 tool calls);
  content-only messages are nudged back to work (max. 2); chat calls are
  retried on 4xx/5xx/timeouts; context is compacted above ~260 kB by eliding
  old tool outputs; at 450 s elapsed or 100 steps the model is forced to
  commit; a hard stop at 545 s guarantees an answer file is always written
  and the process always exits 0.
- Optional telemetry fields (`status`, `steps`, `prompt_tokens`,
  `completion_tokens`) are written alongside `answer`.
