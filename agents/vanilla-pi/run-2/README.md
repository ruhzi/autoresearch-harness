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
