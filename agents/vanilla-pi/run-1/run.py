"""python run.py --question-file Q.json --data-dir DIR --answer-file OUT.json

A long-horizon investigation agent over a workspace of data files and their
documentation. The model drives bash/read tools with the task workspace as
working directory, is pushed to verify its intermediate results, and commits
to one final answer through a submit_answer tool.

Robustness: tool outputs are clipped, the shell has a timeout, chat calls are
retried on transient errors, context is compacted when it grows too large,
and there are step/wall-clock budgets with a forced-commit fallback so the
process always writes an answer file and exits 0.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

# ---------------- tunables ----------------
BASH_TIMEOUT = 120          # seconds per shell command
BASH_CLIP = 3500            # chars of shell output kept (head+tail)
READ_CLIP = 9000            # chars kept from one read call
MAX_CHAT_STEPS = 100        # hard cap on model calls
FORCE_SUBMIT_AT = 450.0     # seconds elapsed before we ask the model to commit
HARD_STOP_AT = 545.0        # seconds elapsed before we stop regardless
MIN_STEPS_BEFORE_SUBMIT = 5 # tool calls required before a submit is accepted
COMPACT_AT = 260_000        # total context chars before compaction
RETRY_DELAYS = [2, 4, 8, 16, 30, 45]

SYSTEM_PROMPT = """\
You are a meticulous data-investigation agent. You get a workspace of data \
files and their documentation (the working directory of your tools), and one \
question about it. Your job: work through the files step by step, verify \
your own intermediate results, and commit to ONE defensible final answer.

The workspace is read-only. Do all scratch work in /tmp.

Method - follow it on every task:
1. SURVEY. List the whole workspace (e.g. `find . -type f | head -100` plus \
`ls -la`), note file names, sizes and formats. Then read ALL documentation \
files (README, *.md, notes/, schema files, data dictionaries) in full before \
computing anything. The documentation defines what the terms in the question \
mean; it overrides your own assumptions.
2. RESTATE the question: what quantity, which exact filters (time range, \
region, category, status...), which doc definition applies. Note the required \
answer format now.
3. INSPECT the data before trusting it: for each relevant file check the \
head/tail, row count, dtypes, missing values, duplicate keys, units, and odd \
values (`head -5`, `wc -l`, a small pandas describe). Files of the same kind \
may have different conventions.
4. COMPUTE in small verifiable steps with python3 (pandas and numpy are \
installed; no other packages, no network). Print intermediate row counts and \
sums. After each filter/join check the shape is what you expect.
5. VERIFY. Recompute the final result a second, independent way (different \
grouping, different function, or a manual spot check of a few rows). If the \
two disagree, find out why before answering. Double-check the filters are \
inclusive/exclusive exactly as the documentation says, and that you used the \
right column definitions.
6. SUBMIT by calling the submit_answer tool with the final answer string, \
formatted EXACTLY as the question's guidelines require (units, rounding, \
thousands separators, "number only", decimals, etc.). The submitted string is \
graded as-is: no explanations, no sources, no restating of the question, \
unless the guidelines explicitly ask for them.

Rules:
- Everything must come from the workspace. Never answer from memory or guess.
- If documentation and intuition disagree, the documentation wins. Read it \
twice before concluding it doesn't cover something.
- Watch for traps: duplicated rows, returns/refunds/cancellations, prices in \
cents vs dollars, unit suffixes, timezone and date-boundary conventions, \
records just outside the asked window, categories whose names look similar \
but are distinct, and definitions that change over time (policy notes).
- Some questions have NO valid answer in the workspace (an entity, period or \
metric that simply does not exist, or the documentation makes the question \
ill-posed). Investigate thoroughly before concluding this. If you do \
conclude it, submit exactly: no valid answer in the workspace
  unless the guidelines specify a different way to report it.
- Be efficient: batch related shell commands into one call when sensible. \
But these tasks are long: expect to need dozens of tool calls, and a \
one-lookup answer is almost certainly wrong.
- If a command fails twice, change the approach instead of repeating it.
- Keep a mental ledger: what is established, what is still open, what the \
current best number is. Re-check it before submitting. For long \
investigations also keep a working notes file at /tmp/notes.md (facts \
established, definitions found, current best numbers) and update it as you \
go; very old tool outputs can be elided from your context, so notes are your \
memory.
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run a bash command inside the task workspace and return combined "
                       "stdout+stderr (long output is truncated). Python 3.12 with pandas and "
                       "numpy is available. Workspace data is read-only; write scratch files to /tmp.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "the bash command to run"}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read",
        "description": "Read a text file from the workspace, with optional line offset/limit. "
                       "Very long output is truncated.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "description": "1-based line number to start from"},
            "limit": {"type": "integer", "description": "maximum number of lines to return"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "submit_answer",
        "description": "Commit the final answer to the question. Before calling: re-read the "
                       "guidelines and check your string matches them exactly (units, rounding, "
                       "no extra words). Call this exactly once, after you have verified the "
                       "result by an independent second method.",
        "parameters": {"type": "object", "properties": {
            "answer": {"type": "string", "description": "the final answer string, formatted "
                                                        "exactly as the guidelines require"}},
            "required": ["answer"]}}},
]


def clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    return (text[:head] + "\n…[" + str(len(text) - max_chars) + " chars elided]…\n"
            + text[-tail:])


def run_bash(command: str, cwd: str) -> str:
    try:
        p = subprocess.run(command, shell=True, cwd=cwd, capture_output=True,
                           text=True, stdin=subprocess.DEVNULL, timeout=BASH_TIMEOUT)
        out = p.stdout + (p.stderr if p.stderr else "")
        if p.returncode != 0:
            out = out + f"\n[exit code: {p.returncode}]"
    except subprocess.TimeoutExpired as e:
        out = f"error: command timed out after {BASH_TIMEOUT}s (it was killed). " \
              f"Partial output:\n{((e.stdout or '') + (e.stderr or ''))[:1000]}"
    return clip(out, BASH_CLIP)


def run_read(path: str, cwd: str, offset=None, limit=None) -> str:
    fp = path if os.path.isabs(path) else os.path.join(cwd, path)
    if not os.path.exists(fp):
        return f"error: no such file: {path}"
    if os.path.isdir(fp):
        try:
            entries = sorted(os.listdir(fp))
            return "directory listing:\n" + "\n".join(entries[:500])
        except Exception as e:
            return f"error: {e}"
    try:
        with open(fp, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return f"error: {e}"
    start = max(1, int(offset or 1))
    sel = lines[start - 1: (start - 1) + int(limit or 400)]
    body = "".join(sel)
    note = ""
    if len(sel) < len(lines) - (start - 1) or (start - 1) > 0:
        note = f"\n[showing lines {start}-{start + len(sel) - 1} of {len(lines)} total]"
    return clip(body, READ_CLIP) + note


def content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(p.get("text", ""))
                elif "text" in p:
                    parts.append(p["text"])
        return "\n".join(parts)
    return "" if content is None else str(content)


def chat(messages, usage):
    """One model call with retries on transient errors. Raises on hard failure."""
    base = os.environ["INNER_API_BASE"].rstrip("/")
    body = {"model": os.environ["INNER_MODEL"], "messages": messages, "tools": TOOLS}
    last_err = None
    for attempt in range(len(RETRY_DELAYS) + 1):
        req = urllib.request.Request(
            base + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json",
                     "authorization": "Bearer " + os.environ.get("INNER_API_KEY", "")})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                resp = json.loads(r.read())
            msg = resp["choices"][0]["message"]
            u = resp.get("usage") or {}
            usage["prompt"] += int(u.get("prompt_tokens") or 0)
            usage["completion"] += int(u.get("completion_tokens") or 0)
            return msg
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode()[:500]
            except Exception:
                pass
            last_err = f"HTTP {e.code}: {err_body}"
            if e.code == 402:          # budget exhausted: no point retrying
                raise
            retryable = e.code in (429, 500, 502, 503, 504) or e.code >= 500
            if not retryable:
                break
        except Exception as e:         # timeouts, connection resets...
            last_err = str(e)
        if attempt < len(RETRY_DELAYS):
            time.sleep(RETRY_DELAYS[attempt])
    raise RuntimeError(f"chat failed after retries: {last_err}")


def compact(messages):
    """Elide large old tool outputs / long old assistant texts in place."""
    n = len(messages)
    for i, m in enumerate(messages):
        if n - i <= 12:
            break
        role = m.get("role")
        c = m.get("content")
        if role == "tool" and isinstance(c, str) and len(c) > 400:
            m["content"] = c[:200] + "\n…[older tool output elided; rerun the command if needed]"
        elif role == "assistant" and isinstance(c, str) and len(c) > 4000:
            m["content"] = c[:1200] + "\n…[older text elided]"


def total_chars(messages):
    t = 0
    for m in messages:
        c = m.get("content")
        t += len(c) if isinstance(c, str) else 100
        for tc in m.get("tool_calls") or []:
            t += len(json.dumps(tc))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--answer-file", required=True)
    a = ap.parse_args()
    q = json.load(open(a.question_file, encoding="utf-8"))
    data_dir = os.path.abspath(a.data_dir)
    os.makedirs(os.path.dirname(os.path.abspath(a.answer_file)) or ".", exist_ok=True)

    answer = ""
    status = "ok"
    usage = {"prompt": 0, "completion": 0}
    steps = 0
    tool_calls_made = 0
    premature_warned = False
    forced = False
    last_text = ""
    nudges = 0
    t0 = time.time()

    user_msg = (f"Question: {q['question']}\n\n"
                f"Required answer format (guidelines): {q.get('guidelines') or 'not specified'}\n\n"
                f"The workspace is the working directory of your tools. Begin with the survey "
                f"step of your method.")
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}]

    def log(*args):
        print(*args, file=sys.stderr, flush=True)

    try:
        while True:
            elapsed = time.time() - t0
            # --- budget guards ---
            if not forced and (steps >= MAX_CHAT_STEPS - 2 or elapsed > FORCE_SUBMIT_AT):
                forced = True
                messages.append({"role": "user", "content":
                    "BUDGET REACHED: you must commit now. Call submit_answer immediately with "
                    "your best verified answer (or exactly 'no valid answer in the workspace' if "
                    "that is your conclusion). Do not run any more tools."})
                log(f"[force-submit at step {steps}, {elapsed:.0f}s]")

            msg = chat(messages, usage)
            steps += 1
            messages.append(msg)
            text = content_to_text(msg.get("content")).strip()
            if text:
                last_text = text

            calls = msg.get("tool_calls") or []
            if not calls:
                if nudges < 2 and not forced:
                    nudges += 1
                    messages.append({"role": "user", "content":
                        "Do not stop here: that is not the final answer yet. Continue with your "
                        "tools, verify the result, and then call the submit_answer tool with the "
                        "final answer string (the answer only, in the required format)."})
                    log(f"[nudge {nudges} at step {steps}]")
                    continue
                answer = text
                status = "ok" if answer else "empty_final"
                log(f"[final text at step {steps}: {answer[:120]!r}]")
                break

            accepted_submit = None
            for call in calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                    if not isinstance(args, dict):
                        args = {}
                except Exception as e:
                    messages.append({"role": "tool", "tool_call_id": call["id"],
                                     "content": f"error: could not parse arguments: {e}"})
                    continue

                if name == "submit_answer":
                    if tool_calls_made < MIN_STEPS_BEFORE_SUBMIT and not premature_warned:
                        premature_warned = True
                        messages.append({"role": "tool", "tool_call_id": call["id"], "content":
                            "rejected: this is a long-horizon task; you have barely explored the "
                            "workspace. Survey it, read the documentation, compute, verify — then "
                            "call submit_answer again. If you are genuinely certain the workspace "
                            "cannot answer the question, call submit_answer again with "
                            "'no valid answer in the workspace'."})
                        continue
                    ans = content_to_text(args.get("answer")).strip()
                    if len(ans) >= 2 and ans[0] == ans[-1] and ans[0] in "\"'":
                        ans = ans[1:-1].strip()
                    if not ans:
                        messages.append({"role": "tool", "tool_call_id": call["id"],
                                         "content": "error: answer was empty; provide the answer "
                                                    "string in the required format."})
                        continue
                    accepted_submit = ans
                    messages.append({"role": "tool", "tool_call_id": call["id"],
                                     "content": "accepted."})
                    continue

                tool_calls_made += 1
                if name == "bash":
                    result = run_bash(args.get("command", ""), data_dir)
                elif name == "read":
                    result = run_read(args.get("path", ""), data_dir,
                                      args.get("offset"), args.get("limit"))
                else:
                    result = f"error: unknown tool {name}"
                if forced:
                    result += "\n[BUDGET REACHED — next, call submit_answer with your final answer.]"
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})

            if accepted_submit is not None:
                answer = accepted_submit
                log(f"[submitted at step {steps}: {answer[:120]!r}]")
                break

            if total_chars(messages) > COMPACT_AT:
                compact(messages)

            if time.time() - t0 > HARD_STOP_AT:
                status = "hard_stop"
                answer = last_text
                log(f"[hard stop at {time.time()-t0:.0f}s]")
                break
    except Exception as e:
        status = f"error: {e}"
        log("[error]", e)
        answer = answer or last_text
    finally:
        with open(a.answer_file, "w", encoding="utf-8") as f:
            json.dump({"answer": answer, "status": status, "steps": steps,
                       "prompt_tokens": usage["prompt"],
                       "completion_tokens": usage["completion"]}, f)
    sys.exit(0)


if __name__ == "__main__":
    main()
