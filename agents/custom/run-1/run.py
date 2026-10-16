"""python run.py --question-file Q.json --data-dir DIR --answer-file OUT.json

Long-horizon agent loop over a workspace of data files and their documentation.
The model gets four tools (read, bash, edit, write) and a system prompt that
scaffolds the investigation method (inventory -> read the docs -> compute
stepwise -> verify -> commit to one answer). Hardening for tens-of-turn runs:

- Step limit: at most AGENT_MAX_TURNS model calls (default 120, env override).
  On reaching the cap the loop does not merely fall back: it makes up to two
  forced final-answer calls with tools removed ("commit to your best answer
  now"), so cap exhaustion usually still yields a real answer, not a placeholder.
  AGENT_MAX_TURNS=0 makes zero model calls (free contract plumbing check).
- Output clipping: every tool result is clipped to AGENT_TOOL_CHAR_LIMIT chars
  (default 16000, head+tail plus a notice telling the model how to narrow the
  command); bash commands have a time limit (AGENT_BASH_TIMEOUT, default 120s)
  and report their exit code. If the message history grows past
  AGENT_HISTORY_CHAR_LIMIT chars (default 240000), old tool outputs are trimmed
  so long runs do not overflow the context window.
- Fault tolerance: transient HTTP errors (429/5xx/connection/timeout) are
  retried with backoff; a request rejected as too large triggers one history
  trim and a single retry; malformed tool-call arguments are returned to the
  model as an error message instead of killing the run; an empty final message
  gets a nudge to answer as text (at most twice).
- Unanswerable gate (T5): when the model's final reply declares there is no
  valid answer, one extra turn (at most once per run, tools still available)
  asks it to verify the slice it reduced the question to — count its records
  or show the asked-for quantity is recorded nowhere — before the declaration
  becomes the answer. Skipped in the forced phase, where there is no turn
  budget left for a check.
- Contract invariants (agent/README.md: missing file, invalid JSON, non-zero
  exit, or an empty answer counts as a failed task): no exception escapes — any
  failure is contained, the answer file is written atomically as valid JSON
  with a non-empty string answer (the forced answer, else the last model text
  seen, else a fixed placeholder), and the process still exits 0. SIGTERM
  (e.g. a harness timeout) takes the same contained path.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

TOOLS = [
    {"type": "function", "function": {
        "name": "read", "description": "Read a file and return its contents (large files are clipped; prefer a bash command that prints a sample).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "bash", "description": "Run a bash command in the workspace and return its output (clipped; keep commands under two minutes).",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "edit", "description": "Replace one exact occurrence of oldText with newText in a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "oldText": {"type": "string"},
                                                        "newText": {"type": "string"}},
                       "required": ["path", "oldText", "newText"]}}},
    {"type": "function", "function": {
        "name": "write", "description": "Create or overwrite a file with the given content.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
]

# Non-empty last resort: an empty answer is a failed task per the contract, so even a
# contained total failure must emit something.
NO_ANSWER = "No answer was produced."

SYSTEM = """You are a careful data analyst answering one quantitative question about a workspace of files. You work only by calling tools; the workspace is your current working directory.

Method:
1. Inventory first: list the workspace and read the README/documentation files before computing. The documentation defines what the question's terms mean — which rows count, which units, which time windows, which fields are missing. A computation that ignores the documented definitions is wrong even when the arithmetic is right.
2. Work step by step: identify the files involved, the filters and joins the definitions require, and compute with python3 (pandas and numpy are installed) in small steps, printing intermediate counts and a few sample rows so you can see what each step did. If a file is in a format pandas cannot parse here (no openpyxl/pyarrow), fall back to the standard library: JSONL is line-delimited JSON, .xlsx is a zip of XML sheets (zipfile + xml.etree), sqlite databases open with the sqlite3 module.
3. Check intermediate results: after each step, sanity-check it (row counts vs expectations from the docs, plausible value ranges, both sides of a join non-empty). If a result contradicts the documentation or your expectation, stop and re-derive it before moving on.
4. Verify before answering: re-derive the key number a second, independent way (a different pandas expression, or pure python over the raw rows) and compare, and check that the count of records entering the computation matches the documented rules. Answer only when the two derivations agree and you can defend the chain from files to number.
5. Then reply with ONLY the final answer itself, in exactly the format the question requests — no tool call, no commentary.

Some questions have no valid answer in the workspace, and recognizing that is part of the job: declaring it is a correct result, not a failure. Before computing anything, identify the physical quantity the question asks for and confirm the workspace actually records that quantity and unit (check the files' columns and the documentation): a workspace full of related data does not answer a question about a quantity it does not record — e.g. a question about gas when only electricity is metered, or wind speed when only temperatures are recorded. Conclude there is no valid answer only on evidence, never to escape hard work, and only after you have checked the slice the question reduces to under the documentation's definitions: (a) the quantity the question asks for is recorded nowhere in the workspace (you checked the files' columns and the documentation), or (b) the question's filters, applied exactly as documented, reduce to zero records (you counted them), or (c) the question's premise contradicts the workspace or its documentation. Run one explicit check of that slice (count its records, or show the field is absent everywhere) before you declare; if the check instead finds qualifying data, answer the question. Do not substitute a nearby quantity — a different month, station, region, or measure — for the one asked, and never invent or estimate a number to avoid declaring. Declaring no valid answer when a real answer exists is as wrong as inventing one. If the answer format prescribes a phrase for this case, use that phrase exactly; otherwise reply exactly: no valid answer

Tool discipline: you have a limited number of turns, so be economical — batch independent shell commands into one call, never print whole data files (use head/sed -n/tail or python that prints only the fields you need; bulk work belongs in python). Large tool outputs are clipped and commands have a time limit. Treat the workspace as read-only evidence: never modify, overwrite, or delete files in it; write any scratch files under /tmp instead."""

NUDGE = ("Your previous reply was empty. Reply now with your final answer to the original "
         "question as plain text only — no tool calls.")

# Evidence gate (T5.3): fires once when a final reply declares unanswerability,
# so the declaration must be earned, not asserted. Conservative markers: this
# only triggers one extra verification round, so missing a rare phrasing is
# cheap, but "no answer" is excluded — as a substring it collides with ordinary
# text (e.g. "window") and with hedged numeric replies.
UNANSWERABLE_MARKERS = ("no valid answer", "cannot be answered", "cannot be computed",
                        "cannot be determined", "not computable", "not answerable",
                        "insufficient data", "no data available", "no such data")


def _declares_unanswerable(text):
    low = text.lower()
    return any(m in low for m in UNANSWERABLE_MARKERS)


EVIDENCE = ("Your reply states that the question has no valid answer. Before that "
            "becomes the final answer, verify it: state the slice the question reduces "
            "to under the documentation's definitions, and run one explicit check of "
            "it — count its records, or show that the quantity it asks for is recorded "
            "nowhere in the workspace. If the check confirms the slice is empty or the "
            "quantity absent, reply again with only the final answer in the requested "
            "format: the phrase the answer format prescribes for this case, otherwise "
            "exactly: no valid answer. If the check instead finds qualifying data, "
            "continue the investigation and answer the question.")
FORCED = ("You have reached the turn limit for this task; tool use is now disabled. Write your final "
          "answer now: commit to the best answer the evidence you have already gathered supports — "
          "if it points to a value, give that value in exactly the format the question requests, and "
          "do not hedge. Reply with only the answer itself and no commentary. Only state that the "
          "question has no valid answer if the evidence you gathered actually shows that (for example "
          "the relevant field is absent everywhere, or no records qualify under the documented rules); "
          "in that case use the phrase the answer format prescribes for this case, otherwise exactly: "
          "no valid answer.")


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def clip(text, limit):
    """Head+tail clip of a tool result, with a notice telling the model how to narrow."""
    if len(text) <= limit:
        return text
    head = (limit * 2) // 3
    tail = limit - head
    notice = (f"\n...[clipped: showing first {head} and last {tail} of {len(text)} chars; "
              f"re-run with a narrower command: head, tail, sed -n ranges, or python printing "
              f"only the fields you need]...\n")
    return text[:head] + notice + text[-tail:]


def trim_history(messages, keep_last=6, keep_chars=2000):
    """In-place: shorten old tool outputs when the history grows too large.

    Only tool-message *contents* are shortened (assistant tool_calls / tool
    result pairing stays valid); recent turns are kept whole.
    """
    total = sum(len(m.get("content") or "") for m in messages)
    if total <= _env_int("AGENT_HISTORY_CHAR_LIMIT", 240000):
        return False
    tool_idxs = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    for i in tool_idxs[:-keep_last]:
        c = messages[i].get("content") or ""
        if len(c) > keep_chars:
            messages[i]["content"] = (c[:keep_chars] +
                                      "...[old tool output trimmed to save context; re-run the command if you need it]")
    return True


def run_tool(name, args, cwd):
    tool_limit = _env_int("AGENT_TOOL_CHAR_LIMIT", 16000)
    try:
        if name == "read":
            with open(os.path.join(cwd, args["path"]), encoding="utf-8", errors="replace") as f:
                return clip(f.read(), tool_limit)
        if name == "bash":
            timeout = _env_int("AGENT_BASH_TIMEOUT", 120)
            try:
                p = subprocess.run(args["command"], shell=True, cwd=cwd, capture_output=True, text=True,
                                   stdin=subprocess.DEVNULL, timeout=timeout)
                out = clip((p.stdout or "") + (p.stderr or ""), tool_limit)
                if p.returncode != 0:
                    out += f"\n[exit code: {p.returncode}]"
                return out
            except subprocess.TimeoutExpired as e:
                so = e.stdout or ""
                se = e.stderr or ""
                if isinstance(so, bytes):
                    so = so.decode("utf-8", "replace")
                if isinstance(se, bytes):
                    se = se.decode("utf-8", "replace")
                return clip(so + se, tool_limit) + f"\n[command timed out after {timeout}s and was killed]"
        if name == "edit":
            path = os.path.join(cwd, args["path"])
            with open(path, encoding="utf-8") as f:
                text = f.read()
            if text.count(args["oldText"]) != 1:
                return f"error: oldText occurs {text.count(args['oldText'])} times; it must occur exactly once"
            with open(path, "w", encoding="utf-8") as f:
                f.write(text.replace(args["oldText"], args["newText"], 1))
            return "ok"
        if name == "write":
            path = os.path.join(cwd, args["path"])
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(args["content"])
            return "ok"
        return f"error: unknown tool {name}"
    except Exception as e:  # the model gets the error text and decides what to do
        return f"error: {e}"


def chat(messages, tools=TOOLS):
    """One model call, with retries on transient failures (429/5xx/connection).

    Five attempts with growing backoff (2/5/12/30s): a real 502 burst from the
    gateway killed a practice run at three attempts (logged in results.tsv), so
    the retry budget must ride out short outages. 402/400 etc. still raise
    immediately.
    """
    body = {"model": os.environ["INNER_MODEL"], "messages": messages}
    if tools is not None:
        body["tools"] = tools
    backoff = (2, 5, 12, 30)
    for attempt in range(5):
        req = urllib.request.Request(
            os.environ["INNER_API_BASE"].rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json",
                     "authorization": "Bearer " + os.environ.get("INNER_API_KEY", "")})
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                data = json.loads(r.read())
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError("model returned no choices")
            return choices[0]["message"]
        except urllib.error.HTTPError as e:  # note: subclass of URLError, catch first
            if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(backoff[attempt])
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < 4:
                time.sleep(backoff[attempt])
                continue
            raise


def write_answer(path, answer):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".answer-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"answer": answer}, f)
        os.replace(tmp, path)  # atomic: the answer file is never left torn or invalid
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _on_term(signum, frame):
    raise KeyboardInterrupt  # contained like any other mid-run failure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question-file", required=True)  # exact contract flags; do not rename
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--answer-file", required=True)
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, _on_term)
    max_turns = _env_int("AGENT_MAX_TURNS", 120)
    answer = ""
    partial = ""  # last non-empty model text; the fallback if no final message arrives
    try:
        q = json.load(open(a.question_file, encoding="utf-8"))
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"{q['question']}\n\nAnswer format: {q.get('guidelines', '')}"}]
        nudges = 0
        gate_used = False  # the unanswerable evidence gate fires at most once per run
        for _ in range(max_turns):  # budget guard: bounds this run's spend (see BUDGET.md)
            trim_history(messages)
            try:
                msg = chat(messages)
            except urllib.error.HTTPError as e:
                if e.code not in (400, 413):  # likely context overflow: trim hard, retry once
                    raise
                print("run.py: request rejected as too large; trimming old tool outputs and retrying once",
                      file=sys.stderr)
                trim_history(messages, keep_last=3, keep_chars=800)
                msg = chat(messages)
            text = (msg.get("content") or "").strip()
            if text:
                partial = text
            messages.append(msg)
            if not msg.get("tool_calls"):
                if text:
                    # T5.3 evidence gate: an unanswerable declaration must be
                    # earned. One verification round, once per run; the forced
                    # phase (no turn budget left) answers without the gate.
                    if not gate_used and _declares_unanswerable(text):
                        gate_used = True
                        messages.append({"role": "user", "content": EVIDENCE})
                        continue
                    answer = text
                    break
                # empty final message: nudge the model to commit to an answer (bounded)
                if nudges < 2:
                    nudges += 1
                    messages.append({"role": "user", "content": NUDGE})
                    continue
                break
            for i, call in enumerate(msg["tool_calls"]):
                fn = call.get("function") or {}
                raw = fn.get("arguments") or "{}"
                if isinstance(raw, dict):
                    args = raw
                else:
                    try:
                        args = json.loads(raw)
                    except Exception:
                        messages.append({"role": "tool", "tool_call_id": call.get("id", f"call_{i}"),
                                         "content": f"error: could not parse tool arguments: {str(raw)[:200]}"})
                        continue
                messages.append({"role": "tool", "tool_call_id": call.get("id", f"call_{i}"),
                                 "content": run_tool(fn.get("name") or "unknown", args, a.data_dir)})
        else:
            # Cap reached without a final message: forced final-answer phase (tools
            # removed so the run cannot extend itself). Guarded on max_turns>0 so that
            # AGENT_MAX_TURNS=0 stays a zero-model-call plumbing check (see BUDGET.md).
            if max_turns > 0:
                print("run.py: turn cap reached; forcing a final answer", file=sys.stderr)
                messages.append({"role": "user", "content": FORCED})
                for _ in range(2):
                    msg = chat(messages, tools=None)
                    text = (msg.get("content") or "").strip()
                    if text:
                        answer = text
                        break
                    messages.append(msg)
                    if msg.get("tool_calls"):  # model tried to call tools anyway: refuse them
                        for i, call in enumerate(msg["tool_calls"]):
                            messages.append({"role": "tool", "tool_call_id": call.get("id", f"call_{i}"),
                                             "content": "tool use is disabled; write your final answer as text"})
                    else:
                        messages.append({"role": "user", "content": NUDGE})
    except (Exception, KeyboardInterrupt) as e:
        print(f"run.py: contained failure: {type(e).__name__}: {e}", file=sys.stderr)
    if not answer:
        answer = partial or NO_ANSWER  # contract: an empty answer is a failed task
    signal.signal(signal.SIGTERM, signal.SIG_IGN)  # nothing interrupts the final write
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    write_answer(a.answer_file, answer)
    # normal return => exit code 0 once the answer file is written


if __name__ == "__main__":
    main()
