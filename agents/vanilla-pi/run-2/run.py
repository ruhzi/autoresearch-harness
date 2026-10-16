"""python run.py --question-file Q.json --data-dir DIR --answer-file OUT.json

Long-horizon investigation agent. The model explores a read-only workspace of data
files and documentation with read/bash/write tools, computes with Python, verifies
intermediate results, then commits to one final answer. Built for quantitative
questions whose terms are defined by workspace documentation, including questions
the workspace cannot answer.

Beyond the minimal loop:
- system prompt with an investigation method and a strict FINAL ANSWER: contract
- per-tool output clipping and conversation compaction for long runs
- step cap, wall-clock cap, hard deadline, forced commit phase (tools withdrawn)
- API retries with backoff; graceful fallback to a transcript scan for an answer
- always writes the answer file and exits 0
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

API_BASE = os.environ.get("INNER_API_BASE", "").rstrip("/")
MODEL = os.environ.get("INNER_MODEL", "")
API_KEY = os.environ.get("INNER_API_KEY", "")

SCRATCH = "/tmp/agent_work"
MAX_STEPS = 40            # model calls while investigating
LOOP_TIME_BUDGET = 330.0  # seconds of investigating before the commit phase
HARD_DEADLINE_S = 540.0   # absolute wall-clock cap for all model work
CALL_TIMEOUT_S = 150.0    # per API attempt cap (clamped to the deadline)
TOOL_CLIP = 9000          # chars kept from one tool result
BASH_TIMEOUT_S = 150

DEADLINE = float("inf")   # set in main()

TOOLS = [
    {"type": "function", "function": {
        "name": "read",
        "description": "Read a text file (documentation, code, small data files). Very long output is clipped; for large data files prefer bash (head/tail/grep/python).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "bash",
        "description": "Run one bash command in the workspace directory and get combined stdout+stderr (clipped). Can run a whole python3 script via a heredoc. Commands time out after 150s.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "write",
        "description": "Write a scratch file. Relative paths land in /tmp/agent_work (the workspace is read-only); run written scripts by absolute path. Absolute paths outside the workspace are kept.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
]

SYSTEM_TEMPLATE = """You are a rigorous data investigator. You have shell access to a read-only workspace
containing data files and documentation that explains them. You will be asked one
quantitative question about the workspace. Investigate it thoroughly with the tools,
then commit to one final answer.

Where you are: the workspace is your current working directory. Never create, modify, or
delete anything inside it. For scratch files use /tmp/agent_work (the write tool puts
relative paths there automatically).

Available: bash, Python 3.12 with pandas and numpy. No network access.

Method:
1. Survey the workspace: list all files with sizes.
2. Read every documentation file in full (READMEs, codebooks, data dictionaries, notes).
   The documentation defines what the terms of the question mean. Its definitions, units,
   exclusion rules, and caveats override your own assumptions. Reread the relevant part
   before you finalize your answer.
3. Inspect each data file before using it: shape, columns, dtypes, sample rows, distinct
   values of the columns you will filter or group by. Sample large files (head, tail,
   grep, python) instead of dumping them into the conversation.
4. Compute with Python, ideally in one bash call running a `python3 - <<'EOF' ... EOF`
   heredoc. Print intermediate results so you can check them. Be careful with: units and
   conversions, date/time formats and time zones, missing or sentinel values, duplicate
   rows, and any rows the documentation says to exclude.
5. Verify before answering: recompute the key quantity a second, independent way (for
   example a different grouping or a spot check of a few raw rows). If the two disagree,
   find out why before answering.
6. Decide whether the workspace can answer the question at all before answering. Keep the
   distinction between "no matching rows" and "a measurable zero": if the requested
   category, channel, column, or time span does not exist in the data — or the
   documentation says it never existed — the question is UNANSWERABLE, not zero, and
   guessing or substituting a nearby quantity is wrong. In that case your answer is to
   say so plainly and briefly, following the required answer format (if the format gives
   a token for this, such as UNANSWERABLE, use exactly that token).

You have a generous step budget (about 40 tool calls). Use what you need; do not rush and
do not pad. Prefer few, dense tool calls (a whole python script per call) over many tiny
ones. If a bash call fails, read the error and fix it; do not repeat it.

Finishing is strict: when — and only when — you can defend your answer, end your message
with exactly one line of the form

FINAL ANSWER: <answer>

<answer> must be only the final answer, in the required format below: no reasoning, no
units unless the format requires them, no markdown, no quotes. Writing that line ends the
task; what follows the colon is recorded as your answer.

Required answer format:
__GUIDELINES__"""


class FatalBudget(Exception):
    """The API account is out of money; no further calls can succeed."""


def as_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                parts.append(str(b.get("text") or b.get("content") or ""))
            else:
                parts.append(str(b))
        return "\n".join(parts)
    return "" if content is None else str(content)


def clip(text, limit=TOOL_CLIP):
    text = text or ""
    if len(text) <= limit:
        return text
    omitted = len(text) - (limit - 400)
    return (text[:limit - 400]
            + f"\n\n[... output clipped: {omitted} of {len(text)} chars omitted; "
              "use head/tail/grep or a python script to sample the file ...]\n\n"
            + text[-200:])


def run_tool(name, args):
    try:
        if name == "read":
            path = str(args.get("path", ""))
            p = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
            with open(p, encoding="utf-8", errors="replace") as f:
                text = f.read()
            return clip(f"[file: {path}]\n{text}")
        if name == "bash":
            cmd = str(args.get("command", ""))
            try:
                p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                   stdin=subprocess.DEVNULL, timeout=BASH_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                return f"error: command timed out after {BASH_TIMEOUT_S}s; do the work in smaller chunks"
            out = p.stdout
            if p.stderr.strip():
                out += ("\n[stderr]\n" if out else "[stderr]\n") + p.stderr
            out = out.strip()
            if p.returncode != 0:
                out += f"\n[exit code {p.returncode}]"
            return clip(out or "(no output)")
        if name == "write":
            path = str(args.get("path", ""))
            p = os.path.realpath(path if os.path.isabs(path) else os.path.join(SCRATCH, path))
            ws = os.path.realpath(os.getcwd())
            if p == ws or p.startswith(ws + os.sep):
                return (f"error: the workspace is read-only; write scratch files under {SCRATCH} "
                        f"(e.g. path /tmp/agent_work/x.py, then run python3 /tmp/agent_work/x.py)")
            os.makedirs(os.path.dirname(p) or "/", exist_ok=True)
            data = str(args.get("content", ""))
            with open(p, "w", encoding="utf-8") as f:
                f.write(data)
            return f"ok: wrote {p} ({len(data)} bytes); run it with that absolute path"
        return f"error: unknown tool {name}"
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


def compact(messages, level):
    """Free context: level 1 trims old tool outputs; level 2 also drops old tool rounds."""
    changed = False
    if level >= 1:
        for m in messages[:-8]:
            if m.get("role") == "tool":
                c = m.get("content") or ""
                if len(c) > 500:
                    m["content"] = c[:250] + f"\n[...older tool output trimmed; was {len(c)} chars...]"
                    changed = True
    if level >= 2:
        blocks = []  # (start, end) of each assistant-with-tool_calls round, tools included
        i = 0
        while i < len(messages):
            if messages[i].get("role") == "assistant" and messages[i].get("tool_calls"):
                j = i + 1
                while j < len(messages) and messages[j].get("role") == "tool":
                    j += 1
                blocks.append((i, j))
                i = j
            else:
                i += 1
        if len(blocks) > 6:
            keep_from = blocks[len(blocks) - 6][0]
            head_end = blocks[0][0]
            new = messages[:head_end] + messages[keep_from:]
            if len(new) < len(messages):
                messages[:] = new
                changed = True
    return changed


def chat(messages, use_tools=True, attempts=5, delays=(2.0, 5.0, 10.0, 20.0)):
    payload = {"model": MODEL, "messages": messages, "temperature": 0.2}
    if use_tools:
        payload["tools"] = TOOLS
    last_err = "no attempt made"
    for attempt in range(attempts + 1):
        remaining = DEADLINE - time.time()
        if remaining < 20:
            raise RuntimeError(f"out of time before API call ({last_err})")
        timeout = max(20.0, min(CALL_TIMEOUT_S, remaining - 30.0))
        try:
            req = urllib.request.Request(
                API_BASE + "/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"content-type": "application/json",
                         "authorization": "Bearer " + API_KEY})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read().decode("utf-8"))
            return resp["choices"][0]["message"], (resp.get("usage") or {})
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", "replace")[:600]
            except Exception:
                body = ""
            if e.code == 402:
                raise FatalBudget(body)
            low = body.lower()
            if e.code in (400, 413) and ("context" in low or "length" in low or "too many token" in low):
                if compact(messages, 2):
                    last_err = f"context overflow (compacted): {body}"
                    continue
                raise RuntimeError(f"context overflow, compaction exhausted: {body}")
            if "temperature" in low and "temperature" in payload:
                del payload["temperature"]
                last_err = f"temperature rejected: {body}"
                continue
            if e.code in (429, 500, 502, 503, 504) and attempt < attempts:
                time.sleep(delays[min(attempt, len(delays) - 1)])
                last_err = f"HTTP {e.code}: {body}"
                continue
            raise RuntimeError(f"HTTP {e.code}: {body}")
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < attempts:
                time.sleep(delays[min(attempt, len(delays) - 1)])
                continue
            raise RuntimeError(last_err)
    raise RuntimeError(f"API retries exhausted: {last_err}")


MARKER_RE = re.compile(r"(?im)^[\s>*-]*\**\s*FINAL\s+ANSWER\s*\**\s*:\s*(.+?)\s*$")
PLACEHOLDERS = {"<answer>", "...", "…", "the answer", "n/a"}


def extract_answer(text):
    """Return the text after the last line-initial FINAL ANSWER: marker, else None."""
    matches = MARKER_RE.findall(as_text(text))
    if not matches:
        return None
    for cand in reversed(matches):
        cand = cand.strip().strip("*`\"' ")
        if cand and cand.lower() not in PLACEHOLDERS and "where <answer>" not in cand.lower():
            return cand
    return None


def scan_for_answer(messages):
    """Last resort: find a marker anywhere in the transcript, else last assistant text."""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            found = extract_answer(as_text(m.get("content")))
            if found:
                return found
    for m in reversed(messages):
        if m.get("role") == "assistant":
            t = as_text(m.get("content")).strip()
            if t:
                return t if len(t) <= 400 else ""
    return ""


def clean_answer(text):
    t = str(text).strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        t = t[1:-1].strip()
    return t


def main():
    global DEADLINE
    t0 = time.time()
    DEADLINE = t0 + HARD_DEADLINE_S
    ap = argparse.ArgumentParser()
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--answer-file", required=True)
    a = ap.parse_args()
    # resolve relative paths against the caller's cwd BEFORE chdir-ing into the workspace
    a.question_file = os.path.abspath(a.question_file)
    a.answer_file = os.path.abspath(a.answer_file)
    a.data_dir = os.path.abspath(a.data_dir)

    answer, status, steps, ptok, ctok = "", "ok", 0, 0, 0
    try:
        subprocess.run(["rm", "-rf", SCRATCH], check=False)
        os.makedirs(SCRATCH, exist_ok=True)
        with open(a.question_file, encoding="utf-8") as f:
            q = json.load(f)
        question = str(q.get("question", "")).strip()
        guidelines = str(q.get("guidelines", "")).strip()
        try:
            os.chdir(a.data_dir)
        except OSError:
            pass

        system = SYSTEM_TEMPLATE.replace(
            "__GUIDELINES__",
            guidelines or "(none specified: give the answer as one short plain sentence)")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": (f"Workspace (your current working directory): {os.getcwd()}\n\n"
                                         f"Question: {question}\n\n"
                                         f"Required answer format: {guidelines or '(no special format)'}\n\n"
                                         "If the workspace cannot answer the question, follow the required "
                                         "answer format for saying so.\n\n"
                                         "Begin the investigation.")},
        ]

        final = None
        warned = nudged = False
        while steps < MAX_STEPS and (time.time() - t0) < LOOP_TIME_BUDGET:
            steps += 1
            if not warned and (time.time() - t0) > LOOP_TIME_BUDGET * 0.6:
                warned = True
                messages.append({"role": "user", "content":
                                 "Note: about two thirds of the investigation budget is used. Finish any "
                                 "decisive checks, then give the FINAL ANSWER: line."})
            try:
                msg, usage = chat(messages, use_tools=True)
            except FatalBudget:
                status = "budget_exhausted"
                break
            except RuntimeError as e:
                status = f"api_error: {str(e)[:120]}"
                break
            ptok += int(usage.get("prompt_tokens") or 0)
            ctok += int(usage.get("completion_tokens") or 0)
            messages.append(msg)

            found = extract_answer(as_text(msg.get("content")))
            if found:
                final = found
                break
            calls = msg.get("tool_calls") or []
            if not calls:
                text = as_text(msg.get("content")).strip()
                if text and nudged:
                    final = text[:800]  # model insists on finishing without the marker; accept it
                    break
                nudged += 1
                messages.append({"role": "user", "content":
                                 "If you are done, reply with the FINAL ANSWER: line (only the answer "
                                 "after the colon, in the required format). If you still have work, "
                                 "continue with tools."})
                continue
            for i, call in enumerate(calls):
                try:
                    fn = call.get("function") or {}
                    raw = fn.get("arguments") or {}
                    args = raw if isinstance(raw, dict) else json.loads(raw)
                    if not isinstance(args, dict):
                        args = {}
                except (json.JSONDecodeError, TypeError) as e:
                    result = f"error: could not parse tool arguments as JSON ({e}); resend the call"
                else:
                    result = run_tool(fn.get("name", ""), args)
                messages.append({"role": "tool", "tool_call_id": call.get("id") or f"call_{steps}_{i}",
                                 "content": clip(str(result), 12000)})
            total = sum(len(as_text(m.get("content"))) for m in messages)
            if total > 350_000:
                compact(messages, 2)
            elif total > 220_000:
                compact(messages, 1)
        else:
            status = "commit"

        if final is None and not status.startswith(("budget_exhausted", "api_error")):
            status = "commit"
            messages.append({"role": "user", "content":
                             "The investigation budget is over. Commit to your answer now: reply with the "
                             "FINAL ANSWER: line — after the colon put only the answer, in the required "
                             "format. If the workspace cannot answer the question, follow the required "
                             "format for saying so."})
            try:
                msg, usage = chat(messages, use_tools=False, attempts=2, delays=(5.0,))
                ptok += int(usage.get("prompt_tokens") or 0)
                ctok += int(usage.get("completion_tokens") or 0)
                messages.append(msg)
                text = as_text(msg.get("content")).strip()
                final = extract_answer(text) or (text[:800] if text else None)
            except Exception:
                final = None
        if final is None:
            final = scan_for_answer(messages)
        answer = clean_answer(final)
    except Exception as e:
        status = f"error: {type(e).__name__}: {str(e)[:150]}"
        try:
            answer = clean_answer(scan_for_answer(messages))
        except Exception:
            answer = ""
    finally:
        answer = (answer or "").strip() or "no answer produced"
        out = {"answer": answer, "status": status, "steps": int(steps),
               "prompt_tokens": int(ptok), "completion_tokens": int(ctok)}
        os.makedirs(os.path.dirname(a.answer_file) or "/", exist_ok=True)
        with open(a.answer_file, "w", encoding="utf-8") as f:
            json.dump(out, f)
        print(json.dumps(out)[:500], file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()
