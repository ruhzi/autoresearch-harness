"""python run.py --question-file Q.json --data-dir DIR --answer-file OUT.json

Contract (frozen; see README.md): this file is the entry point; the three flags
above are the only ones and are all required; whatever happens, --answer-file is
written as {"answer": "<non-empty string>"} and the process exits 0.

The model gets read, bash, edit and write tools; the loop runs whatever it asks
for and feeds the result back. Every requested tool call gets a tool message
back, even when it errors, so the conversation the endpoint sees stays valid.

Safety backstops, so the harness can never be left without an answer file (a
killed process writes nothing, and that counts as a failed task): each HTTP
request is bounded by a per-call time budget (retries included) and each bash
command has a timeout; tool results are clipped so one huge read cannot flood
the context; the loop stops after MAX_TURNS model turns or WALL_CLOCK_S seconds
(reserving time for the final call) and asks the model once more, with tools
withheld, to commit to a final answer; a reply made without any tool call is
nudged once toward actually investigating; any unexpected failure falls back to
a fixed non-empty answer. Reads only INNER_API_BASE, INNER_MODEL and
INNER_API_KEY from the environment (never INNER_TOKEN_CAP, which the harness
does not hand out) and talks to nothing but the chat endpoint.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HTTP_TIMEOUT_S = 120   # per model request
BASH_TIMEOUT_S = 120   # per bash tool call
MAX_TURNS = 150        # model turns before forcing a final answer
WALL_CLOCK_S = 540.0   # seconds before forcing a final answer; the harness kills the
                       # process at 600s (contract.py timeout=600) and a killed process
                       # writes no answer file, so this cap must bind first
SQUEEZE_RESERVE_S = 45  # wall-clock seconds kept for the final squeeze + answer write
TOOL_OUTPUT_LIMIT = 8000  # characters of one tool result kept in the conversation
FALLBACK_ANSWER = "Unable to determine an answer from the provided files."

# The chat endpoint is this agent's only network destination. urllib would
# otherwise honour http_proxy/https_proxy environment variables; an opener with
# an empty ProxyHandler pins the call to a direct connection to $INNER_API_BASE,
# whatever the surrounding environment sets.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# What the model is told about the fixed restricted runtime, so it never depends
# on packages, network or workspace writes the environment does not provide, and
# about how to work an unseen workspace (T3): the agent is handed the workspace
# and the question and nothing else, so everything task-specific must come from
# the files at run time — the terms from their documentation, the answer shape
# from the per-task guidelines, and "no valid answer" from the evidence.
SYSTEM_NOTE = (
    "You are answering one question about a workspace of files you have never "
    "seen, using bash and Python 3.12. pandas and numpy are available; nothing "
    "else is installed and nothing can be installed. There is no network beyond "
    "this chat API. The workspace directory is read-only, so write any scratch "
    "files under /tmp.\n"
    "Method: you know nothing about this workspace in advance, so discover it "
    "before answering - list the files, read the documentation that explains "
    "them, and let that documentation decide what every term in the question "
    "means; never assume file names, columns or schemas. Tool output longer "
    "than a few thousand characters is truncated, so inspect large files with "
    "targeted commands (pandas summaries, head, wc, grep) instead of dumping "
    "them whole. Verify each intermediate result before you build on it. Some "
    "questions have no valid "
    "answer in the workspace: before you conclude that, look carefully - check "
    "the files and their documentation rather than giving up early; then, if "
    "the evidence genuinely shows the workspace does not contain what is asked, "
    "say so plainly instead of guessing - use the exact fallback phrase the "
    "required answer format gives if it has one, otherwise the words 'no valid "
    "answer'. Whatever the question, your final reply must follow the required "
    "answer format exactly, whatever it says.")

TOOLS = [
    {"type": "function", "function": {
        "name": "read", "description": "Read a file and return its contents.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "bash", "description": "Run a bash command in the workspace and return its output.",
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


def to_text(content):
    """Coerce an API message's content (string or list of parts) to a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict):
                out.append(str(part.get("text", "")))
        return "".join(out)
    return str(content)


def run_tool(name, args, cwd):
    try:
        if name == "read":
            with open(os.path.join(cwd, args["path"]), encoding="utf-8", errors="replace") as f:
                return f.read()
        if name == "bash":
            p = subprocess.run(args["command"], shell=True, cwd=cwd, capture_output=True, text=True,
                               stdin=subprocess.DEVNULL, timeout=BASH_TIMEOUT_S)
            return p.stdout + p.stderr
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
    except subprocess.TimeoutExpired:
        return f"error: command exceeded its {BASH_TIMEOUT_S}s timeout"
    except Exception as e:  # the model gets the error text and decides what to do
        return f"error: {e}"


def clip(text, limit=TOOL_OUTPUT_LIMIT):
    """Bound one tool result so a single huge read can never flood the context."""
    text = str(text)
    if len(text) <= limit:
        return text
    head, tail = text[:limit * 2 // 3], text[-limit // 3:]
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n[output truncated: {omitted} characters omitted; use a more targeted command]\n{tail}"


def chat(messages, tools=TOOLS, budget=HTTP_TIMEOUT_S):
    """One model call, bounded to at most `budget` seconds overall. Transient
    failures (timeouts, 5xx, 429) are retried inside that budget; a client error
    such as 402 out-of-budget is raised immediately."""
    body = {"model": os.environ["INNER_MODEL"], "messages": messages}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        os.environ["INNER_API_BASE"].rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json",
                 "authorization": "Bearer " + os.environ.get("INNER_API_KEY", "")})
    last = None
    start = time.monotonic()
    for attempt in range(3):
        left = budget - (time.monotonic() - start)
        if left < 5:
            break  # no time for another attempt; the caller still has its reserve
        try:
            with OPENER.open(req, timeout=min(HTTP_TIMEOUT_S, left - 4)) as r:
                return json.loads(r.read())["choices"][0]["message"]
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429:
                raise  # 402 budget spent, 401 bad key, 400 bad request: a retry cannot help
            last = e
        except Exception as e:  # timeout, refused connection, truncated body: worth one more try
            last = e
        pause = 2 * (attempt + 1)
        left = budget - (time.monotonic() - start)
        if left > pause + 5:
            time.sleep(pause)
    raise last if last is not None else TimeoutError("chat budget exhausted")


def squeeze(messages, budget=HTTP_TIMEOUT_S):
    """One last tool-free request asking the model to commit to a final answer."""
    try:
        msg = chat(messages + [{"role": "user",
                                "content": "Stop investigating. Give your final answer now, as plain "
                                           "text, following the required answer format you were given."}],
                   tools=None, budget=budget)
        return to_text(msg.get("content"))
    except Exception:
        return ""


def write_answer(path, answer):
    """Write {"answer": <non-empty string>} atomically, so a partial file can
    never be observed."""
    answer = to_text(answer).strip() or FALLBACK_ANSWER
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"answer": answer}, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--answer-file", required=True)
    a = ap.parse_args()
    answer = ""
    deadline = time.monotonic() + WALL_CLOCK_S
    try:
        with open(a.question_file, encoding="utf-8") as f:
            q = json.load(f)
        if not isinstance(q, dict):
            q = {}
        question = to_text(q.get("question")).strip()
        if not question:
            raise ValueError("question file has no question")
        guidelines = to_text(q.get("guidelines", "")).strip()
        # T3.5: the guidelines are per-task and may demand any format, so they are
        # framed as the authority over the final reply, never as one fixed shape.
        content = question
        if guidelines:
            content += f"\n\nRequired answer format (follow exactly): {guidelines}"
        messages = [{"role": "system", "content": SYSTEM_NOTE},
                    {"role": "user", "content": content}]
        turns = 0
        tools_used = 0
        nudged = False
        while turns < MAX_TURNS and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining < SQUEEZE_RESERVE_S + 10:
                break  # keep the reserve for the final squeeze + answer write
            turns += 1
            msg = chat(messages, budget=remaining - SQUEEZE_RESERVE_S)
            messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                answer = to_text(msg.get("content"))
                # Guard against a reply made without any investigation: one nudge,
                # then whatever the model answers next is accepted.
                if tools_used == 0 and not nudged and turns < MAX_TURNS - 1 \
                        and deadline - time.monotonic() > SQUEEZE_RESERVE_S + 20:
                    nudged = True
                    answer = ""
                    messages.append({"role": "user",
                                     "content": "You have not looked at the workspace yet. Use your tools to "
                                                "inspect the files and their documentation, then give your final "
                                                "answer only once you have it, following the required answer format."})
                    continue
                break
            for call in calls:
                try:
                    fn = call.get("function") or {}
                    args = json.loads(fn.get("arguments") or "{}")
                    if not isinstance(args, dict):
                        args = {}
                    result = run_tool(fn.get("name"), args, a.data_dir)
                except Exception as e:
                    result = f"error: {e}"
                tools_used += 1
                messages.append({"role": "tool", "tool_call_id": call.get("id"),
                                 "content": clip(result)})
        if not answer.strip():
            left = deadline - time.monotonic()
            if left > 8:
                answer = squeeze(messages, budget=left - 6)
    except Exception:
        answer = ""  # any unexpected failure falls through to the guaranteed write below
    write_answer(a.answer_file, answer)
    sys.exit(0)


if __name__ == "__main__":
    main()
