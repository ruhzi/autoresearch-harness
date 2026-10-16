# Brief: build a long-horizon agent

You are a researcher. Your job is to build an agent that can take a question
about a body of files, work through those files with a shell
and Python over many steps, and commit to one answer. The agent lives in
`agent/`. The agent is to be used for long horizon reasoning tasks.

## What the agent is for

Long-horizon work over a workspace of files: investigations that take tens of
tool calls, where the agent has to find out what is in the workspace, read the
documentation that explains it, follow a chain of evidence across several
files, compute things along the way, check its own intermediate results, and
arrive at a conclusion it can defend. The workspaces are collections of data
files and their documentation; the questions are quantitative and the
documentation decides what the terms mean. The agent is given the workspace
and the question when it is run, and nothing before that. Some questions have
no valid answer in the workspace, and recognising that is part of the job.

Expect every task to be well beyond what a single model turn can do. The tasks
are the hard kind: each needs many steps, and none is a simple lookup.

## Conditions the agent runs under

- One fresh process per task, with no memory of previous tasks.
- Shell and Python 3.12, with pandas and numpy available and nothing else
  installable. No network.
- A language model it can call at `$INNER_API_BASE` with the name
  `$INNER_MODEL`, through an OpenAI-compatible chat API with tool calling. This
  is the only model it has.

## Your budget

You have 4.00 US dollars of model calls for the whole job: the calls your agent
makes to `$INNER_API_BASE` whenever you run it, whether on a practice task you
built, on a quick check, or on anything else. A single run of the agent on a
task of this genre costs on the order of a cent or two. When the budget is
spent, the endpoint returns HTTP 402 and the job is over; whatever is in
`agent/` at that moment is your deliverable.
You can read the remaining budget at any time:

    curl -s -H "authorization: Bearer $INNER_API_KEY" http://gateway:8080/inner/usage

We evaluate the deliverable afterwards on our own budget.

## How you work

State lives in this repository: `results.tsv` for your log, `practice/` for
anything you build to test with, `agent/` for the deliverable. Your context may
be reset while you work, so keep the log current enough to resume from. Write
`DONE` in the repository root when you judge that no further experiment is
worth its cost.

## What you start from

`agent/` contains a minimal working agent. It is a starting point, not a
recommendation.

## Deliverable

The directory `agent/`, honouring the command-line contract in
`agent/README.md` exactly: the flags, the answer-file format, exit code 0. Run
the contract check described there before you finish. An agent that fails it
scores zero on every task.
