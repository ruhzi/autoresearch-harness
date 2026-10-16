# autoresearch-harness

Two autoresearchers, the brief they were given, and the four agents they built.

An *autoresearcher* here is a coding-agent harness that is handed a research brief
and a model budget and left alone to build a deliverable. We compared two:

- **vanilla pi**: [pi](https://github.com/badlogic/pi-mono) 1.1.0 as shipped, with its stock prompts.
- **custom**: the same pi 1.1.0 plus one extension (`harness/`) that makes the researcher
  take cognizance of the entire setup before it implements anything.

Both were given `BRIEF.md`: build a long-horizon agent that answers quantitative
questions over a workspace of data files and their documentation, with $4.00 of
inner-model calls to spend on practice runs. Each autoresearcher was run twice from
scratch, so each produced two agents. All four agents are in `agents/`, exactly as
delivered.

## Layout

```
BRIEF.md                 the task both autoresearchers received, verbatim
harness/                 the custom autoresearcher: a pi agent directory (PI_CODING_AGENT_DIR)
vanilla-pi/README.md     how the vanilla arm was run
agents/
  vanilla-pi/run-1/      agent built by vanilla pi, first run
  vanilla-pi/run-2/      agent built by vanilla pi, second run
  custom/run-1/          agent built by the custom harness, first run
  custom/run-2/          agent built by the custom harness, second run
```

Each agent directory is the delivered `agent/`: a `README.md` stating the command-line
contract and the researcher's own implementation notes, and `run.py`.

## How the custom harness works

Once, before the researcher's first turn, the extension runs a fixed sequence of passes
on the brief:

1. **Root pass.** Deduces the brief's load-bearing *threads*: the parts of the setup the
   build depends on. Each thread quotes the brief's own words and says why it bears load.
   One further pass per thread breaks it into sub-threads. The depth of this fan-out is
   configurable (`harness/autoresearch.json`); we used depth 2 and at most 6 threads per pass.
2. **Chronology.** The root conversation orders the threads by dependency.
3. **Implementation.** In that order, each thread's conversation implements its
   sub-threads in a git worktree, seeing the reports of the threads before it.
4. **Planning turn.** Meanwhile the researcher itself reads the brief and reasons out
   loud about what it would build, with edits refused.
5. **Hand-over.** The implementers' diff is applied and the researcher is made the final
   reviewer and integrator: hold the build to its own plan, change what conflicts. From
   there the run is an ordinary pi session until the budget is spent.

`harness/README.md` has the details, the prompts are in `harness/lib/threads.ts`, and the
logic tests run with `node --test harness/lib/threads.test.ts`.

## Models

| Role | Model (OpenRouter slug) |
|---|---|
| Outer: the autoresearcher itself | GLM 5.3 Flash, `z-ai/glm-5.3-flash` |
| Inner: what the built agent calls | GPT-6 Luna, `openai/gpt-6-luna` |

The outer model was the same in both arms. The inner budget was $4.00 per run. In our runs
both endpoints were reached through a local metering proxy that pinned the model and
provider; `harness/models.json` in this release points at OpenRouter directly.

## Results

The agents were scored on 75 tasks from the *hard* level of
[DABstep](https://arxiv.org/abs/2506.23719) (Egg et al., 2025), a benchmark of multi-step
data-analysis questions, graded by exact match against the reference answer. During
evaluation each agent ran one fresh process per task on GPT-6 Luna with no network. Each
run is one complete run of the outer autoresearcher.

| Autoresearcher | Run 1 | Run 2 | Mean |
|---|---|---|---|
| vanilla pi | 40.0 | 36.0 | 38.0 |
| custom | 44.0 | 48.0 | 46.0 |

Pass rate in percent over the 75 tasks.

## Running an agent

Each agent honours the contract in its own `README.md`:

```
python run.py --question-file Q.json --data-dir DIR --answer-file OUT.json
```

with `INNER_API_BASE`, `INNER_MODEL` and `INNER_API_KEY` set to an OpenAI-compatible chat
endpoint with tool calling. Python 3.12 with pandas and numpy.

## Running the custom harness

```
npm install -g @earendil-works/pi-coding-agent@1.1.0
export OPENROUTER_API_KEY=...
cd <your workspace>            # a git repository containing the brief
PI_CODING_AGENT_DIR=/path/to/harness pi -p --model openrouter/z-ai/glm-5.3-flash "$(cat PROMPT.md)"
```

The pipeline runs once, at the start of the first turn, on the prompt it is given. Its
outputs land in `<workspace>/.harness/`. See `vanilla-pi/README.md` for the prompt we used.

## Not in this release

The sandbox, the metering gateway, the evaluator and the DABstep split are not part of
this release, and neither are the model-call logs of the runs.

## License

MIT. `harness/extensions/subagent/` is copied unchanged from pi's own examples
(MIT, Mario Zechner); see `harness/extensions/subagent/SOURCE.md`.
