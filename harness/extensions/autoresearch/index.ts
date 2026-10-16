/**
 * The autoresearch pipeline, run once before the researcher's first turn, then handed to the researcher
 * after it has planned on its own.
 *
 * 1. Identify load-bearing threads of the task (the researcher's own first prompt) to `max_depth`, then
 *    have the root conversation order the first-level threads chronologically (the canon).
 * 2. Implement, in a git worktree of the workspace: each first-level thread's conversation gets one more
 *    turn, one thread at a time in canon order, each given the reports of the threads before it
 *    (lib/implementer.ts steers an implementer to report at `stop_percent` context). The worktree's changes
 *    are saved as implement.diff and the worktree is removed.
 * 3. The researcher starts with the plain task plus one instruction: plan first, change nothing yet. Edits
 *    are refused during that turn. When it replies without tool calls (its plan), the diff is applied to the
 *    workspace and the next turn tells it what was built and that it is the final reviewer and integrator.
 *
 * Until the hand-over, everything the pipeline produces (tree, reports, diff, worktree, sessions, log) is
 * kept in a pending directory under the OS temp dir, outside the workspace, so the planning turn cannot
 * reach it even through a recursive grep or find; only state.json lives in <cwd>/.harness then. At the
 * hand-over (or if the pipeline fails) the pending directory is moved into <cwd>/.harness, where it is
 * committed with the snapshots.
 *
 * Settings: ../../autoresearch.json. Outputs: <cwd>/.harness/. Every pi process this spawns carries
 * AUTORESEARCH_CHILD=1, so passes and implementers never run the pipeline themselves.
 */
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
	identifyTree, implementThreads, orderThreads, piImplementRunner, piPassRunner, planningMessage, renderImplementReport,
	researcherSection, saveTree, touchesHarness, type ImplementResult, type ThreadTree,
} from "../../lib/threads.ts";

const CHILD = "AUTORESEARCH_CHILD";
const PLANNING_TOOL_RESULT_CAP = 40; // hand over anyway after this many tool results without a plan
const BLOCK_REASON = "Not yet: reply first with your thinking. You can change things after that.";
const HARNESS_REASON = "Not yet: .harness is not part of the repository you are planning for. Reply with your plan first.";
const PENDING_PREFIX = "autoresearch-pending-"; // the pending directory's basename starts with this

interface Config {
	auto_pipeline: boolean;
	max_depth: number;
	max_threads: number;
	stop_percent: number;
}

interface State {
	pipeline?: "done" | "failed";
	error?: string;
	build?: "worktree" | "in_place"; // where the implementers worked
	phase?: "planning" | "handed_over";
	delivered_after?: string;
	applied?: boolean; // the build diff was applied to the workspace at hand-over
	apply_error?: string;
}

function loadConfig(): Config {
	const defaults: Config = { auto_pipeline: false, max_depth: 2, max_threads: 6, stop_percent: 40 };
	try {
		return { ...defaults, ...JSON.parse(fs.readFileSync(path.join(getAgentDir(), "autoresearch.json"), "utf-8")) };
	} catch {
		return defaults;
	}
}

/** How to start another pi like this one: the same runtime and CLI script, else `pi` on PATH. */
function piCommand(): string[] {
	const script = process.argv[1];
	return script && fs.existsSync(script) ? [process.execPath, script] : ["pi"];
}

export default function (pi: ExtensionAPI) {
	if (process.env[CHILD]) return;
	const cfg = loadConfig();
	if (!cfg.auto_pipeline) return;

	let planningToolResults = 0;
	const dir = (cwd: string) => path.join(cwd, ".harness");
	// Where the pipeline's output waits until the hand-over: outside the workspace, keyed by the workspace path.
	const pending = (cwd: string) =>
		path.join(os.tmpdir(), PENDING_PREFIX + createHash("sha1").update(path.resolve(cwd)).digest("hex").slice(0, 12));
	const statePath = (cwd: string) => path.join(dir(cwd), "state.json");
	const readState = (cwd: string): State => {
		try {
			return JSON.parse(fs.readFileSync(statePath(cwd), "utf-8"));
		} catch {
			return {};
		}
	};
	const writeState = (cwd: string, s: State) => {
		fs.mkdirSync(dir(cwd), { recursive: true });
		fs.writeFileSync(statePath(cwd), JSON.stringify(s, null, 2) + "\n", "utf-8");
	};
	/** The pipeline's artifact directory: pending until the hand-over, .harness after it. */
	const art = (cwd: string) => (readState(cwd).phase === "handed_over" ? dir(cwd) : pending(cwd));
	/** Move the pending artifacts into .harness: at the hand-over, or when the pipeline fails. */
	const land = (cwd: string) => {
		const from = pending(cwd);
		if (!fs.existsSync(from)) return;
		fs.mkdirSync(dir(cwd), { recursive: true });
		fs.cpSync(from, dir(cwd), { recursive: true, force: true });
		fs.rmSync(from, { recursive: true, force: true });
	};
	const log = (cwd: string, msg: string) => {
		const d = art(cwd);
		fs.mkdirSync(d, { recursive: true });
		fs.appendFileSync(path.join(d, "pipeline.log"), `${new Date().toISOString()} ${msg}\n`, "utf-8");
	};
	const gitIn = (where: string) => (...a: string[]) =>
		execFileSync("git", ["-C", where, ...a], { encoding: "utf-8", stdio: ["ignore", "pipe", "pipe"] });

	async function runPipeline(cwd: string, prompt: string, model: string | undefined): Promise<void> {
		const git = gitIn(cwd);
		const out = pending(cwd);
		fs.rmSync(out, { recursive: true, force: true });
		fs.mkdirSync(out, { recursive: true });
		let base = "";
		try {
			base = git("rev-parse", "HEAD").trim();
		} catch {
			/* not a git repo, or no commit yet: build in place, no diff */
		}
		// Where the implementers work: a detached worktree of HEAD, outside the workspace, so the researcher's
		// planning turn sees the workspace as it was; in place when there is no commit to build a worktree from.
		const buildDir = path.join(out, "build");
		let build: State["build"] = "in_place";
		if (base) {
			try {
				git("worktree", "prune");
				git("worktree", "add", "--detach", buildDir, "HEAD");
				build = "worktree";
			} catch (e) {
				log(cwd, `worktree failed (${(e as Error).message.trim().slice(-200)}); building in place`);
			}
		}
		const workDir = build === "worktree" ? buildDir : cwd;
		const common = {
			cwd: workDir,
			piCommand: piCommand(),
			model,
			sessionDir: path.join(out, "sessions"),
			env: { ...process.env, [CHILD]: "1" },
		};
		fs.mkdirSync(common.sessionDir, { recursive: true });
		log(cwd, `pipeline: identifying threads (building ${build === "worktree" ? "in a worktree" : "in place"})`);
		const runPass = piPassRunner(common);
		const tree: ThreadTree = await identifyTree({
			task: prompt,
			runPass,
			maxDepth: cfg.max_depth,
			maxThreads: cfg.max_threads,
			sessionFor: (n) => (n.depth === 1 ? `thread-${n.id}` : undefined),
			rootSessionId: "root",
			onEvent: (m) => log(cwd, m),
		});
		if (tree.root_status === "failed") {
			saveTree(out, tree);
			if (build === "worktree") {
				git("worktree", "remove", "--force", buildDir);
				git("worktree", "prune");
			}
			throw new Error(`root pass failed: ${tree.root_error}`); // the researcher then runs plain, with no steer
		}
		await orderThreads({ tree, runPass, onEvent: (m) => log(cwd, m) });
		saveTree(out, tree);
		log(cwd, "pipeline: implementing, one thread at a time");
		const results: ImplementResult[] = await implementThreads({
			tree,
			runImplement: piImplementRunner({
				...common,
				extensionPath: path.join(getAgentDir(), "lib", "implementer.ts"),
				stopPercent: cfg.stop_percent,
			}),
			onEvent: (m) => log(cwd, m),
		});
		fs.writeFileSync(path.join(out, "implement.json"), JSON.stringify(results, null, 2) + "\n", "utf-8");
		fs.writeFileSync(path.join(out, "implement.md"), renderImplementReport(results), "utf-8");
		if (base) {
			const g = gitIn(workDir);
			g("add", "-A");
			// DONE is the researcher's own end-of-job signal to the runner; an implementer creating it must not end
			// the job at hand-over, so it never travels in the diff.
			const diff = g("diff", "--cached", "--binary", base, "--", ".", ":(exclude).harness", ":(exclude)DONE");
			g("reset", "-q");
			fs.writeFileSync(path.join(out, "implement.diff"), diff, "utf-8");
			if (build === "worktree") {
				git("worktree", "remove", "--force", buildDir);
				git("worktree", "prune");
			}
		}
		const prefixes = [`${workDir}/`, `${workDir.replace(/\\/g, "/")}/`, `${cwd}/`, `${cwd.replace(/\\/g, "/")}/`, "/work/.harness/build/", "/work/"];
		fs.writeFileSync(path.join(out, "researcher_section.md"), researcherSection(tree, results, prefixes) + "\n", "utf-8");
		writeState(cwd, { pipeline: "done", build, phase: "planning" });
		log(cwd, "pipeline: done");
	}

	/** Land the pipeline's output, apply the build diff to the workspace and tell the researcher what was built. */
	function handOver(cwd: string, after: string, deliverAs: "steer" | "followUp"): void {
		const s = readState(cwd);
		const next: State = { ...s, phase: "handed_over", delivered_after: after };
		land(cwd); // from here on the pipeline's output is part of the repository
		const diffPath = path.join(dir(cwd), "implement.diff");
		if (s.build === "worktree") {
			const diff = fs.existsSync(diffPath) ? fs.readFileSync(diffPath, "utf-8") : "";
			if (diff.trim()) {
				try {
					gitIn(cwd)("apply", "--whitespace=nowarn", diffPath);
					next.applied = true;
				} catch (e) {
					next.applied = false;
					next.apply_error = (e as Error).message.trim().slice(-400);
				}
			} else {
				next.applied = true; // nothing to apply
			}
		}
		writeState(cwd, next);
		log(cwd, `handed over after ${after}${next.applied === false ? ` (apply failed: ${next.apply_error})` : ""}`);
		let section: string;
		try {
			section = fs.readFileSync(path.join(dir(cwd), "researcher_section.md"), "utf-8");
		} catch {
			section = `Note: the record of the earlier work on this task was lost before it could be handed to you
(the build's diff, reports and thread list are missing). Carry on with the job as the brief describes.\n`;
		}
		if (next.applied === false) {
			section += `\nNote: applying their changes to the working tree failed (${next.apply_error}). The changes are in
.harness/implement.diff; apply what you judge right by hand.\n`;
		}
		pi.sendUserMessage(section, { deliverAs });
	}

	pi.on("before_agent_start", async (event, ctx) => {
		const cwd = ctx.cwd;
		let s = readState(cwd);
		if (!s.pipeline) { // once per job; later harness calls resume the same job
			try {
				await runPipeline(cwd, event.prompt, ctx.model ? `${ctx.model.provider}/${ctx.model.id}` : undefined);
			} catch (e) {
				log(cwd, `pipeline: failed: ${(e as Error).stack ?? e}`);
				writeState(cwd, { pipeline: "failed", error: (e as Error).message });
				land(cwd); // the researcher runs plain; nothing to keep from it
			}
			s = readState(cwd);
		}
		if (s.pipeline === "done" && s.phase === "planning") {
			planningToolResults = 0;
			return { message: { customType: "autoresearch-plan", content: planningMessage(), display: true } };
		}
	});

	// During the planning turn, edits are refused, and so is any look at the pipeline's output: it waits in the
	// pending directory outside the workspace, and the plan is only worth collecting if it was written without it.
	pi.on("tool_call", async (event, ctx) => {
		const s = readState(ctx.cwd);
		if (s.pipeline !== "done" || s.phase !== "planning") return;
		if (event.toolName === "edit" || event.toolName === "write") return { block: true, reason: BLOCK_REASON };
		// The pending directory lives under the temp dir, so during planning the temp dir as a whole is off limits:
		// a glob or recursive grep there would otherwise reach the pipeline's output without naming it.
		const input = JSON.stringify(event.input ?? "");
		const tmp = os.tmpdir().replace(/\\/g, "/");
		// input is JSON text, so a Windows backslash appears doubled; fold both forms to forward slashes before comparing
		if (touchesHarness(event.input) || input.includes(PENDING_PREFIX) || input.replace(/\\\\/g, "/").replace(/\\/g, "/").includes(tmp) || /(^|[\s"'`=:(])\/tmp(?=$|[\s"'`/*;)])/.test(input)) {
			return { block: true, reason: HARNESS_REASON };
		}
	});

	// The thinking turn ends with a reply that calls no tools: hand over, and continue the same run.
	pi.on("turn_end", async (event, ctx) => {
		const cwd = ctx.cwd;
		const s = readState(cwd);
		if (s.pipeline !== "done" || s.phase !== "planning") return;
		const m = event.message as { role?: string; stopReason?: string; content?: { type: string }[] };
		if (m.role !== "assistant" || m.stopReason === "error" || m.stopReason === "aborted") return;
		const calledTools = (m.content ?? []).some((c) => c.type === "toolCall");
		planningToolResults += event.toolResults.length;
		if (!calledTools) handOver(cwd, "it replied with its plan", "followUp");
		else if (planningToolResults >= PLANNING_TOOL_RESULT_CAP) handOver(cwd, `${planningToolResults} tool results without a plan`, "steer");
	});
}
