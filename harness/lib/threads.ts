/**
 * Thread identification: break a task into load-bearing threads, then each thread into its own
 * load-bearing sub-threads, recursively, until a pass returns none or a cap is hit. Then the root
 * conversation orders the first-level threads chronologically (the canon), the threads are implemented
 * one at a time in that order, and the researcher is handed the result after it has planned on its own.
 *
 * Pure Node (no pi imports), so the extension can call it and a script can drive it directly.
 * A "pass" is one one-shot pi process with read-only tools and the identifier prompt appended.
 */
import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

export interface ThreadNode {
	id: string; // "T2.1"
	title: string;
	quote: string;
	source: string;
	why_load_bearing: string;
	depth: number; // 1 = found by the root pass
	status: "expanded" | "atomic" | "depth_cap" | "failed" | "pending";
	children: ThreadNode[];
	error?: string;
	session?: string; // session id of the pass that expanded this node, when kept (for a follow-up turn)
}

export interface ThreadTree {
	task: string;
	created: string;
	settings: { maxDepth: number; maxThreads: number; concurrency: number };
	root_status: "expanded" | "atomic" | "failed";
	root_error?: string;
	root_session?: string; // the root pass's conversation, when kept (the ordering turn continues it)
	threads: ThreadNode[];
	passes: PassRecord[];
	/** First-level thread ids in implementation order (the canon). Set by orderThreads; defaults to T1, T2, ... */
	order?: string[];
	order_source?: "model" | "fallback";
	order_error?: string;
}

export interface PassRecord {
	target: string; // "root" or a thread id
	attempt: number;
	ok: boolean;
	n_threads?: number;
	error?: string;
	seconds: number;
	cost_usd?: number;
}

export interface PassResult {
	text: string; // the final assistant text
	cost_usd?: number;
}

/** Runs one pass: appended system prompt + task message -> final assistant text. Injectable for tests.
 * With `sessionId`, the conversation is saved under that id so a later turn can continue it. */
export type PassRunner = (systemPrompt: string, message: string, opts?: { sessionId?: string }) => Promise<PassResult>;

// ---------------------------------------------------------------- prompts

export function systemPrompt(maxThreads: number): string {
	return `You identify the load-bearing threads of a task.

A thread is one distinct thing the task requires to be worked on: a requirement, a
constraint, a condition the work runs under, a resource, or a risk. A thread is
load-bearing if getting it wrong, or ignoring it, changes whether the task succeeds.

Rules:
- Work only from the task and the files it points to. Read them fully before answering.
  Do not rely on what tasks like this usually involve.
- Anchor every thread in the text. Quote the exact words it comes from, and name the file.
  If a thread is implied rather than stated, quote the words it is implied by.
- Each thread must be distinct. Do not list the same thing twice in different words.
- Return at most ${maxThreads} threads. If there are more, keep the most load-bearing.
- If the part you are given is already a single concrete thing that can be worked on
  directly, return no threads.
- You only identify. Do not plan, solve, or suggest how to do anything.

End your answer with exactly one JSON block in this shape and nothing after it:
{"threads": [{"title": "<short name>",
              "quote": "<exact words from the source>",
              "source": "<file the quote is from>",
              "why_load_bearing": "<what goes wrong in the outcome if this is missed>"}]}`;
}

function taskBlock(task: string): string {
	return `Here is a task, exactly as it was given:\n\n<<<\n${task}\n>>>`;
}

export function rootMessage(task: string): string {
	return `${taskBlock(task)}\n\nIdentify the load-bearing threads of this task.`;
}

export function childMessage(task: string, lineage: ThreadNode[]): string {
	const target = lineage[lineage.length - 1];
	const pathLine = lineage.map((n) => n.id).join(" > ");
	const ancestry = lineage
		.map((n) => `  ${n.id} ${n.title}\n    quote (${n.source}): "${n.quote}"\n    why it bears load: ${n.why_load_bearing}`)
		.join("\n");
	return `${taskBlock(task)}

Earlier passes broke this task into threads. You are given one of them:

Path: ${pathLine}
${ancestry}

Identify the load-bearing threads within ${target.id} only: the distinct things that must be
worked on to get this thread right. Stay inside this thread; other threads are handled
separately.`;
}

// ---------------------------------------------------------------- parsing

type RawThread = Pick<ThreadNode, "title" | "quote" | "source" | "why_load_bearing">;

/** Extract the last {"threads": [...]} JSON object from the text and validate it. Throws on failure. */
export function parseThreads(text: string, maxThreads: number): RawThread[] {
	const key = text.lastIndexOf('"threads"');
	if (key < 0) throw new Error("no JSON block with a \"threads\" key");
	// Walk back to the object's opening brace, then forward to its matching close.
	let start = text.lastIndexOf("{", key);
	while (start >= 0) {
		const end = matchBrace(text, start);
		if (end > key) {
			const obj = JSON.parse(text.slice(start, end + 1));
			if (!Array.isArray(obj.threads)) throw new Error('"threads" is not an array');
			const out: RawThread[] = [];
			for (const [i, t] of obj.threads.entries()) {
				for (const f of ["title", "quote", "source", "why_load_bearing"] as const) {
					if (typeof t?.[f] !== "string" || !t[f].trim()) throw new Error(`thread ${i + 1} is missing "${f}"`);
				}
				out.push({ title: t.title.trim(), quote: t.quote.trim(), source: t.source.trim(), why_load_bearing: t.why_load_bearing.trim() });
			}
			if (out.length > maxThreads) throw new Error(`returned ${out.length} threads, the limit is ${maxThreads}`);
			return out;
		}
		start = text.lastIndexOf("{", start - 1);
	}
	throw new Error("unbalanced JSON block");
}

function matchBrace(s: string, open: number): number {
	let depth = 0;
	let inStr = false;
	for (let i = open; i < s.length; i++) {
		const c = s[i];
		if (inStr) {
			if (c === "\\") i++;
			else if (c === '"') inStr = false;
		} else if (c === '"') inStr = true;
		else if (c === "{") depth++;
		else if (c === "}" && --depth === 0) return i;
	}
	return -1;
}

// ---------------------------------------------------------------- transient failures

/** Waits between attempts when a model call itself fails (rate limit, outage, pi exiting without an answer).
 * Parse failures are not transient and are handled by the caller. Overridable for tests. */
export interface Transient {
	backoffMs?: number[]; // one wait per extra attempt; default 15s, 30s, 60s, 120s
	sleep?: (ms: number) => Promise<void>;
	onRetry?: (attempt: number, waitMs: number, error: Error) => void;
}
const DEFAULT_BACKOFF_MS = [15_000, 30_000, 60_000, 120_000];

export async function withTransientRetry<T>(fn: () => Promise<T>, t: Transient = {}): Promise<T> {
	const backoff = t.backoffMs ?? DEFAULT_BACKOFF_MS;
	const sleep = t.sleep ?? ((ms: number) => new Promise<void>((r) => setTimeout(r, ms)));
	for (let i = 0; ; i++) {
		try {
			return await fn();
		} catch (e) {
			if (i >= backoff.length) throw e;
			t.onRetry?.(i + 1, backoff[i], e as Error);
			await sleep(backoff[i]);
		}
	}
}

// ---------------------------------------------------------------- the loop

export interface IdentifyOptions {
	task: string;
	runPass: PassRunner;
	maxDepth?: number; // levels below the task
	maxThreads?: number; // per pass
	concurrency?: number;
	retries?: number; // extra attempts per pass after a parse failure
	transient?: Transient; // waits between attempts when the call itself fails
	/** Return a session id to keep the conversation of the pass that expands this node (undefined: don't keep). */
	sessionFor?: (node: ThreadNode) => string | undefined;
	/** Keep the root pass's conversation under this id, so the ordering turn can continue it. */
	rootSessionId?: string;
	onEvent?: (msg: string) => void;
}

export async function identifyTree(o: IdentifyOptions): Promise<ThreadTree> {
	const maxDepth = o.maxDepth ?? 3;
	const maxThreads = o.maxThreads ?? 6;
	const concurrency = o.concurrency ?? 4;
	const retries = o.retries ?? 1;
	const sys = systemPrompt(maxThreads);
	const log = o.onEvent ?? (() => {});
	const tree: ThreadTree = {
		task: o.task,
		created: new Date().toISOString(),
		settings: { maxDepth, maxThreads, concurrency },
		root_status: "expanded",
		threads: [],
		passes: [],
	};

	async function pass(target: string, message: string, session?: string): Promise<{ found: RawThread[]; session?: string } | Error> {
		let last: Error = new Error("not attempted");
		for (let attempt = 1; attempt <= 1 + retries; attempt++) {
			const t0 = Date.now();
			// A retry gets its own session, so a kept conversation never contains a failed attempt.
			const sessionId = session ? `${session}-a${attempt}` : undefined;
			try {
				const r = await withTransientRetry(() => o.runPass(sys, message, { sessionId }), {
					...o.transient,
					onRetry: (n, wait, e) => { o.transient?.onRetry?.(n, wait, e); log(`${target}: call failed (${e.message.slice(0, 120)}); retry ${n} in ${wait / 1000}s`); },
				});
				const found = parseThreads(r.text, maxThreads);
				tree.passes.push({ target, attempt, ok: true, n_threads: found.length, seconds: (Date.now() - t0) / 1000, cost_usd: r.cost_usd });
				log(`${target}: ${found.length} thread(s)`);
				return { found, session: sessionId };
			} catch (e) {
				last = e as Error;
				tree.passes.push({ target, attempt, ok: false, error: last.message, seconds: (Date.now() - t0) / 1000 });
				log(`${target}: attempt ${attempt} failed: ${last.message}`);
			}
		}
		return last;
	}

	const make = (raw: RawThread, id: string, depth: number): ThreadNode => ({ id, ...raw, depth, status: "pending", children: [] });

	const root = await pass("root", rootMessage(o.task), o.rootSessionId);
	if (root instanceof Error) {
		tree.root_status = "failed";
		tree.root_error = root.message;
		return tree;
	}
	if (root.session) tree.root_session = root.session;
	tree.threads = root.found.map((r, i) => make(r, `T${i + 1}`, 1));
	if (tree.threads.length === 0) tree.root_status = "atomic";

	// Breadth-first: expand one level at a time, `concurrency` passes in flight.
	let frontier: { node: ThreadNode; lineage: ThreadNode[] }[] = tree.threads.map((n) => ({ node: n, lineage: [n] }));
	while (frontier.length) {
		const next: typeof frontier = [];
		await mapLimit(frontier, concurrency, async ({ node, lineage }) => {
			if (node.depth >= maxDepth) {
				node.status = "depth_cap";
				return;
			}
			const res = await pass(node.id, childMessage(o.task, lineage), o.sessionFor?.(node));
			if (res instanceof Error) {
				node.status = "failed";
				node.error = res.message;
				return;
			}
			if (res.session) node.session = res.session;
			node.children = res.found.map((r, i) => make(r, `${node.id}.${i + 1}`, node.depth + 1));
			node.status = node.children.length ? "expanded" : "atomic";
			for (const c of node.children) next.push({ node: c, lineage: [...lineage, c] });
		});
		frontier = next;
	}
	return tree;
}

async function mapLimit<T>(items: T[], limit: number, fn: (x: T) => Promise<void>): Promise<void> {
	let i = 0;
	const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
		while (i < items.length) await fn(items[i++]);
	});
	await Promise.all(workers);
}

// ---------------------------------------------------------------- output

export function renderMarkdown(tree: ThreadTree): string {
	const lines = [`# Load-bearing threads`, ``];
	if (tree.root_status === "failed") lines.push(`Identification failed: ${tree.root_error}`);
	const walk = (n: ThreadNode, indent: string) => {
		const tag = n.status === "expanded" ? "" : ` [${n.status}]`;
		lines.push(`${indent}- **${n.id} ${n.title}**${tag}`);
		lines.push(`${indent}  "${n.quote}" (${n.source})`);
		lines.push(`${indent}  Why it bears load: ${n.why_load_bearing}`);
		for (const c of n.children) walk(c, indent + "  ");
	};
	for (const n of tree.threads) walk(n, "");
	return lines.join("\n") + "\n";
}

/** One line per thread, indented by depth: the compact view shown to the researcher. */
export function renderIndex(tree: ThreadTree): string {
	const lines: string[] = [];
	const walk = (n: ThreadNode) => {
		lines.push(`${"  ".repeat(n.depth - 1)}${n.id} ${n.title}: ${n.why_load_bearing}`);
		n.children.forEach(walk);
	};
	tree.threads.forEach(walk);
	return lines.join("\n");
}

export function saveTree(dir: string, tree: ThreadTree): void {
	fs.mkdirSync(dir, { recursive: true });
	fs.writeFileSync(path.join(dir, "threads.json"), JSON.stringify(tree, null, 2) + "\n", "utf-8");
	fs.writeFileSync(path.join(dir, "threads.md"), renderMarkdown(tree), "utf-8");
}

// ---------------------------------------------------------------- ordering: the root pass's next turn

export function orderMessage(threads: ThreadNode[]): string {
	const list = threads.map((n) => `- ${n.id} ${n.title}: ${n.why_load_bearing}`).join("\n");
	return `The threads you identified will now be worked on one at a time, each by a different person,
each building on the work of the ones before. Put them in the order they should be worked on:
what has to exist before something else can be built on it comes first. Order by dependency,
not by importance.

${list}

End your answer with exactly one JSON block in this shape and nothing after it:
{"order": ["<thread id>", "<thread id>", ...]}
Every id exactly once.`;
}

/** Extract the last {"order": [...]} block and check it is a permutation of `ids`. Throws on failure. */
export function parseOrder(text: string, ids: string[]): string[] {
	const key = text.lastIndexOf('"order"');
	if (key < 0) throw new Error('no JSON block with an "order" key');
	let start = text.lastIndexOf("{", key);
	while (start >= 0) {
		const end = matchBrace(text, start);
		if (end > key) {
			const obj = JSON.parse(text.slice(start, end + 1));
			if (!Array.isArray(obj.order)) throw new Error('"order" is not an array');
			const got: string[] = obj.order.map((x: unknown) => String(x).trim());
			if (JSON.stringify([...got].sort()) !== JSON.stringify([...ids].sort())) {
				throw new Error(`order [${got.join(", ")}] is not a permutation of [${ids.join(", ")}]`);
			}
			return got;
		}
		start = text.lastIndexOf("{", start - 1);
	}
	throw new Error("unbalanced JSON block");
}

export interface OrderOptions {
	tree: ThreadTree;
	runPass: PassRunner;
	transient?: Transient;
	onEvent?: (msg: string) => void;
}

/** Continue the root conversation for one more turn: order the first-level threads chronologically.
 * On any failure, or with fewer than two threads, the order is T1, T2, ... and the reason is recorded. */
export async function orderThreads(o: OrderOptions): Promise<string[]> {
	const log = o.onEvent ?? (() => {});
	const ids = o.tree.threads.map((n) => n.id);
	const fallback = (why: string) => {
		o.tree.order = ids;
		o.tree.order_source = "fallback";
		o.tree.order_error = why;
		log(`order: ${why}; using ${ids.join(", ")}`);
		return ids;
	};
	if (ids.length < 2) return fallback("fewer than two threads");
	if (!o.tree.root_session) return fallback("the root conversation was not kept");
	const t0 = Date.now();
	try {
		const r = await withTransientRetry(
			() => o.runPass(systemPrompt(o.tree.settings.maxThreads), orderMessage(o.tree.threads), { sessionId: o.tree.root_session }),
			{ ...o.transient, onRetry: (n, wait, e) => { o.transient?.onRetry?.(n, wait, e); log(`order: call failed (${e.message.slice(0, 120)}); retry ${n} in ${wait / 1000}s`); } },
		);
		const order = parseOrder(r.text, ids);
		o.tree.order = order;
		o.tree.order_source = "model";
		o.tree.passes.push({ target: "order", attempt: 1, ok: true, seconds: (Date.now() - t0) / 1000, cost_usd: r.cost_usd });
		log(`order: ${order.join(", ")}`);
		return order;
	} catch (e) {
		o.tree.passes.push({ target: "order", attempt: 1, ok: false, error: (e as Error).message, seconds: (Date.now() - t0) / 1000 });
		return fallback(`ordering failed: ${(e as Error).message}`);
	}
}

/** The first-level threads in canon order (threads the order does not name come last, in id order). */
export function orderedThreads(tree: ThreadTree): ThreadNode[] {
	const order = tree.order ?? [];
	const byId = new Map(tree.threads.map((n) => [n.id, n]));
	const named = order.map((id) => byId.get(id)).filter((n): n is ThreadNode => !!n);
	const rest = tree.threads.filter((n) => !order.includes(n.id));
	return [...named, ...rest];
}

// ---------------------------------------------------------------- implementation: the thread pass's next turn

export interface EditRecord {
	tool: string; // "edit" | "write"
	path: string;
}

export interface ImplementResult {
	id: string;
	title: string;
	session: string;
	ok: boolean;
	report: string; // the implementer's final message
	steered: boolean; // stopped by the context rule after an edit
	steer?: { percent: number; tokens: number | null };
	edits: EditRecord[];
	tool_calls: number;
	seconds: number;
	cost_usd?: number;
	error?: string;
}

/** What one implementation turn returns; injectable for tests. */
export interface ImplementTurn {
	text: string;
	edits: EditRecord[];
	tool_calls: number;
	steer?: { percent: number; tokens: number | null };
	cost_usd?: number;
}
export type ImplementRunner = (sessionId: string, message: string) => Promise<ImplementTurn>;

/** The implement turn for `node`: its sub-threads, its place in the canon, and the reports of the threads before it. */
export function implementMessage(node: ThreadNode, canon: ThreadNode[], prior: ImplementResult[]): string {
	const subs = node.children.map((c) => `- ${c.id} ${c.title}`).join("\n");
	const me = canon.findIndex((n) => n.id === node.id);
	const schedule = canon
		.map((n, i) => `  ${i + 1}. ${n.id} ${n.title}${i < me ? " (done)" : i === me ? " (you)" : " (after you)"}`)
		.join("\n");
	const reports = prior.length
		? prior.map((r) => `--- ${r.id} ${r.title} ---\n${r.ok ? r.report || "(no report)" : `(this implementer failed: ${r.error}; nothing of it is in the workspace)`}`).join("\n\n")
		: "(none: you are first)";
	return `Now implement the threads you identified within ${node.id} ${node.title}, in this workspace:
${subs}

The threads of this task are being implemented one at a time, each by a different person, in
this order:
${schedule}

The work of the threads before yours is already in this workspace. Their reports:

${reports}

What is there is earlier work on other threads of the same task, each done by someone who saw
that one thread; none of it is final. Your own reading of the brief governs your thread. You have
the same authority over earlier threads' work as over your own: when your thread needs it
different, change it, including a design choice an earlier thread presented as settled. Read a
file before you edit it. Keeping earlier work is also a decision: for every earlier decision your
thread touches or depends on, check it against the brief yourself and say whether you kept it or
changed it. Every change to earlier work must be named in your report: the file, the thread the
work came from, and why your thread needs it different.

When you are done, finish with a report. For each of your threads, report:
- what you changed (files and what the change does),
- why, in terms of that thread,
- its state: done, partly done, or not started,
- what is left and anything the reviewer should check.
Then list every earlier decision you checked against the brief, each marked kept or changed with
the thread and the reason; if you changed nothing of earlier work, say so and say why it held.`;
}

export interface ImplementOptions {
	tree: ThreadTree;
	runImplement: ImplementRunner;
	transient?: Transient;
	onEvent?: (msg: string) => void;
}

/** Give every expanded first-level thread's conversation one more turn: implement its sub-threads.
 * One thread at a time, in canon order; each turn is given the reports of the turns before it. */
export async function implementThreads(o: ImplementOptions): Promise<ImplementResult[]> {
	const log = o.onEvent ?? (() => {});
	const canon = orderedThreads(o.tree).filter((n) => n.session && n.children.length > 0);
	const results: ImplementResult[] = [];
	for (const node of canon) {
		const t0 = Date.now();
		const base = { id: node.id, title: node.title, session: node.session! };
		try {
			const r = await withTransientRetry(() => o.runImplement(node.session!, implementMessage(node, canon, results)), {
				...o.transient,
				onRetry: (n, wait, e) => { o.transient?.onRetry?.(n, wait, e); log(`${node.id}: call failed (${e.message.slice(0, 120)}); retry ${n} in ${wait / 1000}s`); },
			});
			results.push({ ...base, ok: true, report: r.text, steered: !!r.steer, steer: r.steer, edits: r.edits,
				tool_calls: r.tool_calls, seconds: (Date.now() - t0) / 1000, cost_usd: r.cost_usd });
			const stop = r.steer ? `, stopped at ${r.steer.percent.toFixed(0)}% context` : "";
			log(`${node.id}: implemented, ${r.edits.length} edit(s)${stop}`);
		} catch (e) {
			results.push({ ...base, ok: false, report: "", steered: false, edits: [], tool_calls: 0,
				seconds: (Date.now() - t0) / 1000, error: (e as Error).message });
			log(`${node.id}: implementation failed: ${(e as Error).message}`);
		}
	}
	return results;
}

export function renderImplementReport(results: ImplementResult[]): string {
	const out = ["# Implementation turns", ""];
	for (const r of results) {
		out.push(`## ${r.id} ${r.title}`, "");
		if (r.ok) {
			const files = [...new Set(r.edits.map((e) => e.path))].join(", ") || "nothing";
			const stop = r.steered ? `; stopped by the context rule at ${r.steer!.percent.toFixed(1)}%` : "";
			out.push(`${r.edits.length} edit(s) to ${files}; ${r.tool_calls} tool call(s); ${r.seconds.toFixed(0)}s${stop}`);
		} else {
			out.push(`Failed: ${r.error}`);
		}
		out.push("", r.report || "(no report)", "");
	}
	return out.join("\n");
}

// ---------------------------------------------------------------- running pi

export interface PiPassOptions {
	cwd: string; // the task's working directory
	piCommand: string[]; // e.g. ["pi"] or [process.execPath, "/path/to/cli.js"]
	model?: string; // e.g. "gateway/z-ai/glm-5.3-flash"
	sessionDir?: string; // where kept conversations are stored (required to keep or continue one)
	env?: NodeJS.ProcessEnv;
	timeoutMs?: number;
}

interface PiRun {
	text: string;
	cost: number;
	toolCalls: { name: string; args: Record<string, unknown> }[];
}

/** Spawn one `pi --mode json -p` run and collect its final assistant text, cost and tool calls.
 * Every process started here is marked AUTORESEARCH_CHILD=1, whoever the caller is: a pass or an
 * implementer that loads the custom agent dir must never run the autoresearch pipeline itself. */
async function runPi(p: PiPassOptions, args: string[], env: NodeJS.ProcessEnv): Promise<PiRun> {
	return await new Promise<PiRun>((resolve, reject) => {
		const [cmd, ...pre] = p.piCommand;
		const childEnv = { ...env, AUTORESEARCH_CHILD: "1" };
		const proc = spawn(cmd, [...pre, ...args], { cwd: p.cwd, env: childEnv, shell: false, stdio: ["ignore", "pipe", "pipe"] });
		let buf = "";
		let stderr = "";
		let finalText = "";
		let cost = 0;
		let errorMessage = "";
		const toolCalls: PiRun["toolCalls"] = [];
		const onLine = (line: string) => {
			if (!line.trim()) return;
			let ev: any;
			try {
				ev = JSON.parse(line);
			} catch {
				return;
			}
			if (ev.type === "message_end" && ev.message?.role === "assistant") {
				const parts = Array.isArray(ev.message.content) ? ev.message.content : [];
				const text = parts.filter((c: any) => c.type === "text").map((c: any) => c.text).join("");
				if (text.trim()) finalText = text;
				for (const c of parts) if (c.type === "toolCall") toolCalls.push({ name: c.name, args: c.arguments ?? {} });
				cost += ev.message.usage?.cost?.total || 0;
				if (ev.message.errorMessage) errorMessage = ev.message.errorMessage;
			}
		};
		const timer = setTimeout(() => proc.kill("SIGKILL"), p.timeoutMs ?? 3_600_000);
		proc.stdout.on("data", (d) => {
			buf += d.toString();
			const lines = buf.split("\n");
			buf = lines.pop() || "";
			lines.forEach(onLine);
		});
		proc.stderr.on("data", (d) => (stderr += d.toString()));
		proc.on("error", (e) => {
			clearTimeout(timer);
			reject(e);
		});
		proc.on("close", (code) => {
			clearTimeout(timer);
			if (buf.trim()) onLine(buf);
			if (code !== 0 && !finalText) reject(new Error(`pi exited ${code}: ${(errorMessage || stderr).slice(-400)}`));
			else if (!finalText) reject(new Error(`no answer${errorMessage ? `: ${errorMessage}` : ""}`));
			else resolve({ text: finalText, cost, toolCalls });
		});
	});
}

function sessionArgs(p: PiPassOptions, sessionId?: string): string[] {
	if (!sessionId) return ["--no-session"];
	if (!p.sessionDir) throw new Error("sessionDir is required to keep or continue a conversation");
	return ["--session-dir", p.sessionDir, "--session-id", sessionId];
}

/** Identification pass: read-only tools, identifier prompt appended. */
export function piPassRunner(p: PiPassOptions): PassRunner {
	return async (sys, message, opts) => {
		const dir = await fs.promises.mkdtemp(path.join(os.tmpdir(), "pi-threads-"));
		const promptFile = path.join(dir, "identifier.md");
		await fs.promises.writeFile(promptFile, sys, { encoding: "utf-8", mode: 0o600 });
		const args = ["--mode", "json", "-p", ...sessionArgs(p, opts?.sessionId), "--tools", "read,ls,find,grep",
			"--append-system-prompt", promptFile];
		if (p.model) args.push("--model", p.model);
		args.push(message);
		try {
			const r = await runPi(p, args, p.env ?? process.env);
			return { text: r.text, cost_usd: r.cost };
		} finally {
			fs.rmSync(dir, { recursive: true, force: true });
		}
	};
}

/** Implementation turn: the same conversation continued, with write tools and the implementer extension.
 * The identifier prompt is not appended again: this turn is not identification. */
export function piImplementRunner(p: PiPassOptions & { extensionPath: string; stopPercent?: number }): ImplementRunner {
	return async (sessionId, message) => {
		const dir = await fs.promises.mkdtemp(path.join(os.tmpdir(), "pi-implement-"));
		const steerLog = path.join(dir, "steer.jsonl");
		const args = ["--mode", "json", "-p", ...sessionArgs(p, sessionId), "--tools", "read,bash,edit,write,ls,find,grep",
			"-e", p.extensionPath];
		if (p.model) args.push("--model", p.model);
		args.push(message);
		const env = { ...(p.env ?? process.env), IMPLEMENTER_LOG: steerLog, IMPLEMENTER_STOP_PERCENT: String(p.stopPercent ?? 40) };
		try {
			const r = await runPi(p, args, env);
			const steerLine = fs.existsSync(steerLog) ? fs.readFileSync(steerLog, "utf-8").trim().split("\n")[0] : "";
			const steer = steerLine ? JSON.parse(steerLine) : undefined;
			const edits = r.toolCalls
				.filter((c) => c.name === "edit" || c.name === "write")
				.map((c) => ({ tool: c.name, path: String(c.args.path ?? c.args.file_path ?? "?") }));
			return { text: r.text, edits, tool_calls: r.toolCalls.length, cost_usd: r.cost,
				steer: steer ? { percent: steer.percent, tokens: steer.tokens } : undefined };
		} finally {
			fs.rmSync(dir, { recursive: true, force: true });
		}
	};
}

// ---------------------------------------------------------------- the planning turn's guard

/** True when a tool call's input names the .harness directory anywhere: as a path, or inside a bash command.
 * Used to keep the researcher's plan uninformed by the pipeline's reports and diff. */
export function touchesHarness(input: unknown): boolean {
	return /(^|[\s"'`=:/\\(])\.harness(?=$|[\s"'`/\\*;)])/.test(JSON.stringify(input ?? ""));
}

// ---------------------------------------------------------------- what the researcher is shown

/** Added to the researcher's context at its first start: think first, change nothing yet. Says nothing about the
 * threads, the deliverable, or what the thinking should contain beyond its shape. */
export function planningMessage(): string {
	return `Before you do anything else: read the brief and look around this repository as much as you need,
but change nothing yet. Then reason out loud, in your reply, about what you would build, how you
would test it, and what could go wrong. Once you have replied, you will be told more and can
start the work.`;
}

/** Delivered after the researcher has replied with its own thinking, when the threads' work has just been applied
 * to the workspace. That reply is the researcher's own words; this message asks it to hold the build to them. */
export function researcherSection(tree: ThreadTree, results: ImplementResult[], stripPrefixes: string[] = ["/work/"]): string {
	const strip = (p: string) => {
		for (const pre of stripPrefixes) if (pre && p.startsWith(pre)) return p.slice(pre.length);
		return p;
	};
	const files = (r: ImplementResult) => [...new Set(r.edits.map((e) => strip(e.path)))].join(", ") || "no files";
	const byId = new Map(results.map((r) => [r.id, r]));
	const canon = orderedThreads(tree);
	const impl = canon
		.map((n, i) => {
			const r = byId.get(n.id);
			if (!r) return `  ${i + 1}. ${n.id} ${n.title}: no sub-threads, nothing implemented`;
			return r.ok
				? `  ${i + 1}. ${n.id} ${n.title}: ${r.edits.length} edit(s) (${files(r)})${r.steered ? "; stopped early, report covers partial work" : ""}`
				: `  ${i + 1}. ${n.id} ${n.title}: the implementer failed (${r.error}); nothing from it to review`;
		})
		.join("\n");
	const index = renderIndex(tree).split("\n").map((l) => `  ${l}`).join("\n");
	return `You are not starting from scratch. While you were thinking, the result of earlier work was
applied to this repository: before you began, the task was broken into load-bearing threads
(a tree: each thread names what it bears and why, with sub-threads under it), and ${results.length}
implementers worked on those threads one after another, each building on the ones before. Their
changes are now in the working tree (all of them: .harness/implement.diff). You are the final
reviewer and integrator: the deliverable is still yours, and your job now is to turn their work
into one coherent, working result that does what the brief asks.

What you wrote in your last reply was written without seeing any of this. Reconcile it in two
passes, and write both down.

Pass 1, your reply against the thread tree. For each thing you said you would build or test, name
the thread that covers it, or say that no thread does. Then the other way: for each thread, say
whether you had it, and if not, whether the thread is right that it bears load. A thread you did
not think of may be a real gap in your thinking, or may be weight on something that does not
matter; decide which.

Pass 2, your reply against the code as built. Treat what you wrote as a design, not a note: go
through it point by point. For each point, establish whether the build does it, does it
differently, or does not do it. Where it does not, you build it. Where it does it differently,
decide which is right and say why.

Disagreement is expected, and you decide it. The implementers each saw one thread of the task, in
sequence, and none of them saw the whole. You do. Nothing they built is settled: any decision in
the tree or in the code, including one every implementer after it inherited, is yours to overturn
when you judge it should be, and you need no permission to change their work. Where their choices
conflict with what you would have done, your judgment governs: change the work, and record what
you changed and why. Treat their reports as claims to verify, not as results.

Approval is not a result. The implementers' own tests passing shows their work does what they
meant; it does not show the deliverable does what the brief asks. You have the budget, and it is
there to be spent: build tasks of your own that look like the ones the brief describes, run the
deliverable on them, judge it the way it will be judged, and change what fails. Ending this job
without having changed anything is a finding you have to justify, not a default. After the
review, carry on with the job as the brief describes, within the budget that remains.

Threads (full detail: .harness/threads.md)
${index}

Implementers, in the order they worked (full reports: .harness/implement.md)
${impl}`;
}
