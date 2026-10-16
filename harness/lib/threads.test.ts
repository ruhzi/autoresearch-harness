// node --test harness-release/pi/lib/threads.test.ts   (no model calls: the pass runner is faked)
import assert from "node:assert/strict";
import { test } from "node:test";
import {
	childMessage, identifyTree, implementMessage, implementThreads, orderMessage, orderThreads, orderedThreads, parseOrder,
	parseThreads, planningMessage, renderIndex, researcherSection, rootMessage, systemPrompt, touchesHarness, withTransientRetry, type PassRunner,
	type ThreadNode,
} from "./threads.ts";

const t = (title: string) => ({ title, quote: `q-${title}`, source: "BRIEF.md", why_load_bearing: `w-${title}` });
const answer = (threads: object[]) => `Some reasoning first.\n\n{"threads": ${JSON.stringify(threads)}}`;

/** Fake runner keyed by the target named in the message ("root", "order", or the thread id being expanded). */
function fake(byTarget: Record<string, string | Error>, seen: string[] = []): PassRunner {
	return async (_sys, msg) => {
		const m = msg.match(/within (T[\d.]+) only/);
		const target = m ? m[1] : msg.includes('"order"') ? "order" : "root";
		seen.push(target);
		const r = byTarget[target] ?? answer([]);
		if (r instanceof Error) throw r;
		return { text: r, cost_usd: 0.001 };
	};
}

test("recurses breadth-first and stops when a pass returns no threads", async () => {
	const seen: string[] = [];
	const tree = await identifyTree({
		task: "TASK",
		runPass: fake({ root: answer([t("a"), t("b")]), T1: answer([t("a1")]), "T1.1": answer([]), T2: answer([]) }, seen),
	});
	assert.deepEqual(tree.threads.map((n) => n.id), ["T1", "T2"]);
	assert.deepEqual(tree.threads[0].children.map((n) => n.id), ["T1.1"]);
	assert.equal(tree.threads[0].status, "expanded");
	assert.equal(tree.threads[1].status, "atomic");
	assert.equal(tree.threads[0].children[0].status, "atomic");
	assert.deepEqual(seen, ["root", "T1", "T2", "T1.1"]); // level by level
});

test("each pass sees the task and its own path, not its siblings", async () => {
	const msgs: string[] = [];
	await identifyTree({
		task: "THE-TASK",
		concurrency: 1,
		runPass: async (_s, msg) => {
			msgs.push(msg);
			if (!msg.includes("Path:")) return { text: answer([t("alpha"), t("beta")]) };
			if (msg.includes("within T1 only")) return { text: answer([t("alpha-sub")]) };
			return { text: answer([]) };
		},
	});
	const t11 = msgs.find((m) => m.includes("within T1.1 only"))!;
	assert.ok(t11.includes("THE-TASK"));
	assert.ok(t11.includes("Path: T1 > T1.1"));
	assert.ok(t11.includes("q-alpha") && t11.includes("q-alpha-sub"));
	assert.ok(!t11.includes("beta"), "a pass must not see sibling threads");
});

test("depth cap stops expansion without a pass", async () => {
	const seen: string[] = [];
	const tree = await identifyTree({
		task: "X",
		maxDepth: 2,
		runPass: fake({ root: answer([t("a")]), T1: answer([t("b")]), "T1.1": answer([t("never")]) }, seen),
	});
	assert.equal(tree.threads[0].children[0].status, "depth_cap");
	assert.ok(!seen.includes("T1.1"));
});

test("more threads than the cap is a failed pass; the retry is used; then the branch is marked failed", async () => {
	let calls = 0;
	const tree = await identifyTree({
		task: "X",
		maxThreads: 2,
		runPass: async (_s, msg) => {
			if (!msg.includes("Path:")) return { text: answer([t("a"), t("b")]) };
			calls++;
			if (msg.includes("within T1 only")) return { text: answer([t("1"), t("2"), t("3")]) };
			return { text: answer([]) };
		},
	});
	assert.equal(tree.threads[0].status, "failed");
	assert.match(tree.threads[0].error!, /limit is 2/);
	assert.equal(tree.threads[1].status, "atomic", "a failed branch must not block its sibling");
	assert.equal(tree.passes.filter((p) => p.target === "T1").length, 2); // first try + one retry
	assert.equal(calls, 3);
});

test("a retry that succeeds is accepted", async () => {
	let n = 0;
	const tree = await identifyTree({
		task: "X",
		runPass: async (_s, msg) => {
			if (msg.includes("Path:")) return { text: answer([]) };
			return { text: ++n === 1 ? "no json here" : answer([t("a")]) };
		},
	});
	assert.equal(tree.root_status, "expanded");
	assert.equal(tree.threads.length, 1);
});

test("root failure is recorded, not thrown", async () => {
	const tree = await identifyTree({ task: "X", transient: { backoffMs: [] }, runPass: async () => { throw new Error("boom"); } });
	assert.equal(tree.root_status, "failed");
	assert.match(tree.root_error!, /boom/);
});

test("parser takes the last threads block and validates fields", () => {
	const txt = `Draft: {"threads": [{"title":"x"}]}\nFinal:\n\`\`\`json\n{"threads": [{"title":"a","quote":"q {braces} \\"x\\"","source":"s","why_load_bearing":"w"}]}\n\`\`\``;
	const r = parseThreads(txt, 6);
	assert.equal(r.length, 1);
	assert.equal(r[0].quote, 'q {braces} "x"');
	assert.throws(() => parseThreads(`{"threads": [{"title":"a"}]}`, 6), /missing "quote"/);
	assert.throws(() => parseThreads("nothing", 6), /no JSON block/);
});

test("prompts carry no benchmark or domain knowledge, and the planning steer says nothing about the threads", () => {
	const node: ThreadNode = { id: "T1", title: "x", quote: "q", source: "s", why_load_bearing: "w", depth: 1, status: "pending", children: [] };
	const all = (systemPrompt(6) + rootMessage("T") + childMessage("T", [node]) + orderMessage([node]) + planningMessage()).toLowerCase();
	for (const w of ["dabstep", "adyen", "payment", "fee", "merchant", "not applicable", "n/a", "abstain",
		"documentation", "manual", "dataset", "benchmark", "agent"]) {
		assert.ok(!all.includes(w), `prompt mentions "${w}"`);
	}
	const plan = planningMessage().toLowerCase();
	for (const w of ["thread", "implement", "deliverable", "review"]) assert.ok(!plan.includes(w), `planning steer mentions "${w}"`);
	assert.match(planningMessage(), /change nothing yet/);
	assert.match(planningMessage(), /reason out loud, in your reply/);
	assert.ok(!planningMessage().toLowerCase().includes("plan"), "the steer asks for thinking, not a document");
});

test("index view is one line per thread, indented by depth", async () => {
	const tree = await identifyTree({ task: "X", runPass: fake({ root: answer([t("a")]), T1: answer([t("b")]) }) });
	assert.equal(renderIndex(tree), "T1 a: w-a\n  T1.1 b: w-b");
});

// ---------------------------------------------------------------- kept conversations and ordering

test("first-level thread passes keep a conversation, a retry gets its own, and the root keeps one only when asked", async () => {
	const ids: (string | undefined)[] = [];
	let t2Tries = 0;
	const run: PassRunner = async (_s, msg, opts) => {
		ids.push(opts?.sessionId);
		if (!msg.includes("Path:")) return { text: answer([t("a"), t("b")]) };
		if (msg.includes("within T2 only") && ++t2Tries === 1) return { text: "garbled" };
		return { text: answer([t("sub")]) };
	};
	const tree = await identifyTree({ task: "X", maxDepth: 2, sessionFor: (n) => (n.depth === 1 ? `thread-${n.id}` : undefined), runPass: run });
	assert.equal(ids[0], undefined, "the root pass keeps no conversation unless asked");
	assert.equal(tree.root_session, undefined);
	assert.equal(tree.threads[0].session, "thread-T1-a1");
	assert.equal(tree.threads[1].session, "thread-T2-a2", "the kept conversation is the successful attempt");
	assert.equal(tree.threads[0].children[0].status, "depth_cap");

	ids.length = 0;
	t2Tries = 0;
	const kept = await identifyTree({ task: "X", maxDepth: 2, rootSessionId: "root", runPass: run });
	assert.equal(ids[0], "root-a1");
	assert.equal(kept.root_session, "root-a1");
});

test("parseOrder takes the last order block and demands a permutation", () => {
	assert.deepEqual(parseOrder('draft {"order":["T1"]} final {"order": ["T2", "T1"]}', ["T1", "T2"]), ["T2", "T1"]);
	assert.throws(() => parseOrder('{"order": ["T2", "T2"]}', ["T1", "T2"]), /not a permutation/);
	assert.throws(() => parseOrder('{"order": ["T1"]}', ["T1", "T2"]), /not a permutation/);
	assert.throws(() => parseOrder("nothing", ["T1"]), /no JSON block/);
});

test("the ordering turn continues the root conversation and sets the canon; failures fall back to id order", async () => {
	const passes = fake({ root: answer([t("a"), t("b"), t("c")]), order: 'ok {"order": ["T3", "T1", "T2"]}' });
	const sessions: (string | undefined)[] = [];
	const spy: PassRunner = (s, m, o) => { sessions.push(o?.sessionId); return passes(s, m, o); };
	const tree = await identifyTree({ task: "X", maxDepth: 1, rootSessionId: "root", runPass: spy });
	assert.deepEqual(await orderThreads({ tree, runPass: spy }), ["T3", "T1", "T2"]);
	assert.equal(tree.order_source, "model");
	assert.equal(sessions[sessions.length - 1], "root-a1", "the ordering turn is the root pass's next turn");
	assert.deepEqual(orderedThreads(tree).map((n) => n.id), ["T3", "T1", "T2"]);
	assert.ok(tree.passes.some((p) => p.target === "order" && p.ok));

	const bad = await identifyTree({ task: "X", maxDepth: 1, rootSessionId: "root", runPass: fake({ root: answer([t("a"), t("b")]) }) });
	await orderThreads({ tree: bad, runPass: fake({ order: '{"order": ["T1", "T1"]}' }) });
	assert.deepEqual(bad.order, ["T1", "T2"]);
	assert.equal(bad.order_source, "fallback");
	assert.match(bad.order_error!, /not a permutation/);

	const unkept = await identifyTree({ task: "X", maxDepth: 1, runPass: fake({ root: answer([t("a"), t("b")]) }) });
	await orderThreads({ tree: unkept, runPass: async () => { throw new Error("must not be called"); } });
	assert.equal(unkept.order_source, "fallback");
	assert.match(unkept.order_error!, /not kept/);

	const single = await identifyTree({ task: "X", maxDepth: 1, rootSessionId: "root", runPass: fake({ root: answer([t("a")]) }) });
	await orderThreads({ tree: single, runPass: async () => { throw new Error("must not be called"); } });
	assert.deepEqual(single.order, ["T1"]);
});

test("the ordering message lists every thread and asks for dependency order, not importance", () => {
	const m = orderMessage([{ ...t("a"), id: "T1", depth: 1, status: "expanded", children: [] }, { ...t("b"), id: "T2", depth: 1, status: "expanded", children: [] }]);
	assert.ok(m.includes("- T1 a: w-a") && m.includes("- T2 b: w-b"));
	assert.match(m, /Order by dependency,\nnot by importance/);
	assert.match(m, /Every id exactly once/);
});

// ---------------------------------------------------------------- implementation turns

test("expanded threads are implemented one at a time in canon order, each given the reports before it", async () => {
	const tree = await identifyTree({
		task: "X", maxDepth: 2, sessionFor: (n) => `thread-${n.id}`,
		runPass: fake({ root: answer([t("a"), t("b"), t("c"), t("d")]), T1: answer([t("a1"), t("a2")]), T2: answer([t("b1")]), T3: answer([]), T4: answer([t("d1")]) }),
	});
	tree.order = ["T4", "T2", "T3", "T1"];
	let inFlight = 0, peak = 0;
	const calls: [string, string][] = [];
	const res = await implementThreads({
		tree,
		transient: { backoffMs: [] }, // a thrown turn fails the thread outright here
		runImplement: async (sid, msg) => {
			calls.push([sid, msg]);
			inFlight++; peak = Math.max(peak, inFlight);
			await new Promise((r) => setTimeout(r, 10));
			inFlight--;
			if (sid === "thread-T2-a1") throw new Error("boom");
			return { text: `REPORT ${sid}`, edits: [{ tool: "edit", path: "agent/run.py" }], tool_calls: 3,
				steer: { percent: 41.5, tokens: 83000 } };
		},
	});
	assert.deepEqual(calls.map((c) => c[0]), ["thread-T4-a1", "thread-T2-a1", "thread-T1-a1"], "canon order; T3 had no sub-threads, so no turn");
	assert.equal(peak, 1, "implementers run one at a time");
	assert.deepEqual(res.map((r) => [r.id, r.ok]), [["T4", true], ["T2", false], ["T1", true]], "one failure does not stop the rest");
	assert.equal(res[0].steered, true);
	assert.match(res[1].error!, /boom/);

	const first = calls[0][1];
	assert.ok(first.includes("T4.1 d1"), "lists its own sub-threads");
	assert.match(first, /1\. T4 d \(you\)\n  2\. T2 b \(after you\)\n  3\. T1 a \(after you\)/);
	assert.match(first, /\(none: you are first\)/);
	const last = calls[2][1];
	assert.ok(last.includes("T1.1 a1") && last.includes("T1.2 a2"));
	assert.match(last, /1\. T4 d \(done\)\n  2\. T2 b \(done\)\n  3\. T1 a \(you\)/);
	assert.ok(last.includes("--- T4 d ---\nREPORT thread-T4-a1"), "earlier reports are quoted");
	assert.ok(last.includes("--- T2 b ---\n(this implementer failed: boom"), "a failed predecessor is named as such");
	assert.match(last, /Every\s+change\s+to\s+earlier\s+work\s+must\s+be\s+named\s+in\s+your\s+report/);
	assert.match(last, /same\s+authority\s+over\s+earlier\s+threads'\s+work\s+as\s+over\s+your\s+own/);
	assert.match(last, /Keeping\s+earlier\s+work\s+is\s+also\s+a\s+decision/);
	assert.match(last, /each\s+marked\s+kept\s+or\s+changed/);
	assert.match(last, /its state: done, partly done, or not started/);
	assert.ok(!last.includes("at the same time"), "nothing runs alongside an implementer");
});

test("the implement message carries no rule about context limits", () => {
	const node: ThreadNode = { id: "T1", title: "x", quote: "q", source: "s", why_load_bearing: "w", depth: 1, status: "expanded",
		children: [{ id: "T1.1", title: "y", quote: "q", source: "s", why_load_bearing: "w", depth: 2, status: "depth_cap", children: [] }] };
	const m = implementMessage(node, [node], []).toLowerCase();
	for (const w of ["40%", "percent", "context", "token"]) assert.ok(!m.includes(w), `mentions "${w}"`);
});

// ---------------------------------------------------------------- the hand-over

test("the researcher section holds the build to the researcher's plan, lists the canon, and points at the files", async () => {
	const tree = await identifyTree({ task: "X", maxDepth: 2, sessionFor: (n) => `s-${n.id}`,
		runPass: fake({ root: answer([t("a"), t("b"), t("c")]), T1: answer([t("a1")]), T2: answer([t("b1")]), T3: answer([]) }) });
	tree.order = ["T2", "T3", "T1"];
	const results = [
		{ id: "T2", title: "b", session: "s", ok: false, report: "", steered: false, edits: [], tool_calls: 0, seconds: 1, error: "boom" },
		{ id: "T1", title: "a", session: "s", ok: true, report: "r", steered: true, steer: { percent: 41, tokens: 1 },
			edits: [{ tool: "edit", path: "/work/.harness/build/agent/run.py" }, { tool: "edit", path: "/work/.harness/build/agent/run.py" }], tool_calls: 4, seconds: 1 },
	];
	const s = researcherSection(tree, results, ["/work/.harness/build/", "/work/"]);
	assert.match(s, /You are the final\s+reviewer and integrator/);
	assert.match(s, /Pass\s+1,\s+your\s+reply\s+against\s+the\s+thread\s+tree/);
	assert.match(s, /Pass\s+2,\s+your\s+reply\s+against\s+the\s+code\s+as\s+built/);
	assert.match(s, /yours\s+to\s+overturn/);
	assert.match(s, /own\s+tests\s+passing\s+shows\s+their\s+work\s+does\s+what\s+they\s+meant/);
	assert.match(s, /one after another/);
	assert.match(s, /Treat what you wrote as a design, not a note/);
	assert.match(s, /Disagreement is expected/);
	assert.match(s, /Approval is not a result/);
	assert.match(s, /a finding you have to justify, not a default/);
	assert.match(s, /treat\s+their reports as claims to\s+verify, not as results/i);
	assert.match(s, /carry on with the\s+job as the brief describes/);
	assert.ok(s.includes("  T1 a: w-a") && s.includes("    T1.1 a1: w-a1"), "tree index, indented");
	assert.ok(s.includes("1. T2 b: the implementer failed (boom)"));
	assert.ok(s.includes("2. T3 c: no sub-threads, nothing implemented"));
	assert.ok(s.includes("3. T1 a: 2 edit(s) (agent/run.py); stopped early"), "build prefix stripped, files deduplicated");
	assert.ok(s.includes(".harness/implement.md") && s.includes(".harness/implement.diff") && s.includes(".harness/threads.md"));
	assert.ok(!s.includes("at the same time"));
});

// ---------------------------------------------------------------- transient failures

test("a failed model call is retried after the backoff waits, then given up", async () => {
	const waits: number[] = [];
	let n = 0;
	const v = await withTransientRetry(async () => { if (++n < 3) throw new Error("429"); return "ok"; },
		{ backoffMs: [5, 7, 9], sleep: async (ms) => { waits.push(ms); } });
	assert.equal(v, "ok");
	assert.deepEqual(waits, [5, 7]);
	n = 0;
	await assert.rejects(withTransientRetry(async () => { n++; throw new Error("down"); }, { backoffMs: [1, 1], sleep: async () => {} }), /down/);
	assert.equal(n, 3, "one try per wait, plus the first");
});

test("the root pass survives rate limits by waiting, and a parse failure does not wait", async () => {
	const waits: number[] = [];
	let calls = 0;
	const tree = await identifyTree({
		task: "X", maxDepth: 1, retries: 1,
		transient: { backoffMs: [3, 3], sleep: async (ms) => { waits.push(ms); } },
		runPass: async () => {
			calls++;
			if (calls <= 2) throw new Error("no answer: 429 rate-limited upstream");
			if (calls === 3) return { text: "garbled" };
			return { text: answer([t("a")]) };
		},
	});
	assert.equal(tree.root_status, "expanded");
	assert.deepEqual(waits, [3, 3], "two waits for the two call failures");
	assert.equal(calls, 4, "then a parse retry with no wait");
	assert.equal(tree.passes.filter((p) => p.target === "root").length, 2, "call failures do not count as attempts");
});

test("ordering and implementation turns get the same treatment", async () => {
	const waits: number[] = [];
	const transient = { backoffMs: [2], sleep: async (ms: number) => { waits.push(ms); } };
	const tree = await identifyTree({ task: "X", maxDepth: 2, rootSessionId: "root", sessionFor: (n) => `s-${n.id}`,
		runPass: fake({ root: answer([t("a"), t("b")]), T1: answer([t("a1")]), T2: answer([t("b1")]) }) });
	let orderCalls = 0;
	await orderThreads({ tree, transient, runPass: async () => { if (++orderCalls === 1) throw new Error("503"); return { text: '{"order": ["T2", "T1"]}' }; } });
	assert.deepEqual(tree.order, ["T2", "T1"]);
	let implCalls = 0;
	const res = await implementThreads({ tree, transient, runImplement: async (sid) => {
		if (++implCalls === 1) throw new Error("429");
		return { text: `R ${sid}`, edits: [], tool_calls: 1 };
	} });
	assert.deepEqual(res.map((r) => [r.id, r.ok]), [["T2", true], ["T1", true]]);
	assert.deepEqual(waits, [2, 2]);
});

// ---------------------------------------------------------------- the planning turn's guard

test("the planning guard catches every way of looking into .harness and nothing else", () => {
	for (const input of [
		{ path: ".harness/implement.md" }, { path: "/work/.harness/researcher_section.md" }, { path: ".harness" },
		{ command: "head -50 /work/.harness/implement.md" }, { command: "ls -la .harness/" }, { command: "cat '.harness/threads.md'" },
		{ command: "awk '/x/,0' .harness/implement.diff; echo" }, { pattern: "x", path: ".harness" }, { command: "cd .harness && ls" },
	]) assert.ok(touchesHarness(input), JSON.stringify(input));
	for (const input of [
		{ path: "BRIEF.md" }, { command: "ls -la /work" }, { path: "harness-release/README.md" }, { command: "grep harness agent/run.py" },
		{ command: "cat docs/the-harness.md" }, { path: "agent/run.py" }, {}, undefined,
	]) assert.ok(!touchesHarness(input), JSON.stringify(input));
});
