/**
 * Loaded (with `pi -e`) only into implementation turns, never into the researcher.
 *
 * After every successful edit or write, read the conversation's context usage. The first time it is at
 * or above IMPLEMENTER_STOP_PERCENT (default 40) of the model's window, steer the implementer to stop
 * implementing and report. Nothing is blocked; the implementer is simply asked. The implementer is not
 * told about this rule in advance. Each steer is logged to IMPLEMENTER_LOG (JSONL) if set.
 */
import * as fs from "node:fs";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export const REPORT_REQUEST = `Stop implementing now. Review what you have changed so far against the sub-threads you
were working on. For each sub-thread, report:
- what you changed (files and what the change does),
- why, in terms of that sub-thread,
- its state: done, partly done, or not started,
- what is left and anything the reviewer should check.
End with the report; do not make further changes.`;

export default function (pi: ExtensionAPI) {
	const threshold = Number(process.env.IMPLEMENTER_STOP_PERCENT ?? "40");
	const logPath = process.env.IMPLEMENTER_LOG;
	let steered = false;
	pi.on("tool_result", async (event, ctx) => {
		if (steered || event.isError || (event.toolName !== "edit" && event.toolName !== "write")) return;
		const usage = ctx.getContextUsage();
		if (usage?.percent == null || usage.percent < threshold) return;
		steered = true;
		if (logPath) {
			fs.appendFileSync(logPath, JSON.stringify({ event: "steer", percent: usage.percent, tokens: usage.tokens,
				context_window: usage.contextWindow, after_tool: event.toolName, ts: Date.now() }) + "\n");
		}
		pi.sendUserMessage(REPORT_REQUEST, { deliverAs: "steer" });
	});
}
