import { AGENT_ORDER, type AgentName } from "@/lib/api";

/**
 * The layout and connector maths from the vanilla panel in frontend/app.js, as pure functions so
 * they can be reasoned about (and tested) without a DOM. Ported, not reimagined:
 *
 *   - nodes are positioned by *centre* fractions (fx, fy) of the canvas, so a resize keeps the
 *     arrangement and a drag needs no re-measuring;
 *   - a connector between two nodes is the centre-to-centre segment clipped to each node's
 *     rectangle, which is what makes the lines meet the boxes' edges at any angle;
 *   - the default layout is measured, not fixed: as many columns as fit, wrapping into rows.
 */

export type PipelineKey = "origin" | AgentName;
export type NodeStatus = "idle" | "working" | "done" | "failed";
export type Position = { fx: number; fy: number };
export type Size = { width: number; height: number };
export type Rect = { cx: number; cy: number; halfWidth: number; halfHeight: number };
export type Point = { x: number; y: number };

export const PIPELINE_NODE_WIDTH = 156;
export const PIPELINE_NODE_HEIGHT = 94;
export const PIPELINE_ORIGIN_WIDTH = 130;
export const PIPELINE_ORIGIN_HEIGHT = 78;
export const PIPELINE_ROW_HEIGHT = 122;
export const PIPELINE_MIN_HEIGHT = 240;
export const PIPELINE_COLUMN_WIDTH = 172;

export const CHAIN: PipelineKey[] = ["origin", ...AGENT_ORDER];

export const NODE_TITLE: Record<PipelineKey, string> = {
  origin: "Upload",
  validation: "Validation",
  signal: "Signal",
  prediction: "Prediction",
  decision: "Decision",
  report: "Report",
};

export const NODE_STEP: Record<PipelineKey, string> = {
  origin: "↑",
  validation: "01",
  signal: "02",
  prediction: "03",
  decision: "04",
  report: "05",
};

export const NODE_CAPTION: Record<PipelineKey, string> = {
  origin: "start · POST /live-jobs",
  validation: "Format, channels, markers",
  signal: "8-30 Hz band-pass, epochs",
  prediction: "FBCNet trial votes",
  decision: "Fixed policy thresholds",
  report: "Evidence bundle",
};

export function nodeSize(key: PipelineKey): Size {
  return key === "origin"
    ? { width: PIPELINE_ORIGIN_WIDTH, height: PIPELINE_ORIGIN_HEIGHT }
    : { width: PIPELINE_NODE_WIDTH, height: PIPELINE_NODE_HEIGHT };
}

/** Column/row split and canvas height for a given width -- the vanilla `layoutPipeline`. */
export function layoutPlan(containerWidth: number, count = CHAIN.length): { columns: number; rows: number; height: number } {
  const columns = Math.max(1, Math.min(count, Math.floor(containerWidth / PIPELINE_COLUMN_WIDTH)));
  const rows = Math.ceil(count / columns);
  return { columns, rows, height: Math.max(PIPELINE_MIN_HEIGHT, rows * PIPELINE_ROW_HEIGHT) };
}

/** Default centre fractions: one row when there is room, wrapping when there is not. */
export function defaultPositions(containerWidth: number, count = CHAIN.length): Position[] {
  const { columns, rows } = layoutPlan(containerWidth, count);
  return Array.from({ length: count }, (_, index) => {
    const row = Math.floor(index / columns);
    const inRow = Math.min(columns, count - row * columns);
    return { fx: ((index % columns) + 0.5) / inRow, fy: (row + 0.5) / rows };
  });
}

export function rectOf(position: Position, size: Size, container: Size): Rect {
  return {
    cx: position.fx * container.width,
    cy: position.fy * container.height,
    halfWidth: size.width / 2,
    halfHeight: size.height / 2,
  };
}

/** Where the line from a node's centre towards `toward` leaves that node's rectangle. */
export function edgePoint(rect: Rect, toward: Point): Point {
  const dx = toward.x - rect.cx;
  const dy = toward.y - rect.cy;
  if (!dx && !dy) return { x: rect.cx, y: rect.cy };
  const scale = Math.min(
    dx ? rect.halfWidth / Math.abs(dx) : Number.POSITIVE_INFINITY,
    dy ? rect.halfHeight / Math.abs(dy) : Number.POSITIVE_INFINITY,
  );
  return { x: rect.cx + dx * scale, y: rect.cy + dy * scale };
}

export function connector(from: Rect, to: Rect): { start: Point; end: Point } {
  return { start: edgePoint(from, { x: to.cx, y: to.cy }), end: edgePoint(to, { x: from.cx, y: from.cy }) };
}

/** Keep a node's whole box inside the canvas while dragging, without re-measuring per move. */
export function clampCentre(value: number, size: number, extent: number): number {
  const margin = size / 2 / Math.max(extent, 1);
  if (margin >= 0.5) return 0.5;
  return Math.min(1 - margin, Math.max(margin, value));
}

export function easeInOut(progress: number): number {
  return progress < 0.5 ? 2 * progress * progress : 1 - (-2 * progress + 2) ** 2 / 2;
}

/** backend AgentUpdate.status -> node state. `blocked` is a refusal, so it reads as failed. */
export function statusToState(status: string): NodeStatus {
  if (status === "completed") return "done";
  if (status === "working") return "working";
  return "failed";
}

/** The short token label, taken from the most informative real field on the update. */
export function handoffLabel(agent: AgentName, findings: Record<string, unknown>, message: string): string {
  const text = (value: unknown) => String(value).replaceAll("_", " ");
  const number = (value: unknown) => (typeof value === "number" ? value : null);
  const labels: Record<AgentName, string | null> = {
    validation:
      number(findings.event_markers) != null
        ? `${number(findings.event_markers)} markers`
        : number(findings.eeg_channels) != null
          ? `${number(findings.eeg_channels)} channels`
          : null,
    signal:
      number(findings.epochs) != null
        ? `${number(findings.epochs)} epochs${findings.quality ? ` · ${text(findings.quality)}` : ""}`
        : null,
    prediction:
      findings.prediction != null
        ? `${text(findings.prediction)}${number(findings.agreement) == null ? "" : ` · ${Math.round(number(findings.agreement)! * 100)}%`}`
        : null,
    decision: findings.decision ? text(findings.decision).toLowerCase() : null,
    report: findings.decision ? `brief · ${text(findings.decision).toLowerCase()}` : null,
  };
  const label = labels[agent] ?? message ?? "";
  return label.length > 30 ? `${label.slice(0, 29)}…` : label;
}

export function elapsedBadge(durationMs: number | undefined): string {
  return durationMs ? `${durationMs} ms` : "";
}

/**
 * Whether an update should send a token along a connector, or settle in place. Ported from the
 * vanilla `playHandoff` guard, and pure so the rule can be executed without a browser: a stage that
 * reports twice (working, then completed), a node already mid-flight, a missing rectangle, or a
 * viewer who asked for reduced motion all settle in place rather than animating a hand-off.
 */
export function shouldTravel(options: {
  source: PipelineKey | null;
  target: PipelineKey;
  targetStatus: NodeStatus;
  sourceStatus?: NodeStatus;
  haveRects: boolean;
  reducedMotion: boolean;
}): boolean {
  const { source, target, targetStatus, sourceStatus, haveRects, reducedMotion } = options;
  if (reducedMotion || !haveRects) return false;
  if (!source || source === target) return false;
  if (targetStatus === "working") return false;
  if (source !== "origin" && sourceStatus === "idle") return false;
  return true;
}

export type QueueStep = { updates: unknown[]; cursor: number };

/**
 * The append-only diff, ported verbatim from the vanilla panel: `agent_updates` only ever grows,
 * so only the tail past the cursor is new; a shorter list than the cursor means the frame of
 * reference changed (a new job), so the cursor resets rather than skipping or replaying.
 */
export function diffUpdates<T>(updates: T[], cursor: number): { fresh: T[]; cursor: number } {
  if (!updates.length) return { fresh: [], cursor: 0 };
  const from = updates.length < cursor ? 0 : cursor;
  return { fresh: updates.slice(from), cursor: updates.length };
}
