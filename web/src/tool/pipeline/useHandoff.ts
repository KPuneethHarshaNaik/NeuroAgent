import { useCallback, useEffect, useRef, useState } from "react";
import type { AgentUpdate } from "@/lib/api";
import {
  CHAIN,
  NODE_CAPTION,
  diffUpdates,
  edgePoint,
  easeInOut,
  elapsedBadge,
  handoffLabel,
  shouldTravel,
  statusToState,
  type NodeStatus,
  type PipelineKey,
  type Rect,
} from "@/tool/pipeline/geometry";

/** The two timings from the vanilla panel. They are the shape of the animation, not decoration. */
export const TRAVEL_MS = 700;
export const HOLD_MS = 400;

export type NodeRuntime = {
  status: NodeStatus;
  message: string;
  update: AgentUpdate | null;
  badge: string;
};

export type Token = { label: string; failed: boolean; x: number; y: number };

const idleNode = (key: PipelineKey): NodeRuntime => ({ status: "idle", message: NODE_CAPTION[key], update: null, badge: "" });

const idleState = (): Record<PipelineKey, NodeRuntime> =>
  Object.fromEntries(CHAIN.map((key) => [key, idleNode(key)])) as Record<PipelineKey, NodeRuntime>;

type Options = {
  updates: AgentUpdate[];
  jobId: string | null;
  /** Live rectangles of the boxes, so a token stays on the line even mid-drag. */
  rectsRef: React.RefObject<Record<PipelineKey, Rect>>;
  reducedMotion: boolean;
};

/**
 * Ported from `queueHandoff` / `drainHandoffs` / `playHandoff` in frontend/app.js.
 *
 * The rule that matters: updates arrive append-only from the poll, so only the tail past the cursor
 * enters the queue, and the queue plays *one update at a time* -- a backend run that finished in
 * 900 ms still resolves node by node, each step waiting for the previous token to land.
 */
export function useHandoff({ updates, jobId, rectsRef, reducedMotion }: Options) {
  const [nodes, setNodes] = useState<Record<PipelineKey, NodeRuntime>>(idleState);
  const [token, setToken] = useState<Token | null>(null);

  const cursorRef = useRef(0);
  const queueRef = useRef<AgentUpdate[]>([]);
  const runningRef = useRef(false);
  const generationRef = useRef(0);
  // A synchronous mirror of node statuses. The queue decides "resolve in place or send a token" from
  // this, not from React state: a stage that reports twice does so in back-to-back updates, and React
  // has not re-rendered in between, so state would still read "idle" and a stray token would fly.
  const statusRef = useRef<Record<PipelineKey, NodeStatus>>(
    Object.fromEntries(CHAIN.map((key) => [key, "idle"])) as Record<PipelineKey, NodeStatus>,
  );

  const applyNode = useCallback((key: PipelineKey, patch: Partial<NodeRuntime>) => {
    if (patch.status) statusRef.current[key] = patch.status;
    setNodes((previous) => ({ ...previous, [key]: { ...previous[key], ...patch } }));
  }, []);

  const reset = useCallback(() => {
    generationRef.current += 1;
    queueRef.current = [];
    cursorRef.current = 0;
    statusRef.current = Object.fromEntries(CHAIN.map((key) => [key, "idle"])) as Record<PipelineKey, NodeStatus>;
    setToken(null);
    setNodes(idleState());
  }, []);

  const play = useCallback(
    async (update: AgentUpdate, generation: number) => {
      const target = update.agent as PipelineKey;
      const state = statusToState(update.status);
      const badge = elapsedBadge(update.duration_ms);
      const label = handoffLabel(update.agent, update.findings ?? {}, update.message);

      // The update is stored before anything animates, so the raw-JSON inspector and the elapsed
      // badge always describe the update that is currently in flight.
      applyNode(target, { update, badge });

      const targetIndex = CHAIN.indexOf(target);
      let source = targetIndex > 0 ? CHAIN[targetIndex - 1] : null;
      // A stage that has not started yet cannot be the source; the hand-off then starts at the entry
      // point, which is what the vanilla panel's origin fallback expressed.
      if (source && source !== "origin" && statusRef.current[source] === "idle") source = "origin";
      const rects = rectsRef.current;
      const sourceRect = source ? rects[source] : undefined;
      const targetRect = rects[target];
      const travel = shouldTravel({
        source,
        target,
        targetStatus: statusRef.current[target],
        sourceStatus: source ? statusRef.current[source] : undefined,
        haveRects: Boolean(sourceRect && targetRect),
        reducedMotion,
      });
      if (!travel || !source || !sourceRect || !targetRect) {
        applyNode(target, { status: state, message: update.message });
        return;
      }

      const startedAt = performance.now();
      let live = true;

      const animated = new Promise<void>((resolve) => {
        const frame = (now: number) => {
          if (!live) return resolve();
          const progress = Math.min(1, (now - startedAt) / TRAVEL_MS);
          const eased = easeInOut(progress);
          // Re-derived every frame from the live rectangles, so dragging a box mid-flight carries
          // the token with it instead of stranding it on a stale line.
          const currentRects = rectsRef.current;
          const from = edgePoint(currentRects[source], { x: currentRects[target].cx, y: currentRects[target].cy });
          const to = edgePoint(currentRects[target], { x: currentRects[source].cx, y: currentRects[source].cy });
          setToken({ label, failed: state === "failed", x: from.x + (to.x - from.x) * eased, y: from.y + (to.y - from.y) * eased });
          if (progress < 1) requestAnimationFrame(frame);
          else resolve();
        };
        const initial = edgePoint(sourceRect, { x: targetRect.cx, y: targetRect.cy });
        setToken({ label, failed: state === "failed", x: initial.x, y: initial.y });
        requestAnimationFrame(frame);
      });

      // A hidden or uncomposited tab never fires requestAnimationFrame; racing the frame loop
      // against the wall clock keeps the queue draining instead of freezing mid-run.
      await Promise.race([animated, new Promise((resolve) => window.setTimeout(resolve, TRAVEL_MS + 500))]);
      live = false;
      if (generation !== generationRef.current) return;
      setToken(null);

      applyNode(target, { status: "working", message: update.message });
      if (state !== "working") {
        await new Promise((resolve) => window.setTimeout(resolve, HOLD_MS));
        if (generation !== generationRef.current) return;
        applyNode(target, { status: state, message: update.message });
      }
    },
    [applyNode, rectsRef, reducedMotion],
  );

  const drain = useCallback(
    async (generation: number) => {
      runningRef.current = true;
      while (queueRef.current.length) {
        const next = queueRef.current.shift();
        if (!next) break;
        await play(next, generation);
        if (generation !== generationRef.current) break;
      }
      runningRef.current = false;
    },
    [play],
  );

  // The diff: only the tail past the cursor is new. `agent_updates` never shrinks, but a shorter
  // list still resets the cursor rather than silently skipping whatever came after it.
  useEffect(() => {
    if (!jobId) return;
    const { fresh, cursor } = diffUpdates(updates, cursorRef.current);
    cursorRef.current = cursor;
    if (!fresh.length) return;
    queueRef.current.push(...fresh);
    if (!runningRef.current) void drain(generationRef.current);
  }, [updates, jobId, drain]);

  useEffect(() => {
    if (!jobId) reset();
  }, [jobId, reset]);

  return { nodes, token, reset };
}
