import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { AgentUpdate } from "@/lib/api";
import {
  CHAIN,
  NODE_CAPTION,
  NODE_STEP,
  NODE_TITLE,
  clampCentre,
  connector,
  defaultPositions,
  layoutPlan,
  nodeSize,
  rectOf,
  type NodeStatus,
  type PipelineKey,
  type Position,
  type Rect,
  type Size,
} from "@/tool/pipeline/geometry";
import { useHandoff, type NodeRuntime } from "@/tool/pipeline/useHandoff";

type Props = {
  updates: AgentUpdate[];
  jobId: string | null;
  reducedMotion: boolean;
};

const STATE_LABEL: Record<NodeStatus, string> = { idle: "idle", working: "working", done: "done", failed: "failed" };

/** Hairline + accent, per the palette: idle is muted, working is the accent animating, failed is the accent. */
const NODE_TONE: Record<NodeStatus, string> = {
  idle: "border-dashed border-line bg-panel text-ink",
  working: "border-solid border-accent bg-panel text-ink",
  done: "border-solid border-line bg-panel text-ink",
  failed: "border-solid border-accent bg-accent/10 text-ink",
};

export function PipelinePanel({ updates, jobId, reducedMotion }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const nodeRefs = useRef<Partial<Record<PipelineKey, HTMLDivElement | null>>>({});
  const [container, setContainer] = useState<Size>({ width: 1200, height: 240 });
  const [positions, setPositions] = useState<Position[]>(() => defaultPositions(1200));
  const [dragged, setDragged] = useState(false);
  const [selected, setSelected] = useState<PipelineKey | null>(null);

  const plan = layoutPlan(container.width);
  const rects = useMemo(() => {
    const map = {} as Record<PipelineKey, Rect>;
    CHAIN.forEach((key, index) => {
      map[key] = rectOf(positions[index] ?? { fx: 0, fy: 0 }, nodeSize(key), container);
    });
    return map;
  }, [positions, container]);
  const rectsRef = useRef(rects);
  rectsRef.current = rects;

  const { nodes, token, reset } = useHandoff({ updates, jobId, rectsRef, reducedMotion });

  // Measure, then (until the user moves anything) lay out for the measured width.
  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    const measure = () => setContainer({ width: element.clientWidth, height: element.clientHeight });
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (dragged) return;
    setPositions(defaultPositions(container.width));
  }, [container.width, dragged]);

  useEffect(() => {
    reset();
  }, [jobId, reset]);

  const links = useMemo(
    () => CHAIN.slice(0, -1).map((key, index) => ({ key, to: CHAIN[index + 1], ...connector(rects[key], rects[CHAIN[index + 1]]) })),
    [rects],
  );

  /** Pointer drag, ported from the vanilla `startDrag`: capture, clamp, store fractions. */
  const startDrag = useCallback(
    (event: React.PointerEvent<HTMLDivElement>, index: number) => {
      if (event.pointerType === "mouse" && event.button !== 0) return;
      const element = event.currentTarget;
      const box = containerRef.current?.getBoundingClientRect();
      if (!box) return;
      const rect = rects[CHAIN[index]];
      const origin = { x: event.clientX - box.left, y: event.clientY - box.top };
      const offset = { x: rect.cx - origin.x, y: rect.cy - origin.y };
      let moved = false;

      const onMove = (moveEvent: PointerEvent) => {
        const next = { x: moveEvent.clientX - box.left, y: moveEvent.clientY - box.top };
        if (Math.hypot(next.x - origin.x, next.y - origin.y) > 3) moved = true;
        const size = nodeSize(CHAIN[index]);
        setPositions((previous) =>
          previous.map((position, positionIndex) =>
            positionIndex === index
              ? {
                  fx: clampCentre((next.x + offset.x) / box.width, size.width, box.width),
                  fy: clampCentre((next.y + offset.y) / box.height, size.height, box.height),
                }
              : position,
          ),
        );
      };
      const onUp = () => {
        element.removeEventListener("pointermove", onMove);
        element.removeEventListener("pointerup", onUp);
        element.removeEventListener("pointercancel", onUp);
        if (moved) setDragged(true);
        else if (nodes[CHAIN[index]].update) setSelected(CHAIN[index]);
      };

      element.addEventListener("pointermove", onMove);
      element.addEventListener("pointerup", onUp);
      element.addEventListener("pointercancel", onUp);
      try {
        element.setPointerCapture(event.pointerId);
      } catch {
        /* the drag still works while the pointer stays over the box */
      }
    },
    [nodes, rects],
  );

  const selectedUpdate = selected ? nodes[selected].update : null;

  return (
    <section className="surface" aria-labelledby="pipeline-title">
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 border-b border-line px-4 py-3">
        <div className="flex items-baseline gap-3">
          <h2 id="pipeline-title" className="text-[15px] font-medium tracking-tight">
            Agent pipeline
          </h2>
          <span className="num text-[11px] text-muted">
            {updates.length} update{updates.length === 1 ? "" : "s"} polled
          </span>
        </div>
        <span className="text-[11.5px] text-muted">Drag a box · click a finished box for its raw update</span>
      </div>

      <div
        ref={containerRef}
        className="relative overflow-hidden border-t border-line"
        style={{ height: plan.height }}
        data-testid="pipeline-canvas"
      >
        <svg className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true" data-testid="pipeline-links">
          {links.map((link) => {
            const to = nodes[link.to].status;
            return (
              <line
                key={link.key}
                x1={link.start.x}
                y1={link.start.y}
                x2={link.end.x}
                y2={link.end.y}
                stroke={to === "done" ? "#3C4650" : to === "working" ? "#E4293F" : "#1E262C"}
                strokeWidth={2}
                strokeDasharray={to === "idle" ? "4 5" : undefined}
              />
            );
          })}
        </svg>

        {CHAIN.map((key, index) => (
          <PipelineNodeBox
            key={key}
            nodeKey={key}
            runtime={nodes[key]}
            position={positions[index] ?? { fx: 0, fy: 0 }}
            onPointerDown={(event) => startDrag(event, index)}
            onActivate={() => nodes[key].update && setSelected(key)}
            selected={selected === key}
            register={(element) => {
              nodeRefs.current[key] = element;
            }}
          />
        ))}

        {token ? (
          <span
            className="num pointer-events-none absolute z-20 max-w-[190px] truncate border bg-canvas px-2 py-[3px] text-[10px]"
            style={{
              left: token.x,
              top: token.y,
              transform: "translate(-50%, -50%)",
              borderColor: token.failed ? "#E4293F" : "#E4293F",
              color: token.failed ? "#E4293F" : "#ECEFEE",
            }}
            data-testid="handoff-token"
          >
            {token.label}
          </span>
        ) : null}
      </div>

      {selectedUpdate ? (
        <div className="border-t border-line bg-panel-well px-4 py-3" data-testid="pipeline-json">
          <div className="flex items-baseline justify-between gap-4">
            <p className="label-micro">{NODE_TITLE[selected!]} · raw AgentUpdate</p>
            <button type="button" className="num text-[11px] text-muted hover:text-accent" onClick={() => setSelected(null)}>
              close
            </button>
          </div>
          <pre className="num mt-2 max-h-[190px] overflow-auto text-[11px] leading-relaxed text-ink">
            {JSON.stringify(selectedUpdate, null, 2)}
          </pre>
        </div>
      ) : null}
    </section>
  );
}

function PipelineNodeBox({
  nodeKey,
  runtime,
  position,
  onPointerDown,
  onActivate,
  selected,
  register,
}: {
  nodeKey: PipelineKey;
  runtime: NodeRuntime;
  position: Position;
  onPointerDown: (event: React.PointerEvent<HTMLDivElement>) => void;
  onActivate: () => void;
  selected: boolean;
  register: (element: HTMLDivElement | null) => void;
}) {
  const size = nodeSize(nodeKey);
  const isOrigin = nodeKey === "origin";
  return (
    <div
      ref={register}
      role="button"
      tabIndex={0}
      aria-label={`${NODE_TITLE[nodeKey]}: ${STATE_LABEL[runtime.status]}`}
      data-testid={`pipeline-node-${nodeKey}`}
      data-status={runtime.status}
      onPointerDown={onPointerDown}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onActivate();
        }
      }}
      className={`absolute z-10 grid cursor-grab touch-none select-none content-start gap-1 border p-2.5 transition-colors ${NODE_TONE[runtime.status]} ${
        selected ? "ring-1 ring-accent" : ""
      } ${runtime.status === "working" && !isOrigin ? "animate-[pulseBorder_1.5s_ease-in-out_infinite]" : ""}`}
      style={{ left: `${position.fx * 100}%`, top: `${position.fy * 100}%`, width: size.width, height: size.height, transform: "translate(-50%, -50%)" }}
    >
      <div className="grid grid-cols-[20px_minmax(0,1fr)] items-center gap-1.5">
        <span className="num grid size-[18px] place-items-center rounded-full border border-line text-[9px] text-muted">{NODE_STEP[nodeKey]}</span>
        <strong className="truncate text-[12.5px] font-medium">{NODE_TITLE[nodeKey]}</strong>
      </div>
      <div className="flex items-baseline justify-between gap-2">
        <em className="num truncate text-[9px] not-italic tracking-wide text-muted">
          {runtime.status === "done" ? "✓ done" : STATE_LABEL[runtime.status]}
        </em>
        {runtime.badge ? <span className="num shrink-0 border border-line px-1 text-[9px] text-muted">{runtime.badge}</span> : null}
      </div>
      <p className="line-clamp-2 text-[11px] leading-[1.35] text-muted" title={runtime.message}>
        {runtime.message || NODE_CAPTION[nodeKey]}
      </p>
    </div>
  );
}
