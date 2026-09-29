import { useState } from "react";
import { AGENT_LABEL, AGENT_ORDER, AGENT_RESPONSIBILITY, type AgentUpdate, type GateResult } from "@/lib/api";
import { statusToState } from "@/tool/pipeline/geometry";

type Props = {
  /** The latest update per agent, from the same polled list the pipeline panel consumes. */
  latest: Partial<Record<string, AgentUpdate>>;
  gates: GateResult[];
};

const TONE: Record<string, string> = {
  idle: "text-muted",
  working: "text-accent",
  done: "text-ink",
  failed: "text-accent",
};

export function EvidencePanel({ latest, gates }: Props) {
  const [open, setOpen] = useState<string | null>(null);

  return (
    <section className="surface" aria-labelledby="evidence-title">
      <div className="flex items-baseline justify-between gap-4 border-b border-line px-4 py-3">
        <h2 id="evidence-title" className="text-[15px] font-medium tracking-tight">
          Evidence checks
        </h2>
        <span className="num text-[11px] text-muted">
          {AGENT_ORDER.filter((agent) => latest[agent] && statusToState(latest[agent]!.status) !== "working").length} / 5
          settled
        </span>
      </div>

      <ul className="divide-y divide-line" data-testid="evidence-list">
        {AGENT_ORDER.map((agent) => {
          const update = latest[agent];
          const state = update ? statusToState(update.status) : "idle";
          const findings = Object.entries(update?.findings ?? {});
          const expanded = open === agent;
          return (
            <li key={agent} className="px-4 py-3">
              <button
                type="button"
                disabled={!update}
                onClick={() => setOpen(expanded ? null : agent)}
                className="grid w-full grid-cols-[3rem_minmax(0,1fr)] items-baseline gap-x-3 text-left disabled:cursor-default"
                data-testid={`evidence-${agent}`}
              >
                <span className={`num text-[11px] ${TONE[state]}`}>{state === "done" ? "✓ done" : state}</span>
                <span className="min-w-0">
                  <span className="block text-[13.5px] font-medium">{AGENT_LABEL[agent]}</span>
                  <span className="block truncate text-[12.5px] text-muted" title={update?.message ?? AGENT_RESPONSIBILITY[agent]}>
                    {update?.message ?? AGENT_RESPONSIBILITY[agent]}
                  </span>
                  {update ? (
                    <span className="num mt-1 block text-[10.5px] text-muted">
                      {update.duration_ms ? `${update.duration_ms} ms · ` : ""}
                      {new Date(update.timestamp).toISOString().slice(11, 23)}
                    </span>
                  ) : null}
                </span>
              </button>

              {expanded ? (
                <dl className="mt-2 space-y-1 border-l border-line pl-3" data-testid={`findings-${agent}`}>
                  {findings.length ? (
                    findings.map(([key, value]) => (
                      <div key={key} className="grid grid-cols-[9rem_minmax(0,1fr)] gap-3">
                        <dt className="num text-[11px] text-muted">{key}</dt>
                        <dd className="num text-[11px] text-ink">{format(value)}</dd>
                      </div>
                    ))
                  ) : (
                    <p className="text-[12px] text-muted">This stage published no findings for this job.</p>
                  )}
                </dl>
              ) : null}
            </li>
          );
        })}
      </ul>

      {gates.length ? (
        <div className="border-t border-line px-4 py-3" data-testid="evidence-gates">
          <p className="label-micro">Deterministic gates ({gates.filter((gate) => gate.passed).length}/{gates.length} passed)</p>
          <ul className="mt-2 space-y-1">
            {gates.map((gate) => (
              <li key={gate.name} className="grid grid-cols-[2.5rem_minmax(0,1fr)] gap-3">
                <span className={`num text-[11px] ${gate.passed ? "text-muted" : "text-accent"}`}>{gate.passed ? "pass" : "fail"}</span>
                <span className="min-w-0">
                  <span className="num block text-[11.5px] text-ink">{gate.name}</span>
                  <span className="block text-[11.5px] text-muted">
                    {gate.observed} — {gate.requirement}
                    {gate.blocking ? "" : " (non-blocking)"}
                  </span>
                </span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  );
}

function format(value: unknown): string {
  if (Array.isArray(value)) return value.map((item) => String(item).replaceAll("_", " ")).join(", ");
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
  return String(value).replaceAll("_", " ");
}
