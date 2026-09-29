import { AGENT_LABEL, AGENT_ORDER, AGENT_RESPONSIBILITY, type AgentName } from "@/lib/api";
import { useInView } from "@/lib/useInView";

/** What each stage is allowed to do, in the words the code uses. */
const STAGE_ROLE: Record<AgentName, string> = {
  validation: "Reads file format, channels and event markers. An invalid recording skips straight to the policy.",
  signal: "Band-passes 8-30 Hz, re-references, epochs, and measures amplitude and flat channels.",
  prediction: "Runs the FBCNet checkpoint over motor-imagery trials and reports the vote agreement.",
  decision: "The only stage that authorises anything. Fixed thresholds, no model in the loop.",
  report: "Assembles the evidence bundle the reviewer reads, with provenance it cannot alter.",
};

export function StageSequence() {
  const { ref, inView } = useInView<HTMLDivElement>();

  return (
    <div ref={ref} className="border-t border-line" data-in-view={inView ? "true" : "false"} data-testid="stage-sequence">
      {/* The one drawn moment here: the sequence rule sweeps once, left to right. */}
      <div className="relative h-px -mt-px">
        <span
          className="absolute inset-y-0 left-0 block bg-accent transition-[width] duration-1000 ease-out motion-reduce:transition-none"
          style={{ width: inView ? "100%" : "0%" }}
        />
      </div>
      {AGENT_ORDER.map((agent, index) => (
        <div
          key={agent}
          className="group grid grid-cols-[auto_1fr] items-baseline gap-x-6 gap-y-1 border-b border-line py-5 transition-transform duration-500 ease-out motion-reduce:transition-none sm:grid-cols-[3rem_10rem_1fr]"
          style={{ transitionDelay: inView ? `${index * 90}ms` : "0ms", transform: inView ? "translateY(0)" : "translateY(8px)" }}
        >
          <span className="num text-[11px] text-muted">{String(index + 1).padStart(2, "0")}</span>
          <span className="text-[15px] font-medium text-ink">{AGENT_LABEL[agent]}</span>
          <span className="col-span-2 text-[13.5px] leading-relaxed text-muted sm:col-span-1">
            <span className="text-muted">{AGENT_RESPONSIBILITY[agent]}. </span>
            {STAGE_ROLE[agent]}
          </span>
        </div>
      ))}
    </div>
  );
}
