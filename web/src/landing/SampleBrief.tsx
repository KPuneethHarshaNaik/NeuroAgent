import { SAMPLE_CALIBRATION, SAMPLE_GATES, SAMPLE_JOB, SAMPLE_QUALITY, SAMPLE_SIGNAL, SAMPLE_VERDICT } from "@/landing/sample";
import { useInView } from "@/lib/useInView";

const percent = (value: number | null) => (value == null ? "—" : `${Math.round(value * 100)}%`);
const fixed = (value: number, digits = 3) => value.toFixed(digits);

/**
 * A real brief, rendered the way /app/ renders one -- same hairlines, same mono for numbers, the
 * verdict stamping in once when it reaches the viewport. Every value below comes from the stored
 * evidence bundle for this job; none of it is illustrative.
 */
export function SampleBrief() {
  const { ref, inView } = useInView<HTMLDivElement>();

  return (
    <div className="surface" ref={ref} data-testid="sample-brief">
      <div className="flex flex-wrap items-baseline justify-between gap-x-8 gap-y-2 border-b border-line px-5 py-4">
        <div>
          <p className="label-micro">Stored review brief</p>
          <p className="mt-1 text-[19px] font-medium tracking-tight">S001R03.edf</p>
        </div>
        <dl className="grid grid-cols-2 gap-x-8 gap-y-1 text-right sm:grid-cols-3">
          <Field label="job" value={`${SAMPLE_JOB.jobId.slice(0, 8)}…`} />
          <Field label="input sha256" value={SAMPLE_JOB.inputHashShort} />
          <Field label="pipeline" value="0.2.0" />
        </dl>
      </div>

      <div className="grid grid-cols-1 gap-px bg-line lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <div className="bg-panel px-5 py-6">
          <p className="label-micro">Automated verdict</p>
          <p
            className="mt-2 text-[30px] leading-none font-semibold tracking-tight transition-transform duration-700 ease-out motion-reduce:transition-none"
            style={{ transform: inView ? "translateY(0)" : "translateY(6px)" }}
          >
            {SAMPLE_VERDICT.decision}
          </p>
          <p className="mt-3 max-w-md text-[13.5px] leading-relaxed text-muted">
            The policy never returns a bare ACCEPT for this checkpoint: the trial agreement gate is not blocking, so the
            verdict carries its disagreement into the brief instead of hiding it.
          </p>

          <div className="mt-6 space-y-3">
            <Bar label="left_hand" value={SAMPLE_VERDICT.probabilities.left_hand} />
            <Bar label="right_hand" value={SAMPLE_VERDICT.probabilities.right_hand} />
          </div>

          <dl className="mt-6 grid grid-cols-2 gap-x-8 gap-y-4 border-t border-line pt-5 sm:grid-cols-4">
            <Field label="trials" value={String(SAMPLE_VERDICT.trialCount)} />
            <Field label="agreement" value={percent(SAMPLE_VERDICT.agreement)} />
            <Field label="peak-to-peak" value={`${fixed(SAMPLE_QUALITY.peakToPeakUv)} µV`} />
            <Field label="flat channels" value={String(SAMPLE_QUALITY.flatChannelCount)} />
            <Field label="sampling" value={`${SAMPLE_SIGNAL.samplingRateHz} Hz`} />
            <Field label="channels" value={String(SAMPLE_SIGNAL.eegChannelCount)} />
            <Field label="epochs" value={String(SAMPLE_SIGNAL.epochCount)} />
            <Field label="quality" value={SAMPLE_QUALITY.status} />
          </dl>

          {SAMPLE_VERDICT.warning ? (
            <p className="mt-6 border-l-2 border-accent pl-3 text-[13px] leading-relaxed text-ink">
              {SAMPLE_VERDICT.warning}
            </p>
          ) : null}
        </div>

        <div className="bg-panel px-5 py-6">
          <p className="label-micro">Deterministic gates</p>
          <ul className="mt-3 divide-y divide-line">
            {SAMPLE_GATES.map((gate) => (
              <li key={gate.name} className="grid grid-cols-[auto_1fr] items-baseline gap-x-3 py-2.5">
                <span className={gate.passed ? "num text-[11px] text-muted" : "num text-[11px] text-accent"}>
                  {gate.passed ? "pass" : "fail"}
                </span>
                <span>
                  <span className="num block text-[12px] text-ink">{gate.name}</span>
                  <span className="block text-[12.5px] leading-relaxed text-muted">
                    {gate.observed} — {gate.requirement}
                    {gate.blocking ? "" : " (non-blocking)"}
                  </span>
                </span>
              </li>
            ))}
          </ul>

          <p className="label-micro mt-6">What the probability is worth</p>
          <dl className="mt-3 grid grid-cols-2 gap-x-8 gap-y-4">
            <Field label="temperature" value={fixed(SAMPLE_CALIBRATION.temperature, 4)} />
            <Field label="band" value={SAMPLE_CALIBRATION.band} />
            <Field label="band accuracy" value={percent(SAMPLE_CALIBRATION.bandObservedAccuracy)} />
            <Field label="band n" value={String(SAMPLE_CALIBRATION.bandSampleSize)} />
            <Field label="test accuracy" value={percent(SAMPLE_CALIBRATION.trainedAccuracy)} />
            <Field label="calibration error" value={fixed(SAMPLE_CALIBRATION.expectedCalibrationError, 4)} />
          </dl>
          <p className="mt-4 text-[13px] leading-relaxed text-muted">
            A 0.54 probability lands in the low band, where 144 held-out trials scored{" "}
            {percent(SAMPLE_CALIBRATION.bandObservedAccuracy)}. That number — not the 0.54 — is what a reviewer should
            weigh.
          </p>
        </div>
      </div>
    </div>
  );
}

function Bar({ label, value }: { label: string; value: number }) {
  const { ref, inView } = useInView<HTMLDivElement>();
  const width = `${(value * 100).toFixed(1)}%`;
  return (
    <div ref={ref} className="grid grid-cols-[7rem_1fr_3.5rem] items-center gap-3">
      <span className="text-[13px] text-muted">{label}</span>
      <span className="h-[3px] w-full bg-panel-well">
        <span
          className="block h-full bg-accent transition-[width] duration-700 ease-out motion-reduce:transition-none"
          style={{ width: inView ? width : "0%", transitionDelay: "120ms" }}
        />
      </span>
      <span className="num text-right text-[12px] text-ink">{value.toFixed(4)}</span>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="label-micro">{label}</dt>
      <dd className="num mt-1 text-[13px] text-ink">{value}</dd>
    </div>
  );
}
