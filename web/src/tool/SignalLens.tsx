import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, getSignalPreview, type QualityReport, type SignalPreview } from "@/lib/api";

type Props = {
  jobId: string | null;
  /** True once the signal stage has completed, from live updates or from a finished report. */
  signalDone: boolean;
  quality: QualityReport | null;
};

type LensState =
  | { state: "waiting" }
  | { state: "loading" }
  | { state: "ready"; preview: SignalPreview }
  | { state: "unavailable"; detail: string };

const TRACE_HEIGHT = 74;

/**
 * Real signal, or nothing. The trace comes from GET /jobs/{job_id}/signal-preview, which reads the
 * cleaned continuous recording the signal stage persisted (8-30 Hz band-pass, average reference,
 * decimated for transport). There is deliberately no illustrative fallback: before the signal stage
 * has written that file, this panel says so.
 */
export function SignalLens({ jobId, signalDone, quality }: Props) {
  const [lens, setLens] = useState<LensState>({ state: "waiting" });
  const requestedFor = useRef<string | null>(null);

  useEffect(() => {
    if (!jobId || !signalDone || requestedFor.current === jobId) return;
    requestedFor.current = jobId;
    const controller = new AbortController();
    setLens({ state: "loading" });
    getSignalPreview(jobId, controller.signal)
      .then((preview) => setLens({ state: "ready", preview }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        if (error instanceof ApiError) setLens({ state: "unavailable", detail: error.message });
        else setLens({ state: "unavailable", detail: "The cleaned signal could not be read." });
      });
    return () => controller.abort();
  }, [jobId, signalDone]);

  const limit = 300;
  const channels = lens.state === "ready" ? lens.preview.channels : [];

  return (
    <section className="surface" aria-labelledby="lens-title">
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 border-b border-line px-4 py-3">
        <div className="flex items-baseline gap-3">
          <h2 id="lens-title" className="text-[15px] font-medium tracking-tight">
            Signal Lens
          </h2>
          <span className="num text-[11px] text-muted">
            {lens.state === "ready"
              ? `${channels.map((channel) => channel.name).join(" · ")} — cleaned continuous signal`
              : "cleaned continuous signal"}
          </span>
        </div>
        {lens.state === "ready" ? (
          <span className="num text-[11px] text-muted">
            {lens.preview.sample_rate_effective} Hz effective · {lens.preview.duration_seconds} s · {lens.preview.unit}
          </span>
        ) : null}
      </div>

      <div className="px-4 py-4">
        {lens.state === "ready" ? (
          <>
            <div className="space-y-3" data-testid="lens-traces">
              {channels.map((channel) => (
                <Trace key={channel.name} name={channel.name} values={channel.values} />
              ))}
            </div>
            {!lens.preview.matched ? (
              <p className="mt-3 border-l-2 border-accent pl-3 text-[12.5px] leading-relaxed text-muted">
                This montage does not expose {lens.preview.requested.join(", ")} by name, so the first three EEG channels
                are shown instead. They are not the motor-cortex sites the classification rests on.
              </p>
            ) : null}
          </>
        ) : null}

        {quality ? (
          <dl className="mt-5 grid grid-cols-2 gap-x-6 gap-y-3 border-t border-line pt-4 sm:grid-cols-4" data-testid="lens-quality">
            <Field label="quality" value={quality.status} />
            <Field label="peak-to-peak" value={quality.peak_to_peak_uv == null ? "—" : `${quality.peak_to_peak_uv.toFixed(3)} µV`} />
            <Field label="amplitude limit" value={`${limit.toFixed(1)} µV`} />
            <Field
              label="rejected epochs"
              value={`${(quality.rejected_epoch_ratio * 100).toFixed(1)}%`}
            />
          </dl>
        ) : null}
      </div>
    </section>
  );
}

/** One channel as an SVG polyline, scaled to its own peak-to-peak so the shape stays readable. */
function Trace({ name, values }: { name: string; values: number[] }) {
  const path = useMemo(() => {
    if (!values.length) return "";
    const peak = Math.max(...values.map((value) => Math.abs(value)), 1e-6);
    return values
      .map((value, index) => {
        const x = (index / Math.max(values.length - 1, 1)) * 1000;
        const y = TRACE_HEIGHT / 2 - (value / peak) * (TRACE_HEIGHT / 2 - 4);
        return `${index === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(" ");
  }, [values]);

  const peak = values.length ? Math.max(...values.map((value) => Math.abs(value))) : 0;

  return (
    <div className="grid grid-cols-[3.5rem_1fr_5.5rem] items-center gap-3">
      <span className="num text-[11.5px] text-ink">{name}</span>
      <svg
        viewBox={`0 0 1000 ${TRACE_HEIGHT}`}
        preserveAspectRatio="none"
        className="h-[74px] w-full"
        aria-label={`${name}, cleaned signal, peak ${peak.toFixed(1)} microvolts`}
        role="img"
      >
        {[TRACE_HEIGHT / 2].map((y) => (
          <line key={y} x1="0" y1={y} x2="1000" y2={y} stroke="#1E262C" strokeWidth="1" vectorEffect="non-scaling-stroke" />
        ))}
        <path d={path} fill="none" stroke="#E4293F" strokeWidth="1.2" vectorEffect="non-scaling-stroke" />
      </svg>
      <span className="num text-right text-[11px] text-muted">±{peak.toFixed(1)} µV</span>
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
