import { useEffect, useState } from "react";
import { ApiError, recordReview, type JobReport, type ReviewAction } from "@/lib/api";

type Props = {
  jobId: string | null;
  report: JobReport | null;
};

const percent = (value: number | null | undefined) => (value == null ? "—" : `${Math.round(value * 100)}%`);

/**
 * The brief and the reviewer's decision. The form posts the same body the vanilla page did --
 * `{action, reviewer_comment[, approved_label]}` -- and surfaces the backend's own refusals instead
 * of a generic failure: 409 when there is no prediction to approve, 422 for an incomplete override.
 */
export function ReviewBrief({ jobId, report }: Props) {
  const [action, setAction] = useState<ReviewAction>("approve");
  const [comment, setComment] = useState("");
  const [override, setOverride] = useState<"left_hand" | "right_hand">("left_hand");
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const [recorded, setRecorded] = useState(report?.human_review ?? null);

  useEffect(() => {
    setRecorded(report?.human_review ?? null);
  }, [report]);

  if (!jobId || !report) {
    return (
      <section className="surface" aria-labelledby="brief-title">
        <div className="border-b border-line px-4 py-3">
          <h2 id="brief-title" className="text-[15px] font-medium tracking-tight">
            Review brief
          </h2>
        </div>
        <p className="px-4 py-4 text-[13px] text-muted">
          The brief appears when the pipeline finishes. Until then the pipeline panel is the record of what has run.
        </p>
      </section>
    );
  }

  const classification = report.classification;
  const predicted = classification?.predicted_label ?? null;
  const confidence = predicted ? classification?.probabilities?.[predicted] : undefined;
  const canApprove = predicted != null;

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setStatus("");
    try {
      const updated = await recordReview(jobId, {
        action,
        reviewer_comment: comment,
        ...(action === "override" ? { approved_label: override } : {}),
      });
      setRecorded(updated.human_review);
      setStatus(`Review recorded: ${updated.human_review?.approved_label ?? "—"}.`);
    } catch (error) {
      setStatus(error instanceof ApiError ? `${error.status}: ${error.message}` : "Could not record the review.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="surface" aria-labelledby="brief-title">
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 border-b border-line px-4 py-3">
        <h2 id="brief-title" className="text-[15px] font-medium tracking-tight">
          Review brief
        </h2>
        <span className="num text-[11px] text-muted">
          {report.input_filename} · {report.status} · sha256 {report.input_hash.slice(0, 12)}…
        </span>
      </div>

      <div className="grid grid-cols-1 gap-px bg-line lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
        <div className="bg-panel px-4 py-5">
          <p className="label-micro">Automated verdict (policy {String((report.evidence?.provenance as { policy_version?: string })?.policy_version ?? "—")})</p>
          <p className="mt-1 text-[26px] leading-none font-semibold tracking-tight" data-testid="brief-verdict">
            {report.decision ?? report.status}
          </p>

          {classification?.status === "classified" ? (
            <>
              <div className="mt-4 space-y-2">
                {Object.entries(classification.probabilities).map(([label, value]) => (
                  <div key={label} className="grid grid-cols-[6rem_1fr_3.5rem] items-center gap-3">
                    <span className="text-[12.5px] text-muted">{label.replaceAll("_", " ")}</span>
                    <span className="h-[3px] bg-panel-well">
                      <span className="block h-full bg-accent" style={{ width: `${value * 100}%` }} />
                    </span>
                    <span className="num text-right text-[12px] text-ink">{value.toFixed(4)}</span>
                  </div>
                ))}
              </div>
              <dl className="mt-5 grid grid-cols-2 gap-x-6 gap-y-3 border-t border-line pt-4 sm:grid-cols-4">
                <Field label="model" value={classification.model} />
                <Field label="confidence" value={percent(confidence)} />
                <Field label="trials" value={String(classification.trial_count)} />
                <Field label="agreement" value={percent(classification.agreement)} />
                <Field label="band" value={classification.calibration?.band ?? "—"} />
                <Field label="band accuracy" value={percent(classification.calibration?.band_observed_accuracy)} />
                <Field label="band n" value={String(classification.calibration?.band_sample_size ?? "—")} />
                <Field label="temperature" value={classification.calibration ? classification.calibration.temperature.toFixed(4) : "—"} />
              </dl>
            </>
          ) : (
            <p className="mt-3 text-[13px] text-muted">{classification?.reason ?? "No classification was produced."}</p>
          )}

          {[...(report.evidence?.signals ?? []), ...report.warnings].map((signal) => (
            <p key={signal} className="mt-3 border-l-2 border-accent pl-3 text-[12.5px] leading-relaxed text-ink">
              {signal}
            </p>
          ))}
        </div>

        <div className="bg-panel px-4 py-5">
          <p className="label-micro">Reviewer decision</p>
          {recorded ? (
            <p className="mt-2 text-[12.5px] text-muted" data-testid="review-recorded">
              Recorded {recorded.action.replaceAll("_", " ")} → <span className="num text-ink">{recorded.approved_label}</span>{" "}
              at {new Date(recorded.created_at).toISOString().slice(11, 19)}Z
              {recorded.reviewer_comment ? ` — “${recorded.reviewer_comment}”` : ""}
            </p>
          ) : null}

          <form className="mt-3 grid gap-3" onSubmit={submit} data-testid="review-form">
            <label className="grid gap-1 text-[12px]">
              <span className="text-muted">Decision</span>
              <select
                value={action}
                onChange={(event) => setAction(event.target.value as ReviewAction)}
                className="border border-line bg-panel-well px-2 py-2 text-[13px] text-ink"
                data-testid="review-action"
              >
                <option value="approve" disabled={!canApprove}>
                  Approve prediction{canApprove ? "" : " (no prediction)"}
                </option>
                <option value="mark_uncertain">Mark uncertain</option>
                <option value="override">Override hand label</option>
              </select>
            </label>

            {action === "override" ? (
              <label className="grid gap-1 text-[12px]">
                <span className="text-muted">Override label</span>
                <select
                  value={override}
                  onChange={(event) => setOverride(event.target.value as "left_hand" | "right_hand")}
                  className="border border-line bg-panel-well px-2 py-2 text-[13px] text-ink"
                  data-testid="review-override"
                >
                  <option value="left_hand">left_hand</option>
                  <option value="right_hand">right_hand</option>
                </select>
              </label>
            ) : null}

            <label className="grid gap-1 text-[12px]">
              <span className="text-muted">Reason</span>
              <textarea
                value={comment}
                onChange={(event) => setComment(event.target.value)}
                maxLength={2000}
                rows={3}
                placeholder="What did you observe?"
                className="border border-line bg-panel-well px-2 py-2 text-[13px] text-ink"
                data-testid="review-comment"
              />
            </label>

            <button
              type="submit"
              disabled={busy}
              className="justify-self-start border border-line px-4 py-2 text-[13px] text-ink transition-colors hover:border-accent disabled:opacity-50"
              data-testid="review-submit"
            >
              {busy ? "Recording…" : "Record review"}
            </button>
            <p className="min-h-[18px] text-[12px] text-muted" role="status" data-testid="review-status">
              {status}
            </p>
          </form>
        </div>
      </div>
    </section>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="label-micro">{label}</dt>
      <dd className="num mt-0.5 text-[12.5px] text-ink">{value}</dd>
    </div>
  );
}
