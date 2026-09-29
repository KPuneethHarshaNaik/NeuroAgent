import { useEffect, useMemo, useState } from "react";
import { AGENT_ORDER, getHealth, type AgentUpdate, type Health } from "@/lib/api";
import { EvidencePanel } from "@/tool/EvidencePanel";
import { PipelinePanel } from "@/tool/PipelinePanel";
import { ReviewBrief } from "@/tool/ReviewBrief";
import { SignalLens } from "@/tool/SignalLens";
import { UploadPanel } from "@/tool/UploadPanel";
import { statusToState } from "@/tool/pipeline/geometry";
import { useLiveJob } from "@/tool/useLiveJob";

export function Tool() {
  const job = useLiveJob();
  const [health, setHealth] = useState<Health | null>(null);
  const [reducedMotion, setReducedMotion] = useState(false);
  const [resumedFrom, setResumedFrom] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    getHealth(controller.signal)
      .then(setHealth)
      .catch(() => setHealth(null));
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const apply = () => setReducedMotion(query.matches);
    apply();
    query.addEventListener("change", apply);
    return () => query.removeEventListener("change", apply);
  }, []);

  // /app/?job=<id> resumes a job that is already running or already finished.
  useEffect(() => {
    const requested = new URLSearchParams(window.location.search).get("job");
    if (!requested) return;
    setResumedFrom(requested);
    void job.resume(requested);
    // Only on mount: the resume is an entry condition, not a reaction to state changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const latest = useMemo(() => {
    const map: Partial<Record<string, AgentUpdate>> = {};
    for (const update of job.updates) map[update.agent] = update;
    return map;
  }, [job.updates]);

  const signalDone = useMemo(() => {
    // Only try to read the cleaned-signal trace once the signal stage has actually settled.
    // For built-in (in-process) jobs the signal stage never persists signal_cleaned_eeg.fif, so
    // the trace endpoint 404s by design — the lens stays in its "waiting" state, which is the
    // honest representation, rather than surfacing a 404 as if the request had failed.
    const update = latest.signal;
    if (!update) return false;
    const state = statusToState(update.status);
    if (state === "working") return false;
    return true;
  }, [latest]);

  const status = (() => {
    if (job.phase === "uploading") return "Uploading recording…";
    if (job.phase === "failed") return job.error ?? "Something went wrong.";
    if (job.phase === "finished") return `Job ${job.jobId?.slice(0, 8)}… finished: ${job.report?.status ?? "—"}.`;
    if (job.phase === "polling") return `Polling /live-jobs/${job.jobId?.slice(0, 8)}… every 450 ms (${job.updates.length} updates).`;
    return job.jobId ? `Attached to ${job.jobId.slice(0, 8)}…` : "";
  })();

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-30 border-b border-line bg-canvas/90 backdrop-blur-sm">
        <div className="mx-auto flex h-14 max-w-[1560px] items-center justify-between gap-6 px-5">
          <a href="/" className="no-underline">
            <span className="text-[15px] font-semibold tracking-tight text-ink">NeuroAgent</span>
          </a>
          <nav aria-label="Workspace" className="hidden items-baseline gap-3 sm:flex">
            <span className="text-[12.5px] text-ink">Review workspace</span>
            <span className="text-[12.5px] text-muted">Motor imagery</span>
          </nav>
          <div className="flex items-baseline gap-5">
            <span className="num hidden text-[11px] text-muted md:inline" data-testid="tool-health">
              {health ? `pipeline ${health.pipeline_version} · policy ${health.policy_version}` : "backend unreachable"}
            </span>
            <a href="/" className="text-[12.5px] text-muted no-underline hover:text-ink">
              About
            </a>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1560px] px-5 py-5">
        <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
          <div className="flex items-baseline gap-3">
            <h1 className="text-[19px] font-medium tracking-tight">Recording review</h1>
            {resumedFrom ? (
              <span className="num text-[11px] text-muted" data-testid="resumed-job">
                resumed from ?job={resumedFrom.slice(0, 8)}…
              </span>
            ) : null}
          </div>
          <p className="num text-[11px] text-muted" role="status" data-testid="job-status">
            {status}
          </p>
        </div>

        <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,20rem)_minmax(0,1fr)]">
          <div className="grid content-start gap-4">
            <UploadPanel busy={job.phase === "uploading"} status={job.phase === "idle" ? "" : status} onSubmit={(file) => void job.start(file)} />
            <EvidencePanel latest={latest} gates={job.report?.evidence?.gates ?? []} />
          </div>

          <div className="grid content-start gap-4">
            <PipelinePanel updates={job.updates} jobId={job.jobId} reducedMotion={reducedMotion} />
            <SignalLens jobId={job.jobId} signalDone={signalDone} quality={job.report?.quality ?? null} />
            <ReviewBrief jobId={job.jobId} report={job.report} />
          </div>
        </div>

        <footer className="mt-6 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 border-t border-line pt-4">
          <p className="max-w-2xl text-[12px] leading-relaxed text-muted">
            Agents provide evidence. The automated verdict and the reviewer's decision are stored separately — the policy
            verdict is never rewritten by a review.
          </p>
          <span className="num text-[11px] text-muted">
            {AGENT_ORDER.length} stages · {job.updates.length} updates seen
          </span>
        </footer>
      </main>
    </div>
  );
}
