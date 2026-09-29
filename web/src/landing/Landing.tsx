import { Suspense, lazy, useEffect, useState } from "react";
import { SampleBrief } from "@/landing/SampleBrief";
import { StageSequence } from "@/landing/StageSequence";
import { POLICY_THRESHOLDS } from "@/landing/sample";
import { getHealth, type Health } from "@/lib/api";

// three.js is ~700 kB of the bundle and belongs only to this one moment, so the hero streams in
// after first paint. The hero copy is HTML and stays readable while it does.
const HeroScene = lazy(() => import("@/landing/HeroScene").then((module) => ({ default: module.HeroScene })));

export function Landing() {
  const health = useBackendHealth();

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-line bg-canvas/85 backdrop-blur-sm">
        <div className="mx-auto flex h-14 max-w-[1240px] items-center justify-between gap-6 px-5">
          <a href="/" className="flex items-baseline gap-2 no-underline">
            <span className="text-[15px] font-semibold tracking-tight text-ink">NeuroAgent</span>
            <span className="label-micro hidden sm:inline">motor imagery · human reviewed</span>
          </a>
          <div className="flex items-center gap-5">
            <span className="num hidden text-[11px] text-muted md:inline" data-testid="health-line">
              {health.state === "ready"
                ? `pipeline ${health.value.pipeline_version} · policy ${health.value.policy_version}`
                : health.state === "loading"
                  ? "reading backend…"
                  : "backend unreachable"}
            </span>
            <a
              href="/app/"
              className="border border-accent bg-accent px-4 py-2 text-[13px] font-medium text-white no-underline transition-colors hover:bg-[#c42136]"
            >
              Launch tool
            </a>
          </div>
        </div>
      </header>

      <section className="relative border-b border-line">
        <div className="pointer-events-none absolute inset-0 overflow-hidden">
          <Suspense fallback={null}>
            <HeroScene />
          </Suspense>
        </div>
        <div className="relative mx-auto max-w-[1240px] px-5 pt-28 pb-24 sm:pt-40 sm:pb-32">
          <p className="label-micro">EEG review workspace</p>
          <h1 className="mt-5 max-w-3xl text-[38px] leading-[1.05] font-semibold tracking-tight sm:text-[54px]">
            Motor-imagery EEG classification that a clinician can actually check.
          </h1>
          <p className="mt-6 max-w-2xl text-[15.5px] leading-relaxed text-muted">
            Five bounded stages read one recording: validation, signal, prediction, decision, report. Every number in the
            brief — amplitude, trial agreement, probability — is written into an evidence bundle with the thresholds that
            judged it. A human records the final call, and that call is stored next to the machine's.
          </p>
          <div className="mt-9 flex flex-wrap items-center gap-x-8 gap-y-4">
            <a
              href="/app/"
              className="border border-accent bg-accent px-5 py-3 text-[14px] font-medium text-white no-underline transition-colors hover:bg-[#c42136]"
            >
              Launch tool
            </a>
            <a href="#auditable" className="text-[14px] text-ink underline decoration-line underline-offset-4 hover:decoration-accent">
              How the verdict is fixed
            </a>
          </div>
          <p className="mt-10 max-w-xl text-[12.5px] leading-relaxed text-muted">
            The trace behind this page is drawn, not recorded. The tool renders your own cleaned signal once the signal
            stage has run.
          </p>
        </div>
      </section>

      <section className="mx-auto max-w-[1240px] px-5 py-20">
        <div className="grid gap-10 lg:grid-cols-[minmax(0,20rem)_minmax(0,1fr)]">
          <div>
            <h2 className="text-[24px] font-semibold tracking-tight">The pipeline</h2>
            <p className="mt-3 text-[13.5px] leading-relaxed text-muted">
              Each stage is a function with one responsibility, and each one publishes its own status update. The
              reviewer sees the same sequence the machine ran, including the stage that refused to continue.
            </p>
          </div>
          <StageSequence />
        </div>
      </section>

      <section id="auditable" className="border-y border-line bg-panel/40">
        <div className="mx-auto grid max-w-[1240px] gap-12 px-5 py-20 lg:grid-cols-[minmax(0,1fr)_minmax(0,26rem)]">
          <div>
            <h2 className="text-[24px] font-semibold tracking-tight">Why the verdict is auditable</h2>
            <p className="mt-4 max-w-2xl text-[15px] leading-relaxed text-muted">
              The authorising code is a small module of frozen constants — <span className="num text-ink">backend/policy.py</span>,
              policy version {health.state === "ready" ? health.value.policy_version : "2.1"}. It does not call a language
              model, it does not learn, and it cannot be talked into a friendlier answer. Its thresholds are copied into
              every evidence bundle it produces, so a verdict can be re-derived from the bundle alone.
            </p>
            <p className="mt-4 max-w-2xl text-[15px] leading-relaxed text-muted">
              A recheck is a bounded diagnostic re-run — one per job — not a retry until acceptance. When the budget is
              spent, the only remaining escalation is a person.
            </p>

            <dl className="mt-10 divide-y divide-line border-t border-line" data-testid="threshold-table">
              {POLICY_THRESHOLDS.map((threshold) => (
                <div key={threshold.key} className="grid grid-cols-[1fr_auto] items-baseline gap-x-6 py-3 sm:grid-cols-[14rem_5rem_1fr]">
                  <dt className="num text-[12.5px] text-ink">{threshold.key}</dt>
                  <dd className="num text-[12.5px] text-accent sm:text-left">{threshold.value}</dd>
                  <dd className="col-span-2 text-[13px] text-muted sm:col-span-1">{threshold.note}</dd>
                </div>
              ))}
            </dl>
          </div>

          <aside className="surface h-fit px-5 py-6">
            <p className="label-micro">What this checkpoint is worth</p>
            <p className="mt-3 text-[13.5px] leading-relaxed text-muted">
              The shipped model scores <span className="num text-ink">0.6102</span> on a held-out subject split. Temperature
              scaling (<span className="num text-ink">T=1.6995</span>) brings expected calibration error down to{" "}
              <span className="num text-ink">0.0785</span> without changing the accuracy.
            </p>
            <p className="mt-4 text-[13.5px] leading-relaxed text-muted">
              Its confidence is discriminative only just: top-quartile accuracy clears the base rate by{" "}
              <span className="num text-ink">0.1017</span> against a rule that demands{" "}
              <span className="num text-ink">≥ 0.1</span>. That is why the signal you see in the brief is the reliability
              band's measured accuracy, not the raw probability.
            </p>
            <p className="mt-4 text-[13.5px] leading-relaxed text-muted">
              Nothing here is tuned to look better than it is. A 61% classifier with an honest band and a reviewer in the
              loop is the product.
            </p>
          </aside>
        </div>
      </section>

      <section className="mx-auto max-w-[1240px] px-5 py-20">
        <div className="mb-8 grid gap-6 lg:grid-cols-[minmax(0,20rem)_minmax(0,1fr)]">
          <h2 className="text-[24px] font-semibold tracking-tight">A brief the tool actually produced</h2>
          <p className="text-[13.5px] leading-relaxed text-muted">
            Read from the stored evidence bundle for <span className="num text-ink">S001R03.edf</span> — 64 channels,
            30 event markers, 15 usable motor-imagery trials. The policy returned{" "}
            <span className="num text-ink">ACCEPT_WITH_WARNING</span> because the trials disagreed with each other, and the
            brief says so rather than rounding the disagreement away.
          </p>
        </div>
        <SampleBrief />
      </section>

      <footer className="border-t border-line">
        <div className="mx-auto flex max-w-[1240px] flex-wrap items-center justify-between gap-4 px-5 py-7">
          <p className="text-[12.5px] text-muted">
            Agents provide evidence. The reviewer makes the final call — and it is recorded separately from the policy
            verdict.
          </p>
          <div className="flex items-center gap-6">
            <a href="/app/" className="text-[12.5px] text-ink no-underline hover:text-accent">
              Launch tool
            </a>
            <span className="num text-[11px] text-muted">
              {health.state === "ready" ? `pipeline ${health.value.pipeline_version}` : "—"}
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}

type HealthState = { state: "loading" } | { state: "ready"; value: Health } | { state: "unreachable" };

/** The header prints the real versions from GET /health, or says plainly that it could not. */
function useBackendHealth(): HealthState {
  const [state, setState] = useState<HealthState>({ state: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    getHealth(controller.signal)
      .then((value) => setState({ state: "ready", value }))
      .catch(() => setState({ state: "unreachable" }));
    return () => controller.abort();
  }, []);

  return state;
}
