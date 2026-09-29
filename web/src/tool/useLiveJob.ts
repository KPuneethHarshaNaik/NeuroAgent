import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  createLiveJob,
  getJob,
  getLiveJob,
  type AgentUpdate,
  type JobReport,
  type LiveJobStatus,
} from "@/lib/api";

/** The vanilla panel polled at exactly this interval; it is the product's real cadence. */
export const POLL_INTERVAL_MS = 450;

export type JobPhase = "idle" | "uploading" | "polling" | "finished" | "failed";

export type LiveJobState = {
  jobId: string | null;
  phase: JobPhase;
  /** The latest polled live status, or null before the first response. */
  live: LiveJobStatus | null;
  /** The finished report -- from the live status, or fetched when the live entry is gone. */
  report: JobReport | null;
  /** Every update seen so far, in arrival order (append-only, exactly as the backend serves it). */
  updates: AgentUpdate[];
  error: string | null;
  /** True when a job was resumed from a URL and its live entry no longer exists. */
  archived: boolean;
};

const INITIAL: LiveJobState = { jobId: null, phase: "idle", live: null, report: null, updates: [], error: null, archived: false };

const updateKey = (update: AgentUpdate) => `${update.agent}|${update.status}|${update.timestamp}`;

/**
 * Appends updates that have not been seen, keeping arrival order. This exists because of a real
 * asymmetry in the built-in pipeline: its live stream publishes a `working` update for the signal,
 * prediction, decision and report stages but never their `completed` ones, while the stored report
 * carries all five completed updates with their durations. Polling alone therefore leaves four
 * boxes spinning forever; merging the report's updates through the same queue lets every stage
 * settle on the truth the brief is about to show.
 */
export function mergeUpdates(seen: AgentUpdate[], incoming: AgentUpdate[]): AgentUpdate[] {
  if (!incoming.length) return seen;
  const known = new Set(seen.map(updateKey));
  const additions = incoming.filter((update) => !known.has(updateKey(update)));
  return additions.length ? [...seen, ...additions] : seen;
}

/**
 * Port of `pollLiveJob` / `renderLive` / `finish` from frontend/app.js.
 *
 * Behaviour carried over deliberately:
 *  - the POST response is rendered immediately, before any poll;
 *  - an immediate poll is taken, then the interval starts at 450 ms;
 *  - polling stops the moment status leaves "processing" (that is what `finish` did);
 *  - a polling error surfaces the backend's own `detail` and stops the loop.
 *
 * One addition, for resuming: if GET /live-jobs/{id} is gone but GET /jobs/{id} answers, the job is
 * finished and archived, so it is rendered from the stored report instead of polling forever.
 */
export function useLiveJob() {
  const [state, setState] = useState<LiveJobState>(INITIAL);
  const timer = useRef<number | null>(null);
  const jobIdRef = useRef<string | null>(null);

  const stopPolling = useCallback(() => {
    if (timer.current !== null) {
      window.clearInterval(timer.current);
      timer.current = null;
    }
  }, []);

  const applyStatus = useCallback((live: LiveJobStatus) => {
    setState((previous) => {
      // `renderLive` iterated the whole list every poll; the list is append-only, so only the tail
      // past the cursor ever reaches the queue (that diffing lives in useHandoff).
      const polled = live.agent_updates.length >= previous.updates.length ? live.agent_updates : previous.updates;
      const finished = live.status !== "processing" && live.report !== null;
      let updates: AgentUpdate[];
      if (finished) {
        // The built-in pipeline publishes a `working` update for signal/prediction/decision/report
        // but never their `completed` ones to the live store; the stored report carries all five
        // completed updates with real durations. When the live entry settles, the report's per-agent
        // completed update is the authoritative one and should win over any stale `working` entry that
        // arrived earlier — otherwise four boxes keep spinning even though the job is done.
        const reportUpdates = live.report!.agent_updates;
        updates = reportUpdates.length
          ? mergeUpdates(previous.updates, reportUpdates)
          : polled;
      } else {
        updates = polled;
      }
      return {
        ...previous,
        live,
        updates,
        report: finished ? live.report : previous.report,
        phase: live.status !== "processing" ? "finished" : "polling",
        error: null,
      };
    });
    if (live.status !== "processing") stopPolling();
  }, [stopPolling]);

  const poll = useCallback(async (jobId: string) => {
    try {
      const live = await getLiveJob(jobId);
      applyStatus(live);
      return live;
    } catch (error) {
      // A resumed job may have outlived its in-memory live entry; the stored report is still real.
      if (error instanceof ApiError && error.status === 404) {
        try {
          const report = await getJob(jobId);
          stopPolling();
          setState((previous) => ({
            ...previous,
            phase: "finished",
            report,
            updates: mergeUpdates(previous.updates, report.agent_updates),
            archived: true,
            error: null,
          }));
          return null;
        } catch {
          /* fall through to the original error */
        }
      }
      stopPolling();
      setState((previous) => ({ ...previous, phase: "failed", error: error instanceof Error ? error.message : "Could not read live review status." }));
      return null;
    }
  }, [applyStatus, stopPolling]);

  /** The upload path: POST then start the interval, exactly like the vanilla submit handler. */
  const start = useCallback(async (file: File) => {
    stopPolling();
    jobIdRef.current = null;
    setState({ ...INITIAL, phase: "uploading" });
    try {
      const created = await createLiveJob(file);
      jobIdRef.current = created.job_id;
      setState({
        ...INITIAL,
        jobId: created.job_id,
        phase: created.status === "processing" ? "polling" : "finished",
        live: created,
        updates: created.agent_updates,
        report: created.report,
      });
      await poll(created.job_id);
      if (timer.current === null) {
        timer.current = window.setInterval(() => {
          void poll(created.job_id);
        }, POLL_INTERVAL_MS);
      }
      return created.job_id;
    } catch (error) {
      setState((previous) => ({ ...previous, phase: "failed", error: error instanceof Error ? error.message : "The recording could not be processed." }));
      return null;
    }
  }, [poll, stopPolling]);

  /** Resume: /app/?job=<id> attaches to a job that is already running or already finished. */
  const resume = useCallback(async (jobId: string) => {
    stopPolling();
    jobIdRef.current = jobId;
    setState({ ...INITIAL, jobId, phase: "polling" });
    const live = await poll(jobId);
    if (live && live.status === "processing" && timer.current === null) {
      timer.current = window.setInterval(() => {
        void poll(jobId);
      }, POLL_INTERVAL_MS);
    }
  }, [poll, stopPolling]);

  const reset = useCallback(() => {
    stopPolling();
    jobIdRef.current = null;
    setState(INITIAL);
  }, [stopPolling]);

  useEffect(() => stopPolling, [stopPolling]);

  return { ...state, start, resume, reset };
}
