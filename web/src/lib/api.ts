/**
 * The only place the React app talks to the backend.
 *
 * Every shape here is transcribed from backend/schemas.py -- field for field, nothing invented.
 * Routes used (all pre-existing except the last, which is additive and read-only):
 *   GET  /health                    -> Health
 *   POST /live-jobs                 -> 202 LiveJobStatus
 *   GET  /live-jobs/{job_id}        -> LiveJobStatus
 *   GET  /jobs/{job_id}             -> JobReport
 *   POST /jobs/{job_id}/review      -> JobReport
 *   GET  /jobs/{job_id}/signal-preview -> SignalPreview   (Phase B, see README)
 */

export type AgentName = "validation" | "signal" | "prediction" | "decision" | "report";
export type AgentStatus = "working" | "completed" | "blocked" | "failed";
export type LiveStatus = "processing" | "completed" | "invalid_input" | "processing_failed";
export type Decision =
  | "ACCEPT"
  | "ACCEPT_WITH_WARNING"
  | "RECHECK_QUALITY"
  | "RETURN_UNCERTAIN"
  | "REQUEST_HUMAN_REVIEW"
  | "REJECT_INVALID_INPUT";
export type ReviewAction = "approve" | "mark_uncertain" | "override";

/** backend/schemas.py :: AgentUpdate */
export interface AgentUpdate {
  agent: AgentName;
  status: AgentStatus;
  message: string;
  findings: Record<string, unknown>;
  timestamp: string;
  duration_ms: number;
}

/** backend/schemas.py :: GateResult */
export interface GateResult {
  name: string;
  passed: boolean;
  blocking: boolean;
  observed: string;
  requirement: string;
}

/** backend/schemas.py :: EvidenceBundle */
export interface EvidenceBundle {
  schema_version: string;
  provenance: Record<string, unknown>;
  measurements: Record<string, unknown>;
  signals: string[];
  gates: GateResult[];
  recheck_budget: { limit?: number; used?: number; remaining?: number };
  allowed_actions: string[];
}

/** backend/schemas.py :: CalibrationInfo */
export interface CalibrationInfo {
  method: string;
  temperature: number;
  evaluated_on: string;
  confidence_discriminative: boolean;
  band: "low" | "medium" | "high";
  band_observed_accuracy: number | null;
  band_sample_size: number | null;
}

/** backend/schemas.py :: ClassificationReport */
export interface ClassificationReport {
  status: "classified" | "unavailable";
  model: string;
  reason_code: "no_model" | "no_annotations" | "montage_mismatch" | "trial_window" | "unknown" | null;
  reason: string | null;
  predicted_label: "left_hand" | "right_hand" | null;
  probabilities: Record<string, number>;
  vote_counts: Record<string, number>;
  agreement: number | null;
  trial_count: number;
  calibration: CalibrationInfo | null;
  warnings: string[];
}

/** backend/schemas.py :: QualityReport */
export interface QualityReport {
  status: "acceptable" | "warning" | "poor";
  peak_to_peak_uv: number | null;
  flat_channel_count: number;
  rejected_epoch_ratio: number;
  warnings: string[];
}

/** backend/schemas.py :: ValidationReport */
export interface ValidationReport {
  status: "valid" | "invalid";
  file_format: string;
  sampling_rate: number | null;
  channel_count: number;
  eeg_channel_count: number;
  duration_seconds: number | null;
  has_events: boolean;
  event_count: number;
  warnings: string[];
}

/** backend/schemas.py :: HumanReview */
export interface HumanReview {
  action: ReviewAction;
  reviewer_comment: string;
  approved_label: "left_hand" | "right_hand" | "uncertain";
  created_at: string;
}

/** backend/schemas.py :: JobReport */
export interface JobReport {
  job_id: string;
  status: LiveStatus;
  input_filename: string;
  input_hash: string;
  validation: ValidationReport;
  decision: Decision | null;
  preprocessing: Record<string, unknown>;
  epoching: Record<string, unknown>;
  quality: QualityReport | null;
  classification: ClassificationReport | null;
  evidence: EvidenceBundle | null;
  human_review: HumanReview | null;
  warnings: string[];
  audit_events: { timestamp: string; stage: string; detail: string }[];
  agent_updates: AgentUpdate[];
}

/** backend/schemas.py :: LiveJobStatus */
export interface LiveJobStatus {
  job_id: string;
  status: LiveStatus;
  agent_updates: AgentUpdate[];
  report: JobReport | null;
}

export interface Health {
  status: string;
  phase: number;
  pipeline_version: string;
  policy_version: string;
}

export interface SignalPreviewChannel {
  name: string;
  values: number[];
}

/** GET /jobs/{job_id}/signal-preview -- read-only view of the cleaned signal. */
export interface SignalPreview {
  job_id: string;
  channels: SignalPreviewChannel[];
  sample_rate_effective: number;
  duration_seconds: number;
  unit: string;
  /** Names that were asked for (C3, Cz, C4) ... */
  requested: string[];
  /** ... and whether the montage actually exposed them; false means the fallback channels are frontal. */
  matched: boolean;
  source: string;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function readError(response: Response): Promise<never> {
  let detail = `Request failed with ${response.status}.`;
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") detail = body.detail;
    else if (body.detail) detail = JSON.stringify(body.detail);
  } catch {
    /* the body was not JSON; the status line is all we have */
  }
  throw new ApiError(detail, response.status);
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, { signal, headers: { Accept: "application/json" } });
  if (!response.ok) await readError(response);
  return (await response.json()) as T;
}

export function getHealth(signal?: AbortSignal): Promise<Health> {
  return getJson<Health>("/health", signal);
}

/** POST /live-jobs -- multipart upload; the backend owns the job id. */
export async function createLiveJob(file: File, signal?: AbortSignal): Promise<LiveJobStatus> {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch("/live-jobs", { method: "POST", body, signal });
  if (!response.ok) await readError(response);
  return (await response.json()) as LiveJobStatus;
}

export function getLiveJob(jobId: string, signal?: AbortSignal): Promise<LiveJobStatus> {
  return getJson<LiveJobStatus>(`/live-jobs/${encodeURIComponent(jobId)}`, signal);
}

export function getJob(jobId: string, signal?: AbortSignal): Promise<JobReport> {
  return getJson<JobReport>(`/jobs/${encodeURIComponent(jobId)}`, signal);
}

/** GET /jobs/{job_id}/signal-preview; 404 means "the signal stage has not persisted a file". */
export async function getSignalPreview(jobId: string, signal?: AbortSignal): Promise<SignalPreview> {
  return getJson<SignalPreview>(`/jobs/${encodeURIComponent(jobId)}/signal-preview`, signal);
}

/** POST /jobs/{job_id}/review -- the reviewer's decision, recorded next to the policy verdict. */
export async function recordReview(
  jobId: string,
  payload: { action: ReviewAction; reviewer_comment: string; approved_label?: "left_hand" | "right_hand" },
): Promise<JobReport> {
  const response = await fetch(`/jobs/${encodeURIComponent(jobId)}/review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) await readError(response);
  return (await response.json()) as JobReport;
}

export const AGENT_ORDER: AgentName[] = ["validation", "signal", "prediction", "decision", "report"];

export const AGENT_LABEL: Record<AgentName, string> = {
  validation: "Validation",
  signal: "Signal",
  prediction: "Prediction",
  decision: "Decision",
  report: "Report",
};

/** The stage's own name from backend/agents.py, not a marketing name. */
export const AGENT_RESPONSIBILITY: Record<AgentName, string> = {
  validation: "Format, channels, event markers",
  signal: "8-30 Hz band-pass, epochs, quality gates",
  prediction: "FBCNet trial-level classification",
  decision: "Fixed policy thresholds and gates",
  report: "Evidence bundle for the reviewer",
};
