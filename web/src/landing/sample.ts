import type { GateResult } from "@/lib/api";

/**
 * A real review brief, transcribed verbatim from the evidence bundle this pipeline stored for
 * job 0aa438b5-6c2a-4fde-950a-ee4e13115f1f (S001R03.edf, sha256 3427c8d0...bfe2e).
 * Not illustrative: these are the numbers the tool actually produced.
 */
export const SAMPLE_JOB = {
  jobId: "0aa438b5-6c2a-4fde-950a-ee4e13115f1f",
  filename: "S001R03.edf",
  inputHashShort: "3427c8d0…bfe2e",
  startedAt: "2026-09-29T12:12:24+00:00",
} as const;

export const SAMPLE_VERDICT = {
  decision: "ACCEPT_WITH_WARNING" as const,
  predictedLabel: "left_hand" as const,
  probabilities: { left_hand: 0.5414, right_hand: 0.4586 },
  trialCount: 15,
  agreement: 0.5333,
  model: "fbcnet_v1",
  warning: "Trial-level predictions disagree: only 53% of trials favour left_hand.",
} as const;

export const SAMPLE_QUALITY = {
  status: "acceptable" as const,
  peakToPeakUv: 246.859,
  flatChannelCount: 0,
  rejectedEpochRatio: 0.0,
} as const;

export const SAMPLE_SIGNAL = {
  samplingRateHz: 160,
  channelCount: 64,
  eegChannelCount: 64,
  durationSeconds: 125,
  eventCount: 30,
  epochCount: 30,
  bandpassHz: [8, 30] as const,
  reference: "average" as const,
  epochSeconds: [0, 2] as const,
} as const;

export const SAMPLE_CALIBRATION = {
  method: "temperature_scaling" as const,
  temperature: 1.6995,
  evaluatedOn: "test" as const,
  band: "low" as const,
  bandObservedAccuracy: 0.5972,
  bandSampleSize: 144,
  trainedAccuracy: 0.6102,
  expectedCalibrationError: 0.0785,
} as const;

/** The deterministic gates the policy actually ran on that job, in order, verbatim. */
export const SAMPLE_GATES: GateResult[] = [
  {
    name: "input_valid",
    passed: true,
    blocking: true,
    observed: "valid",
    requirement: "valid EEG channels and event markers",
  },
  { name: "signal_quality", passed: true, blocking: true, observed: "acceptable", requirement: "quality better than poor" },
  {
    name: "classification_available",
    passed: true,
    blocking: true,
    observed: "classified",
    requirement: "a motor-imagery prediction",
  },
  { name: "enough_trials", passed: true, blocking: true, observed: "15 trials", requirement: "at least 5 trials" },
  {
    name: "trial_agreement",
    passed: false,
    blocking: false,
    observed: "53% agreement",
    requirement: "at least 60% of trials agree",
  },
  {
    name: "confidence_discriminative",
    passed: true,
    blocking: false,
    observed: "True",
    requirement: "held-out confidence must separate correct from incorrect trials",
  },
];

/** backend/policy.py :: DEFAULT_THRESHOLDS -- frozen constants, recorded in every bundle. */
export const POLICY_THRESHOLDS = [
  { key: "minimum_trials", value: "5", note: "an EEG recording with fewer usable trials is not a result" },
  { key: "minimum_agreement", value: "0.60", note: "of trials must favour the reported class" },
  { key: "maximum_peak_to_peak_uv", value: "300.0", note: "per-epoch amplitude ceiling" },
  { key: "maximum_flat_channels", value: "0", note: "a dead electrode fails the signal gate" },
  { key: "maximum_rejected_epoch_ratio", value: "0.25", note: "above this, quality is not acceptable" },
  { key: "recheck_limit", value: "1", note: "one diagnostic re-run, never a search for a friendlier answer" },
] as const;

export const PIPELINE_VERSION = "0.2.0";
export const POLICY_VERSION = "2.1";
