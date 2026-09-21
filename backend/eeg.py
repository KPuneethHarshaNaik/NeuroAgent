from __future__ import annotations

from pathlib import Path

import mne
import numpy as np

from backend.policy import DEFAULT_THRESHOLDS
from backend.schemas import QualityReport, ValidationReport

SUPPORTED_EXTENSIONS = {".fif", ".edf", ".bdf"}


def load_raw(path: Path) -> mne.io.BaseRaw:
    readers = {".fif": mne.io.read_raw_fif, ".edf": mne.io.read_raw_edf, ".bdf": mne.io.read_raw_bdf}
    return readers[path.suffix.lower()](path, preload=True, verbose=False)


def validate(raw: mne.io.BaseRaw, extension: str) -> ValidationReport:
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    events, _ = mne.events_from_annotations(raw, verbose=False)
    warnings: list[str] = []
    if not len(eeg_picks):
        warnings.append("No EEG channels were detected.")
    if not len(events):
        warnings.append("No event markers were found; trial-based classification cannot run.")
    return ValidationReport(
        status="valid" if len(eeg_picks) and len(events) else "invalid",
        file_format=extension.removeprefix("."),
        sampling_rate=float(raw.info["sfreq"]),
        channel_count=len(raw.ch_names),
        eeg_channel_count=len(eeg_picks),
        duration_seconds=round(raw.n_times / raw.info["sfreq"], 3),
        has_events=bool(len(events)),
        event_count=int(len(events)),
        warnings=warnings,
    )


def preprocess(raw: mne.io.BaseRaw) -> tuple[mne.io.BaseRaw, dict]:
    cleaned = raw.copy()
    eeg_picks = mne.pick_types(cleaned.info, eeg=True, exclude=[])
    cleaned.filter(8.0, 30.0, picks=eeg_picks, verbose=False)
    cleaned.set_eeg_reference("average", projection=False, verbose=False)
    return cleaned, {"bandpass_hz": [8.0, 30.0], "reference": "average", "ica_applied": False}


def epoch_and_measure(raw: mne.io.BaseRaw) -> tuple[dict, QualityReport]:
    events, event_ids = mne.events_from_annotations(raw, verbose=False)
    epochs = mne.Epochs(raw, events, event_id=event_ids, tmin=0.0, tmax=2.0, baseline=None, preload=True, verbose=False)
    data = epochs.get_data(picks="eeg")
    peak_to_peak_by_epoch_uv = np.ptp(data, axis=2).max(axis=1) * 1_000_000
    peak_to_peak_uv = float(peak_to_peak_by_epoch_uv.max())
    channel_std = data.std(axis=(0, 2))
    flat_channels = int(np.count_nonzero(channel_std < 1e-8))
    rejected_epoch_ratio = float(np.mean(peak_to_peak_by_epoch_uv > DEFAULT_THRESHOLDS.maximum_peak_to_peak_uv))
    warnings: list[str] = []
    if peak_to_peak_uv > DEFAULT_THRESHOLDS.maximum_peak_to_peak_uv:
        warnings.append("High peak-to-peak amplitude detected.")
    if flat_channels > DEFAULT_THRESHOLDS.maximum_flat_channels:
        warnings.append(f"{flat_channels} near-flat EEG channel(s) detected.")
    if rejected_epoch_ratio:
        warnings.append(f"{rejected_epoch_ratio:.0%} of event-locked epochs exceed the amplitude limit.")
    status = "poor" if flat_channels > DEFAULT_THRESHOLDS.maximum_flat_channels or rejected_epoch_ratio > DEFAULT_THRESHOLDS.maximum_rejected_epoch_ratio else "warning" if warnings else "acceptable"
    return (
        {"epoch_count": len(epochs), "window_seconds": [0.0, 2.0], "event_ids": event_ids},
        QualityReport(status=status, peak_to_peak_uv=round(peak_to_peak_uv, 3), flat_channel_count=flat_channels, rejected_epoch_ratio=round(rejected_epoch_ratio, 4), warnings=warnings),
    )
