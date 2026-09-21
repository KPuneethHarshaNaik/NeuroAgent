"""Left/right motor-imagery inference with a trained FBCNet checkpoint.

Inference is deliberately isolated from the Phase 1 pipeline: an uploaded recording is
preprocessed here exactly the way ``training.prepare_dataset`` prepared the training
data (160 Hz, 8-30 Hz band-pass, average reference, 0.5-3.5 s T1/T2 trials), and every
failure mode — missing or corrupt checkpoint, no motor-imagery annotations, a montage
the checkpoint was not trained on — degrades to ``status="unavailable"`` with a machine
readable ``reason_code`` instead of failing the job.

Reported probabilities are temperature-calibrated when ``<checkpoint>_calibration.json``
exists (see ``training/calibrate_fbcnet.py``). ``CalibrationInfo`` also carries what a
held-out split says about the confidence band the prediction landed in, plus whether
confidence is discriminative at all — the current checkpoint reports honest numbers but
``confidence_discriminative: false``, meaning confidence must not be read as reliability.

The checkpoint stores no channel names, so a recording is only valid for classification
if its EEG channels are already in the training order (the PhysioNet 64-channel montage).
Only the channel and sample counts are verified here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

from backend.schemas import CalibrationInfo, ClassificationReport, FailureCode

DEFAULT_MODEL_PATH = Path("models/fbcnet_v1.pt")
TRAINING_SAMPLING_RATE_HZ = 160.0
TRAINING_BANDPASS_HZ = (8.0, 30.0)
TRAINING_EPOCH_SECONDS = (0.5, 3.5)
TRAINING_EVENT_IDS = {"T1": 0, "T2": 1}  # Mirrors training.prepare_dataset.EVENT_IDS.
SUPPORTED_LABELS = ("left_hand", "right_hand")
LOW_AGREEMENT = 0.6
MINIMUM_TRIALS = 5


class TrialExtractionError(RuntimeError):
    """Raised when a recording cannot yield motor-imagery trials for the classifier."""


@dataclass(frozen=True)
class Calibration:
    method: str
    temperature: float
    evaluated_on: str
    confidence_discriminative: bool
    bands: tuple[dict, ...]

    def band_for(self, confidence: float) -> dict | None:
        for band in self.bands:
            if band["min_probability"] <= confidence < band["max_probability"]:
                return band
        return None


@dataclass(frozen=True)
class Classifier:
    model: object
    channels: int
    samples: int
    labels: dict[int, str]
    normalization_mean: np.ndarray
    normalization_std: np.ndarray
    calibration: Calibration | None = None


_cache: dict[Path, tuple[Classifier | None, str | None]] = {}


def configured_model_path() -> Path:
    return Path(os.getenv("NEUROAGENT_MODEL_PATH", str(DEFAULT_MODEL_PATH)))


def calibration_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(f"{checkpoint.stem}_calibration.json")


def _torch():
    import torch

    return torch


def _read_calibration(checkpoint: Path) -> Calibration | None:
    """Calibration is optional: without it probabilities stay uncalibrated and unqualified."""
    path = calibration_path(checkpoint)
    if not path.exists():
        return None
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        return Calibration(
            method=str(artifact["method"]),
            temperature=float(artifact["temperature"]),
            evaluated_on=str(artifact["evaluated_on"]),
            confidence_discriminative=bool(artifact["confidence_discriminative"]),
            bands=tuple(artifact["reliability_bands"]),
        )
    except Exception:
        return None


def _read_checkpoint(path: Path) -> tuple[Classifier | None, str | None]:
    if not path.exists():
        return None, f"No trained model found at {path}."
    try:
        torch = _torch()
        from training.fbcnet import FBCNet
    except ImportError as exc:
        return None, f"Classifying requires torch, which could not be imported: {exc}."
    try:
        # weights_only=False is required: the checkpoint also carries NumPy normalization arrays.
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        labels = {int(index): str(label) for index, label in bundle["labels"].items()}
        if sorted(labels.values()) != sorted(SUPPORTED_LABELS):
            return None, f"{path} does not predict {SUPPORTED_LABELS}."
        model = FBCNet(channels=int(bundle["channels"]), samples=int(bundle["samples"]))
        model.load_state_dict(bundle["state_dict"])
        model.eval()
        classifier = Classifier(
            model=model,
            channels=int(bundle["channels"]),
            samples=int(bundle["samples"]),
            labels=labels,
            normalization_mean=np.asarray(bundle["normalization_mean"]),
            normalization_std=np.asarray(bundle["normalization_std"]),
            calibration=_read_calibration(path),
        )
    except Exception as exc:  # A corrupt checkpoint must not fail the job.
        return None, f"{path} could not be loaded: {type(exc).__name__}."
    return classifier, None


def load_classifier(path: Path | None = None) -> tuple[Classifier | None, str | None]:
    """Load and memoize a checkpoint. Returns the classifier and, on failure, why."""
    target = (path or configured_model_path()).resolve()
    if target not in _cache:
        _cache[target] = _read_checkpoint(target)
    return _cache[target]


def _training_trials(raw: mne.io.BaseRaw) -> np.ndarray:
    """Reproduce training.prepare_dataset.process_recording preprocessing for one recording."""
    prepared = raw.copy().pick("eeg")
    if not np.isclose(prepared.info["sfreq"], TRAINING_SAMPLING_RATE_HZ):
        prepared.resample(TRAINING_SAMPLING_RATE_HZ, verbose=False)
    prepared.filter(*TRAINING_BANDPASS_HZ, verbose=False)
    prepared.set_eeg_reference("average", projection=False, verbose=False)
    missing = sorted(set(TRAINING_EVENT_IDS) - set(prepared.annotations.description))
    if missing:
        raise TrialExtractionError(f"Motor-imagery annotations {'/'.join(missing)} were not found.")
    events, _ = mne.events_from_annotations(prepared, event_id=TRAINING_EVENT_IDS, verbose=False)
    if not len(events):
        raise TrialExtractionError("No T1/T2 motor-imagery annotations were found.")
    trials = mne.Epochs(prepared, events, event_id=TRAINING_EVENT_IDS, tmin=TRAINING_EPOCH_SECONDS[0], tmax=TRAINING_EPOCH_SECONDS[1], baseline=None, preload=True, verbose=False)
    data = trials.get_data(copy=False).astype(np.float32)
    if not len(data):
        raise TrialExtractionError(f"No complete {TRAINING_EPOCH_SECONDS[0]}-{TRAINING_EPOCH_SECONDS[1]} s trial could be extracted from the recording.")
    return data


def _unavailable(model_name: str, reason: str, reason_code: FailureCode) -> ClassificationReport:
    return ClassificationReport(status="unavailable", model=model_name, reason_code=reason_code, reason=reason)


def classify(raw: mne.io.BaseRaw, path: Path | None = None) -> ClassificationReport:
    """Classify left vs. right imagined hand movement. Never raises: failures are reported."""
    model_path = path or configured_model_path()
    model_name = model_path.stem
    classifier, problem = load_classifier(model_path)
    if classifier is None:
        return _unavailable(model_name, problem or "The model is unavailable.", "no_model")
    if not len(mne.pick_types(raw.info, eeg=True, exclude=[])):
        return _unavailable(model_name, "The recording has no EEG channels to classify.", "unknown")
    try:
        data = _training_trials(raw)
    except TrialExtractionError as exc:
        code: FailureCode = "no_annotations" if "annotations" in str(exc) else "trial_window"
        return _unavailable(model_name, str(exc), code)
    except Exception as exc:
        return _unavailable(model_name, f"Motor-imagery trials could not be prepared: {type(exc).__name__}.", "unknown")
    if data.shape[1] != classifier.channels:
        return _unavailable(model_name, f"{model_name} was trained on {classifier.channels} EEG channels but the recording has {data.shape[1]}.", "montage_mismatch")
    if data.shape[2] != classifier.samples:
        return _unavailable(model_name, f"{model_name} expects {classifier.samples} samples per trial but preprocessing produced {data.shape[2]}.", "trial_window")

    calibration = classifier.calibration
    torch = _torch()
    normalized = (data - classifier.normalization_mean) / classifier.normalization_std.clip(1e-6)
    with torch.no_grad():
        logits = classifier.model(torch.from_numpy(normalized).unsqueeze(1))
        # Temperature divides the logits, so it moves the probabilities but never the ranking.
        per_trial = torch.softmax(logits / (calibration.temperature if calibration else 1.0), dim=1).numpy()
    mean_probabilities = per_trial.mean(axis=0)
    predicted_label = classifier.labels[int(mean_probabilities.argmax())]
    winners = per_trial.argmax(axis=1)
    vote_counts = {label: int(np.count_nonzero(winners == index)) for index, label in classifier.labels.items()}
    agreement = round(vote_counts[predicted_label] / len(per_trial), 4)
    confidence = float(mean_probabilities.max())
    warnings: list[str] = []
    if agreement < LOW_AGREEMENT:
        warnings.append(f"Trial-level predictions disagree: only {agreement:.0%} of trials favour {predicted_label}.")
    if len(per_trial) < MINIMUM_TRIALS:
        warnings.append(f"Only {len(per_trial)} motor-imagery trial(s) were available; the prediction is unreliable.")
    info = None
    if calibration is not None:
        band = calibration.band_for(confidence) or {}
        info = CalibrationInfo(
            method=calibration.method,
            temperature=calibration.temperature,
            evaluated_on=calibration.evaluated_on,
            confidence_discriminative=calibration.confidence_discriminative,
            band=band.get("band", "low"),
            band_observed_accuracy=band.get("observed_accuracy"),
            band_sample_size=band.get("n"),
        )
        if not calibration.confidence_discriminative:
            warnings.append("Confidence is not discriminative for this model, so it cannot support acceptance.")
    return ClassificationReport(
        status="classified",
        model=model_name,
        predicted_label=predicted_label,
        probabilities={label: round(float(mean_probabilities[index]), 4) for index, label in classifier.labels.items()},
        vote_counts=vote_counts,
        agreement=agreement,
        trial_count=len(per_trial),
        calibration=info,
        warnings=warnings,
    )
