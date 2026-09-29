import asyncio
import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import mne
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.agents import decision_agent, failure_update, prediction_agent, report_agent, signal_agent, update, validation_agent
from backend.eeg import SUPPORTED_EXTENSIONS, load_raw
from backend.evidence import PIPELINE_VERSION, build_evidence
from backend.feedback import save_reviewer_feedback
from backend.graph import RunContext, decision_node, prediction_node, report_node, signal_node, stream_pipeline, validation_node
from backend.policy import POLICY_VERSION, PolicyOutcome, PolicyThresholds, RecheckState
from backend.schemas import AgentUpdate, ClassificationReport, GateResult, HumanReview, HumanReviewRequest, JobReport, LiveJobStatus, QualityReport, ValidationReport

app = FastAPI(title="NeuroAgent", version=PIPELINE_VERSION)
RUNTIME_DIR = Path(os.getenv("NEUROAGENT_RUNTIME_DIR", "runtime"))
MAX_UPLOAD_BYTES = int(os.getenv("NEUROAGENT_MAX_UPLOAD_MB", "250")) * 1024 * 1024
jobs: dict[str, JobReport] = {}
live_jobs: dict[str, LiveJobStatus] = {}

logger = logging.getLogger("neuroagent")

# When set, POST /jobs hands each upload to this orchestrator webhook instead of running the
# built-in pipeline here. Empty (the default) keeps the pipeline entirely in-process.
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "").strip()
N8N_HANDOFF_TIMEOUT_SECONDS = 10.0


def event(stage: str, detail: str) -> dict:
    return {"timestamp": datetime.now(UTC).isoformat(), "stage": stage, "detail": detail}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "phase": 3, "pipeline_version": PIPELINE_VERSION, "policy_version": POLICY_VERSION}


async def save_upload(file: UploadFile) -> tuple[str, str, Path, str]:
    filename = Path(file.filename or "upload").name
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, "Only .fif, .edf, and .bdf EEG files are supported.")
    payload = await file.read()
    if not payload:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_BYTES // 1024 // 1024} MB limit.")

    job_id = str(uuid4())
    job_dir = RUNTIME_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    # ``_eeg`` keeps MNE's FIF reader from warning about a nonconforming raw filename.
    input_path = job_dir / f"original_eeg{extension}"
    input_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    # Record what this job id refers to, so a later /internal/* step -- which receives nothing
    # but the job id -- can rebuild the job without the recording being uploaded again.
    internal_write_json(
        job_dir / JOB_METADATA_FILE,
        {"job_id": job_id, "filename": filename, "digest": digest, "extension": extension, "input_path": str(input_path)},
    )
    return job_id, filename, input_path, digest


def legacy_process_job(job_id: str, filename: str, input_path: Path, digest: str, agent_updates: list[AgentUpdate] | None = None) -> JobReport:
    updates = agent_updates if agent_updates is not None else []
    audit = [event("uploaded", "Original file stored unchanged."), event("validating", "Reading EEG metadata and events.")]
    extension = input_path.suffix.lower()

    def working(agent: str, message: str) -> None:
        updates.append(AgentUpdate(agent=agent, status="working", message=message, timestamp=datetime.now(UTC).isoformat()))

    active_agent = "validation"
    try:
        working("validation", "Reading EEG metadata and event markers.")
        raw = load_raw(input_path)
        validation, validation_update = validation_agent(raw, extension)
        updates.append(validation_update)
        if validation.status == "invalid":
            active_agent = "decision"
            working("decision", "Checking whether the recording can proceed.")
            outcome, decision_update = decision_agent(validation, None, None)
            updates.append(decision_update)
            working("report", "Preparing evidence for the reviewer.")
            updates.append(report_agent(outcome.decision))
            report = JobReport(job_id=job_id, status="invalid_input", input_filename=filename, input_hash=digest, validation=validation, decision=outcome.decision, evidence=build_evidence(job_id, filename, digest, validation, {}, {}, None, None, outcome), warnings=validation.warnings, audit_events=audit + [event("rejected", "Input is missing EEG channels or events."), event("policy", f"{outcome.decision} (policy {POLICY_VERSION})")], agent_updates=updates)
        else:
            active_agent = "signal"
            working("signal", "Cleaning the signal and measuring epochs.")
            _cleaned, preprocessing, epochs, quality, signal_update = signal_agent(raw)
            updates.append(signal_update)
            active_agent = "prediction"
            working("prediction", "Reading motor-imagery trials with FBCNet.")
            classification, prediction_update = prediction_agent(raw)
            updates.append(prediction_update)
            if classification.status == "classified":
                classification_event = event("classification", f"{classification.model} predicted {classification.predicted_label} from {classification.trial_count} motor-imagery trial(s).")
            else:
                classification_event = event("classification_skipped", classification.reason or "Classification is unavailable.")
            active_agent = "decision"
            working("decision", "Comparing model evidence with safety rules.")
            outcome, decision_update = decision_agent(validation, quality, classification)
            updates.append(decision_update)
            evidence = build_evidence(job_id, filename, digest, validation, preprocessing, epochs, quality, classification, outcome)
            working("report", "Preparing evidence for the reviewer.")
            updates.append(report_agent(outcome.decision))
            report = JobReport(job_id=job_id, status="completed", input_filename=filename, input_hash=digest, validation=validation, decision=outcome.decision, preprocessing=preprocessing, epoching=epochs, quality=quality, classification=classification, evidence=evidence, warnings=quality.warnings + classification.warnings, audit_events=audit + [event("preprocessing", "Applied 8-30 Hz band-pass and average reference."), event("epoching", f"Created {epochs['epoch_count']} event-locked epochs."), classification_event, event("policy", f"{outcome.decision} (policy {POLICY_VERSION})")], agent_updates=updates)
    except Exception as exc:
        updates.append(failure_update(active_agent, exc))
        report = JobReport(job_id=job_id, status="processing_failed", input_filename=filename, input_hash=digest, validation=ValidationReport(status="invalid", file_format=extension.removeprefix("."), warnings=["EEG file could not be read or processed."]), warnings=[str(exc)], audit_events=audit + [event("failed", type(exc).__name__)], agent_updates=updates)
    return report


def process_job(job_id: str, filename: str, input_path: Path, digest: str, agent_updates: list[AgentUpdate] | None = None) -> JobReport:
    use_langgraph = os.getenv("USE_LANGGRAPH", "1").lower() in ("1", "true", "yes")
    if not use_langgraph:
        return legacy_process_job(job_id, filename, input_path, digest, agent_updates)

    updates = agent_updates if agent_updates is not None else []
    
    # Audit trail writing to both RUNTIME_DIR/job_id/events.jsonl and jobs/job_id/events.jsonl
    events_dir1 = RUNTIME_DIR / job_id
    events_dir1.mkdir(parents=True, exist_ok=True)
    events_file1 = events_dir1 / "events.jsonl"

    events_dir2 = Path("jobs") / job_id
    events_dir2.mkdir(parents=True, exist_ok=True)
    events_file2 = events_dir2 / "events.jsonl"

    def record_event(evt_data: dict) -> None:
        line = json.dumps(evt_data) + "\n"
        with events_file1.open("a", encoding="utf-8") as f1:
            f1.write(line)
        with events_file2.open("a", encoding="utf-8") as f2:
            f2.write(line)

    record_event({"type": "start", "timestamp": datetime.now(UTC).isoformat(), "job_id": job_id})

    final_report: JobReport | None = None
    for item in stream_pipeline(job_id, filename, input_path, digest):
        if isinstance(item, AgentUpdate):
            updates.append(item)
            record_event({"type": "update", "data": item.model_dump()})
        elif isinstance(item, JobReport):
            final_report = item
            record_event({"type": "report", "data": item.model_dump()})

    if final_report is None:
        raise RuntimeError("LangGraph pipeline failed to produce a JobReport.")
    return final_report



def delegated_report(job_id: str, filename: str, input_path: Path, digest: str) -> JobReport:
    """The body POST /jobs returns when an external orchestrator owns the job.

    The built-in pipeline deliberately did not run, so there is no verdict to report yet. It is
    recorded as ``processing_failed`` -- the only one of the three statuses that does not claim
    an analysis took place -- and the warning says plainly that the real report is still to come.
    """
    return JobReport(
        job_id=job_id,
        status="processing_failed",
        input_filename=filename,
        input_hash=digest,
        validation=ValidationReport(
            status="invalid",
            file_format=input_path.suffix.lower().removeprefix("."),
            warnings=["Not validated here: this job was handed to an external orchestrator."],
        ),
        warnings=[f"Handed to the orchestrator at {N8N_WEBHOOK_URL}; the finished report will replace this one in the job store."],
    )


@app.post("/jobs", response_model=JobReport, status_code=201)
async def create_job(file: UploadFile = File(...)) -> JobReport:
    job_id, filename, input_path, digest = await save_upload(file)
    if N8N_WEBHOOK_URL:

        async def hand_off() -> None:
            """Post the job to the orchestrator, falling back to the built-in pipeline."""
            try:
                async with httpx.AsyncClient(timeout=N8N_HANDOFF_TIMEOUT_SECONDS) as client:
                    response = await client.post(N8N_WEBHOOK_URL, json={"job_id": job_id})
                    response.raise_for_status()
            except Exception as exc:
                logger.warning(
                    "Hand-off of job %s to %s failed (%s: %s); running the built-in pipeline instead.",
                    job_id,
                    N8N_WEBHOOK_URL,
                    type(exc).__name__,
                    exc,
                )
                jobs[job_id] = await asyncio.to_thread(process_job, job_id, filename, input_path, digest)

        asyncio.create_task(hand_off())
        report = delegated_report(job_id, filename, input_path, digest)
    else:
        report = process_job(job_id, filename, input_path, digest)
    jobs[job_id] = report
    return report


@app.post("/live-jobs", response_model=LiveJobStatus, status_code=202)
async def create_live_job(file: UploadFile = File(...)) -> LiveJobStatus:
    """Start a job without making the reviewer wait for each agent to finish."""
    job_id, filename, input_path, digest = await save_upload(file)
    updates: list[AgentUpdate] = []
    state = LiveJobStatus(job_id=job_id, status="processing", agent_updates=updates)
    live_jobs[job_id] = state

    async def finish() -> None:
        report = await asyncio.to_thread(process_job, job_id, filename, input_path, digest, state.agent_updates)
        jobs[job_id] = report
        state.status = report.status
        state.report = report

    if N8N_WEBHOOK_URL:

        async def hand_off() -> None:
            """Post the job to the orchestrator, falling back to the built-in pipeline."""
            try:
                async with httpx.AsyncClient(timeout=N8N_HANDOFF_TIMEOUT_SECONDS) as client:
                    response = await client.post(N8N_WEBHOOK_URL, json={"job_id": job_id})
                    response.raise_for_status()
            except Exception as exc:
                logger.warning(
                    "Hand-off of job %s to %s failed (%s: %s); running the built-in pipeline instead.",
                    job_id,
                    N8N_WEBHOOK_URL,
                    type(exc).__name__,
                    exc,
                )
                await finish()

        # The orchestrator owns the run, so the live view stays "processing" until its
        # report_node result arrives -- that status is what keeps the UI polling.
        state.agent_updates.append(
            update(
                "validation",
                "working",
                f"Handed to the orchestrator at {N8N_WEBHOOK_URL}; progress appears here as it reports back.",
            )
        )
        jobs[job_id] = delegated_report(job_id, filename, input_path, digest)
        asyncio.create_task(hand_off())
    else:
        asyncio.create_task(finish())
    return state


@app.get("/live-jobs/{job_id}", response_model=LiveJobStatus)
def get_live_job(job_id: str) -> LiveJobStatus:
    if job_id not in live_jobs:
        raise HTTPException(404, "Live job not found.")
    return live_jobs[job_id]


@app.get("/jobs/{job_id}", response_model=JobReport)
def get_job(job_id: str) -> JobReport:
    if job_id not in jobs:
        raise HTTPException(404, "Job not found.")
    return jobs[job_id]


@app.post("/jobs/{job_id}/review", response_model=JobReport)
def review_job(job_id: str, request: HumanReviewRequest) -> JobReport:
    """Record a human outcome without rewriting the automated policy decision."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found.")
    report = jobs[job_id]
    if request.action == "approve":
        label = report.classification.predicted_label if report.classification else None
        if label is None:
            raise HTTPException(409, "A prediction is required before it can be approved.")
    elif request.action == "mark_uncertain":
        label = "uncertain"
    else:
        if request.approved_label not in {"left_hand", "right_hand"}:
            raise HTTPException(422, "An override must specify left_hand or right_hand.")
        label = request.approved_label
    report.human_review = HumanReview(
        action=request.action,
        reviewer_comment=request.reviewer_comment.strip(),
        approved_label=label,
        created_at=datetime.now(UTC).isoformat(),
    )
    report.audit_events.append(event("human_review", f"{request.action}: {label}"))
    save_reviewer_feedback(RUNTIME_DIR / "reviewer_feedback.jsonl", report)
    return report


SIGNAL_PREVIEW_CHANNELS = ("C3", "Cz", "C4")
SIGNAL_PREVIEW_POINTS = 700


def normalised_channel_name(name: str) -> str:
    """EDF headers pad channel names to four characters ("C3.."), so compare letters and digits only.

    A literal ``"C3" in raw.ch_names`` check fails on every PhysioNet eegmmidb recording and would
    silently return the first three channels instead -- which are frontal, not motor cortex. That
    fallback is only acceptable when it is reported, never when it is silent.
    """
    return "".join(character for character in name.lower() if character.isalnum())


@app.get("/jobs/{job_id}/signal-preview")
def signal_preview(job_id: str) -> dict:
    """A read-only view of the cleaned signal one job persisted. Additive: it changes no other route.

    The file it reads is the one the signal step already wrote (``internal/signal_cleaned_eeg.fif``
    -- the cleaned *continuous* recording, band-passed 8-30 Hz and average-referenced, not the
    epoched trials). No MNE processing runs here: the recording is loaded, three channels are
    decimated and returned. C3, Cz and C4 are the motor-cortex sites the classification evidence
    rests on; if a montage does not expose them, the first three EEG channels come back with
    ``matched=false`` so the caller can label them honestly.
    """
    path = internal_dir(job_id) / INTERNAL_CLEANED_FILE
    if not path.is_file():
        raise HTTPException(
            404,
            "No cleaned signal is persisted for this job. The signal stage writes one when it runs; "
            "a job processed by the built-in pipeline (rather than by the /internal/* steps) keeps "
            "its cleaned recording in memory only.",
        )
    try:
        raw = load_raw(path)
    except Exception as exc:
        raise HTTPException(422, f"The persisted cleaned signal could not be read ({type(exc).__name__}: {exc}).") from exc

    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if not len(picks):
        raise HTTPException(422, "The persisted cleaned signal has no EEG channels.")

    by_name = {normalised_channel_name(raw.ch_names[position]): position for position in picks}
    chosen = [by_name.get(normalised_channel_name(name)) for name in SIGNAL_PREVIEW_CHANNELS]
    matched = all(position is not None for position in chosen)
    if not matched:
        chosen = list(picks[:3])

    sampling_rate = float(raw.info["sfreq"])
    total_samples = raw.n_times
    step = max(1, total_samples // SIGNAL_PREVIEW_POINTS)
    # Volts -> microvolts: the quality report and the amplitude ceiling the policy uses are in uV,
    # so the trace is reported in the same unit as the number it is meant to be checked against.
    data = np.asarray(raw.get_data(picks=chosen))[:, ::step] * 1e6
    return {
        "job_id": job_id,
        "channels": [
            {"name": raw.ch_names[position].rstrip(" ."), "values": [round(float(value), 3) for value in row]}
            for position, row in zip(chosen, data)
        ],
        "sample_rate_effective": round(sampling_rate / step, 3),
        "duration_seconds": round(total_samples / sampling_rate, 3),
        "unit": "uV",
        "requested": list(SIGNAL_PREVIEW_CHANNELS),
        "matched": matched,
        "source": f"runtime/{job_id}/internal/{INTERNAL_CLEANED_FILE}",
    }


# --------------------------------------------------------------------------------------
# Step-wise internal API
#
# backend/graph.py already owns the review pipeline as a LangGraph state machine. These five
# routes expose its stages as independently callable HTTP steps so an external orchestrator
# (n8n) can drive them one at a time, without a second implementation of any domain logic:
# each route calls the same node function LangGraph would have called, and mirrors the routing
# the graph performs implicitly -- an invalid recording skips signal and prediction and goes
# straight to decision, and a stage that raises collapses into status="failed" and also routes
# to decision.
#
# Only the job id, small JSON metrics and paths cross the HTTP boundary: MNE objects are
# persisted under runtime/<job_id>/ and reloaded by the next step. Updates are appended to the
# existing live-job store, so the polling UI needs no changes to see this path progress.
# --------------------------------------------------------------------------------------

JOB_METADATA_FILE = "job.json"
INTERNAL_DIR = "internal"
INTERNAL_STATE_FILE = "state.json"
INTERNAL_RAW_FILE = "raw_eeg.fif"
INTERNAL_CLEANED_FILE = "signal_cleaned_eeg.fif"

# Mirrors the running messages backend.graph.stage() streams for each node, so the polling UI
# shows identical progress text whichever path produced it.
INTERNAL_RUNNING_MESSAGES = {
    "validation": "Reading EEG metadata and event markers.",
    "signal": "Cleaning the signal and measuring epochs.",
    "prediction": "Reading motor-imagery trials with FBCNet.",
    "decision": "Comparing model evidence with safety rules.",
    "report": "Preparing evidence for the reviewer.",
}

# The state keys backend.graph declares with an ``operator.add`` reducer.
INTERNAL_ACCUMULATED_KEYS = ("agent_updates", "audit_events")


class InternalStageRequest(BaseModel):
    """Body of every internal route that needs to know only which job to act on."""

    job_id: str = Field(min_length=1)


class InternalDecideRequest(BaseModel):
    """The upstream reports are optional: each is persisted per job, so an orchestrator may echo
    back what it received or pass nothing at all and get the same verdict."""

    job_id: str = Field(min_length=1)
    validation_report: ValidationReport | None = None
    quality_report: QualityReport | None = None
    classification_report: ClassificationReport | None = None


class InternalReportRequest(BaseModel):
    job_id: str = Field(min_length=1)
    policy_outcome: dict | None = None


def internal_dir(job_id: str) -> Path:
    """Where the heavy objects and the step-to-step state for one job live."""
    return RUNTIME_DIR / job_id / INTERNAL_DIR


def internal_state_path(job_id: str) -> Path:
    return internal_dir(job_id) / INTERNAL_STATE_FILE


def internal_read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def internal_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def internal_job_metadata(job_id: str) -> dict:
    """Resolve what a job id refers to.

    ``save_upload`` records this at upload time. A recording placed into the runtime directory
    by hand is reconstructed from the file itself, with the digest recomputed from its bytes.
    """
    directory = RUNTIME_DIR / job_id
    recorded = internal_read_json(directory / JOB_METADATA_FILE)
    if isinstance(recorded, dict) and "input_path" in recorded:
        return recorded
    candidates = sorted(path for path in directory.glob("original_eeg*") if path.is_file())
    if not candidates:
        raise HTTPException(404, f"No uploaded recording is registered for job '{job_id}'.")
    recording = candidates[0]
    return {
        "job_id": job_id,
        "filename": recording.name,
        "digest": hashlib.sha256(recording.read_bytes()).hexdigest(),
        "extension": recording.suffix.lower(),
        "input_path": str(recording),
    }


def internal_record_event(job_id: str, payload: dict) -> None:
    """Mirror ``process_job``'s twin audit logs so both paths leave one trace."""
    line = json.dumps(payload) + "\n"
    for directory in (RUNTIME_DIR / job_id, Path("jobs") / job_id):
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line)


def internal_publish(job_id: str, agent_update: AgentUpdate) -> None:
    """Append to the store the frontend already polls, and to the audit log."""
    live = live_jobs.get(job_id)
    if live is not None:
        live.agent_updates.append(agent_update)
    internal_record_event(job_id, {"type": "update", "data": agent_update.model_dump()})


def internal_outcome_to_json(outcome: PolicyOutcome) -> dict:
    return {
        "decision": outcome.decision,
        "gates": [gate.model_dump() for gate in outcome.gates],
        "signals": list(outcome.signals),
        "allowed_actions": list(outcome.allowed_actions),
        "recheck": outcome.recheck.as_dict(),
        "thresholds": outcome.thresholds.as_dict(),
    }


def internal_outcome_from_json(payload: dict) -> PolicyOutcome:
    recheck = payload.get("recheck") or {}
    return PolicyOutcome(
        decision=payload["decision"],
        gates=tuple(GateResult.model_validate(gate) for gate in payload.get("gates", [])),
        signals=tuple(payload.get("signals", [])),
        allowed_actions=tuple(payload.get("allowed_actions", [])),
        recheck=RecheckState(limit=int(recheck.get("limit", 1)), used=int(recheck.get("used", 0))),
        thresholds=PolicyThresholds(**payload.get("thresholds", {})),
    )


def internal_initial_state(metadata: dict) -> dict:
    """The same starting state ``graph.stream_pipeline`` builds for a fresh job."""
    return {
        "job_id": metadata["job_id"],
        "filename": metadata["filename"],
        "extension": metadata["extension"],
        "digest": metadata["digest"],
        "input_path": metadata["input_path"],
        "status": "ok",
        "failed_stage": None,
        "validation_report": None,
        "quality_report": None,
        "classification_report": None,
        "preprocessing": {},
        "epoching": {},
        "policy_outcome": None,
        "agent_updates": [],
        "audit_events": [event("uploaded", "Original file stored unchanged.")],
    }


def internal_node_state(persisted: dict) -> dict:
    """Rebuild a LangGraph state dict from the JSON that survived the previous HTTP step."""
    state = dict(persisted)
    state["input_path"] = Path(persisted["input_path"])
    state["validation_report"] = None if persisted.get("validation_report") is None else ValidationReport.model_validate(persisted["validation_report"])
    state["quality_report"] = None if persisted.get("quality_report") is None else QualityReport.model_validate(persisted["quality_report"])
    state["classification_report"] = None if persisted.get("classification_report") is None else ClassificationReport.model_validate(persisted["classification_report"])
    state["preprocessing"] = persisted.get("preprocessing") or {}
    state["epoching"] = persisted.get("epoching") or {}
    outcome = persisted.get("policy_outcome")
    state["policy_outcome"] = internal_outcome_from_json(outcome) if outcome else None
    state["agent_updates"] = list(persisted.get("agent_updates") or [])
    state["audit_events"] = list(persisted.get("audit_events") or [])
    return state


def internal_persist(state: dict) -> dict:
    """The JSON form of the pipeline state: everything except the heavy objects."""
    validation = state.get("validation_report")
    quality = state.get("quality_report")
    classification = state.get("classification_report")
    outcome = state.get("policy_outcome")
    return {
        "job_id": state["job_id"],
        "filename": state["filename"],
        "extension": state.get("extension", ""),
        "digest": state["digest"],
        "input_path": str(state["input_path"]),
        "status": state.get("status", "ok"),
        "failed_stage": state.get("failed_stage"),
        "validation_report": None if validation is None else validation.model_dump(),
        "quality_report": None if quality is None else quality.model_dump(),
        "classification_report": None if classification is None else classification.model_dump(),
        "preprocessing": state.get("preprocessing") or {},
        "epoching": state.get("epoching") or {},
        "policy_outcome": internal_outcome_to_json(outcome) if outcome else None,
        "agent_updates": state.get("agent_updates") or [],
        "audit_events": state.get("audit_events") or [],
    }


def internal_merge(state: dict, output: dict) -> dict:
    """Apply a node's partial result the way LangGraph's ``operator.add`` reducers would."""
    for key, value in output.items():
        if key in INTERNAL_ACCUMULATED_KEYS:
            state[key] = list(state.get(key) or []) + list(value)
        else:
            state[key] = value
    return state


def internal_run_node(name: str, node, state: dict, ctx: RunContext) -> dict:
    """Call one pipeline node, capturing an exception exactly as ``graph.stage`` does."""
    try:
        return node(state, ctx)
    except Exception as exc:
        return {
            "status": "failed",
            "failed_stage": name,
            "agent_updates": [failure_update(name, exc).model_dump()],
            "audit_events": [event("failed", f"{type(exc).__name__}: {exc}")],
        }


def internal_save_raw(path: Path, raw) -> None:
    """Persist an MNE object for the next step.

    ``fmt="double"`` is required: the default single precision would perturb the signal, and
    with it the probabilities, so the reloaded recording would no longer match the one the
    in-process graph classified.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(path, overwrite=True, fmt="double", verbose=False)


def internal_load_raw(job_id: str):
    """Reload the recording persisted by the validate step, falling back to the upload."""
    persisted = internal_dir(job_id) / INTERNAL_RAW_FILE
    if persisted.exists():
        return load_raw(persisted)
    return load_raw(Path(internal_job_metadata(job_id)["input_path"]))


def internal_begin(job_id: str, stage: str) -> dict:
    """Load or create a job's persisted state, then announce that a stage is starting."""
    persisted = internal_read_json(internal_state_path(job_id))
    if not isinstance(persisted, dict) or "job_id" not in persisted:
        persisted = internal_persist(internal_initial_state(internal_job_metadata(job_id)))
        internal_record_event(job_id, {"type": "start", "timestamp": datetime.now(UTC).isoformat(), "job_id": job_id})
    internal_publish(job_id, update(stage, "working", INTERNAL_RUNNING_MESSAGES[stage]))
    return internal_node_state(persisted)


def internal_commit(job_id: str, state: dict, output: dict, ctx: RunContext) -> dict:
    """Merge a node result, publish its updates, then durably record the new state."""
    merged = internal_merge(state, output)
    for payload in output.get("agent_updates", []):
        internal_publish(job_id, AgentUpdate(**payload))
    if ctx.raw is not None:
        internal_save_raw(internal_dir(job_id) / INTERNAL_RAW_FILE, ctx.raw)
    if ctx.cleaned is not None:
        internal_save_raw(internal_dir(job_id) / INTERNAL_CLEANED_FILE, ctx.cleaned)
    internal_write_json(internal_state_path(job_id), internal_persist(merged))
    return merged


def internal_failed(job_id: str, stage: str) -> dict:
    """The uniform failure envelope every route returns instead of raising a 500."""
    internal_record_event(job_id, {"type": "failed", "stage": stage, "timestamp": datetime.now(UTC).isoformat()})
    return {"job_id": job_id, "status": "failed", "failed_stage": stage}


@app.post("/internal/validate")
def internal_validate(request: InternalStageRequest) -> dict:
    """Read the recording metadata and say whether trial-based analysis can run.

    Wraps ``graph.validation_node`` -> ``backend.eeg.validate()``.
    """
    job_id = request.job_id
    try:
        state = internal_begin(job_id, "validation")
        ctx = RunContext()
        merged = internal_commit(job_id, state, internal_run_node("validation", validation_node, state, ctx), ctx)
        report = merged.get("validation_report")
        if report is None or merged.get("status") == "failed":
            return internal_failed(job_id, "validation")
        return {"job_id": job_id, "status": report.status, "validation_report": report.model_dump()}
    except HTTPException:
        raise
    except Exception as exc:
        internal_publish(job_id, failure_update("validation", exc))
        return internal_failed(job_id, "validation")


@app.post("/internal/signal")
def internal_signal(request: InternalStageRequest) -> dict:
    """Filter, re-reference and epoch the recording, persisting the heavy objects to disk.

    Wraps ``graph.signal_node`` -> ``backend.eeg.preprocess()`` + ``backend.eeg.epoch_and_measure()``.
    """
    job_id = request.job_id
    try:
        state = internal_begin(job_id, "signal")
        ctx = RunContext()
        ctx.raw = internal_load_raw(job_id)
        merged = internal_commit(job_id, state, internal_run_node("signal", signal_node, state, ctx), ctx)
        quality = merged.get("quality_report")
        if quality is None or merged.get("status") == "failed":
            return internal_failed(job_id, "signal")
        return {"job_id": job_id, "status": "ok", "quality_report": quality.model_dump()}
    except HTTPException:
        raise
    except Exception as exc:
        internal_publish(job_id, failure_update("signal", exc))
        return internal_failed(job_id, "signal")


@app.post("/internal/predict")
def internal_predict(request: InternalStageRequest) -> dict:
    """Run the FBCNet checkpoint over the persisted recording.

    Wraps ``graph.prediction_node`` -> ``backend.inference.classify()``. ``classify`` builds its
    own motor-imagery trials from a raw recording, so the object persisted for it is the MNE
    Raw rather than an epoch array.
    """
    job_id = request.job_id
    try:
        state = internal_begin(job_id, "prediction")
        ctx = RunContext()
        ctx.raw = internal_load_raw(job_id)
        merged = internal_commit(job_id, state, internal_run_node("prediction", prediction_node, state, ctx), ctx)
        classification = merged.get("classification_report")
        if classification is None or merged.get("status") == "failed":
            return internal_failed(job_id, "prediction")
        return {"job_id": job_id, "status": "ok", "classification_report": classification.model_dump()}
    except HTTPException:
        raise
    except Exception as exc:
        internal_publish(job_id, failure_update("prediction", exc))
        return internal_failed(job_id, "prediction")


@app.post("/internal/decide")
def internal_decide(request: InternalDecideRequest) -> dict:
    """Apply the deterministic policy.

    Wraps ``graph.decision_node`` -> ``backend.policy.evaluate()``. The route reproduces the
    graph's routing rather than trusting the caller: when the recording was rejected, or an
    earlier stage failed, the signal and prediction evidence is treated as absent, exactly as
    ``graph.route_after_validation`` / ``route_after_signal`` would have left it.
    """
    job_id = request.job_id
    try:
        state = internal_begin(job_id, "decision")
        usable = state.get("status", "ok") == "ok"

        def evidence(provided: Any, persisted: Any) -> Any:
            """A caller may echo a report back, but a stage the graph skipped stays skipped."""
            if not usable:
                return None
            return provided if provided is not None else persisted

        state["validation_report"] = request.validation_report or state.get("validation_report")
        state["quality_report"] = evidence(request.quality_report, state.get("quality_report"))
        state["classification_report"] = evidence(request.classification_report, state.get("classification_report"))
        ctx = RunContext()
        merged = internal_commit(job_id, state, internal_run_node("decision", decision_node, state, ctx), ctx)
        outcome = merged.get("policy_outcome")
        if outcome is None:
            return internal_failed(job_id, "decision")
        return {"job_id": job_id, "status": "ok", "policy_outcome": internal_outcome_to_json(outcome)}
    except HTTPException:
        raise
    except Exception as exc:
        internal_publish(job_id, failure_update("decision", exc))
        return internal_failed(job_id, "decision")


@app.post("/internal/report")
def internal_report(request: InternalReportRequest) -> dict:
    """Assemble the final JobReport and publish it to the store the UI already reads.

    Wraps ``graph.report_node`` -> ``backend.evidence.build_evidence()`` plus the JobReport
    assembly and the ``completed`` / ``invalid_input`` / ``processing_failed`` branching.
    """
    job_id = request.job_id
    try:
        state = internal_begin(job_id, "report")
        if request.policy_outcome is not None:
            state["policy_outcome"] = internal_outcome_from_json(request.policy_outcome)
        ctx = RunContext()
        merged = internal_commit(job_id, state, internal_run_node("report", report_node, state, ctx), ctx)
        report = merged.get("report")
        if report is None:
            return internal_failed(job_id, "report")
        jobs[job_id] = report
        live = live_jobs.get(job_id)
        if live is not None:
            live.status = report.status
            live.report = report
        internal_record_event(job_id, {"type": "report", "data": report.model_dump()})
        return report.model_dump()
    except HTTPException:
        raise
    except Exception as exc:
        internal_publish(job_id, failure_update("report", exc))
        return internal_failed(job_id, "report")


# Register the UI after API routes so /jobs is never shadowed by static-file routing.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
