import asyncio
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from backend.agents import decision_agent, failure_update, prediction_agent, report_agent, signal_agent, validation_agent
from backend.eeg import SUPPORTED_EXTENSIONS, load_raw
from backend.evidence import PIPELINE_VERSION, build_evidence
from backend.feedback import save_reviewer_feedback
from backend.graph import stream_pipeline
from backend.policy import POLICY_VERSION
from backend.schemas import AgentUpdate, HumanReview, HumanReviewRequest, JobReport, LiveJobStatus, ValidationReport

app = FastAPI(title="NeuroAgent", version=PIPELINE_VERSION)
RUNTIME_DIR = Path(os.getenv("NEUROAGENT_RUNTIME_DIR", "runtime"))
MAX_UPLOAD_BYTES = int(os.getenv("NEUROAGENT_MAX_UPLOAD_MB", "250")) * 1024 * 1024
jobs: dict[str, JobReport] = {}
live_jobs: dict[str, LiveJobStatus] = {}


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



@app.post("/jobs", response_model=JobReport, status_code=201)
async def create_job(file: UploadFile = File(...)) -> JobReport:
    job_id, filename, input_path, digest = await save_upload(file)
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


# Register the UI after API routes so /jobs is never shadowed by static-file routing.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
