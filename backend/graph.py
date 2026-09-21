"""LangGraph orchestration graph for the NeuroAgent EEG review pipeline."""

from __future__ import annotations

import functools
import operator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Callable, Generator, Literal, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from backend.agents import failure_update, update
from backend.eeg import epoch_and_measure, load_raw, preprocess, validate
from backend.evidence import POLICY_VERSION, build_evidence
from backend.inference import classify
from backend.policy import PolicyOutcome, evaluate
from backend.schemas import (
    AgentUpdate,
    ClassificationReport,
    EvidenceBundle,
    JobReport,
    QualityReport,
    ValidationReport,
)


@dataclass
class RunContext:
    """Heavy non-serializable objects (MNE Raw/Epochs) passed outside graph state."""

    raw: Any = None
    epochs: Any = None
    cleaned: Any = None


class PipelineState(TypedDict, total=False):
    """Typed State for LangGraph execution with reducer accumulators."""

    # Inputs
    job_id: str
    filename: str
    extension: str
    digest: str
    input_path: Path

    # Status
    status: Literal["ok", "invalid", "failed"]
    failed_stage: Optional[str]

    # Stage outputs (Pydantic models / dicts)
    validation_report: Optional[ValidationReport]
    quality_report: Optional[QualityReport]
    classification_report: Optional[ClassificationReport]
    preprocessing: dict
    epoching: dict
    policy_outcome: Optional[PolicyOutcome]
    report: Optional[JobReport]

    # Accumulators with reducers
    agent_updates: Annotated[list[dict], operator.add]
    audit_events: Annotated[list[dict], operator.add]


def _event(stage: str, detail: str) -> dict:
    return {"timestamp": datetime.now(UTC).isoformat(), "stage": stage, "detail": detail}


def stage(name: str):
    """Wrapper used by every node. Emits running update and captures uncaught exceptions into failure state."""

    def wrap(fn: Callable):
        def inner(state: PipelineState, config: RunnableConfig) -> dict:
            try:
                writer = get_stream_writer()
                if writer:
                    messages = {
                        "validation": "Reading EEG metadata and event markers.",
                        "signal": "Cleaning the signal and measuring epochs.",
                        "prediction": "Reading motor-imagery trials with FBCNet.",
                        "decision": "Comparing model evidence with safety rules.",
                        "report": "Preparing evidence for the reviewer.",
                    }
                    running_update = update(name, "working", messages.get(name, f"Running {name} stage..."))
                    writer(running_update.model_dump())
            except Exception:
                pass

            if isinstance(config, dict):
                ctx = config.get("configurable", {}).get("ctx")
            else:
                ctx = getattr(config, "configurable", {}).get("ctx") if config else None

            if ctx is None:
                ctx = RunContext()

            try:
                return fn(state, ctx)
            except Exception as exc:
                fail_upd = failure_update(name, exc).model_dump()
                fail_evt = _event("failed", f"{type(exc).__name__}: {exc}")
                return {
                    "status": "failed",
                    "failed_stage": name,
                    "agent_updates": [fail_upd],
                    "audit_events": [fail_evt],
                }

        return inner

    return wrap


def validation_node(state: PipelineState, ctx: RunContext) -> dict:
    started_at = perf_counter()
    if ctx.raw is None:
        ctx.raw = load_raw(state["input_path"])
    report = validate(ctx.raw, state["extension"])
    message = "EEG channels and event markers are ready." if report.status == "valid" else "The recording cannot be used for trial-based analysis."
    status_str = "completed" if report.status == "valid" else "blocked"
    upd = update("validation", status_str, message, {"eeg_channels": report.eeg_channel_count, "event_markers": report.event_count}, started_at)

    new_status = "ok" if report.status == "valid" else "invalid"
    audit = [_event("validating", "Reading EEG metadata and events.")]
    if report.status != "valid":
        audit.append(_event("rejected", "Input is missing EEG channels or events."))

    return {
        "status": new_status,
        "validation_report": report,
        "agent_updates": [upd.model_dump()],
        "audit_events": audit,
    }


def signal_node(state: PipelineState, ctx: RunContext) -> dict:
    started_at = perf_counter()
    if ctx.raw is None:
        ctx.raw = load_raw(state["input_path"])
    cleaned, preprocessing = preprocess(ctx.raw)
    ctx.cleaned = cleaned
    epochs, quality = epoch_and_measure(cleaned)
    ctx.epochs = epochs
    message = f"Created {epochs['epoch_count']} epochs; signal quality is {quality.status}."
    upd = update("signal", "completed", message, {"epochs": epochs["epoch_count"], "quality": quality.status, "rejected_epoch_ratio": quality.rejected_epoch_ratio}, started_at)

    audit = [
        _event("preprocessing", "Applied 8-30 Hz band-pass and average reference."),
        _event("epoching", f"Created {epochs['epoch_count']} event-locked epochs."),
    ]
    return {
        "preprocessing": preprocessing,
        "epoching": epochs,
        "quality_report": quality,
        "agent_updates": [upd.model_dump()],
        "audit_events": audit,
    }


def prediction_node(state: PipelineState, ctx: RunContext) -> dict:
    started_at = perf_counter()
    if ctx.raw is None:
        ctx.raw = load_raw(state["input_path"])
    report = classify(ctx.raw)

    if report.status == "classified":
        message = f"Predicted {report.predicted_label} from {report.trial_count} trials."
        findings = {"prediction": report.predicted_label, "agreement": report.agreement, "trials": report.trial_count}
        upd = update("prediction", "completed", message, findings, started_at)
        cls_event = _event("classification", f"{report.model} predicted {report.predicted_label} from {report.trial_count} motor-imagery trial(s).")
    else:
        message = report.reason or "Prediction is unavailable."
        upd = update("prediction", "blocked", message, {"reason_code": report.reason_code}, started_at)
        cls_event = _event("classification_skipped", message)

    return {
        "classification_report": report,
        "agent_updates": [upd.model_dump()],
        "audit_events": [cls_event],
    }


def decision_node(state: PipelineState, ctx: RunContext) -> dict:
    started_at = perf_counter()
    val = state.get("validation_report")
    qual = state.get("quality_report")
    cls_rep = state.get("classification_report")

    outcome = evaluate(val, qual, cls_rep)
    upd = update(
        "decision",
        "completed",
        f"Policy decision: {outcome.decision.replace('_', ' ').lower()}.",
        {"decision": outcome.decision, "allowed_actions": list(outcome.allowed_actions)},
        started_at,
    )

    return {
        "policy_outcome": outcome,
        "agent_updates": [upd.model_dump()],
        "audit_events": [_event("policy", f"{outcome.decision} (policy {POLICY_VERSION})")],
    }


def report_node(state: PipelineState, ctx: RunContext) -> dict:
    job_id = state["job_id"]
    filename = state["filename"]
    digest = state["digest"]
    status = state.get("status", "ok")
    failed_stage = state.get("failed_stage")

    val = state.get("validation_report")
    qual = state.get("quality_report")
    cls_rep = state.get("classification_report")
    prep = state.get("preprocessing", {})
    ep = state.get("epoching", {})
    outcome = state.get("policy_outcome")

    agent_updates_models = [AgentUpdate(**u) for u in state.get("agent_updates", [])]
    decision_str = outcome.decision if outcome else "REJECT_INVALID_INPUT"
    agent_updates_models.append(update("report", "completed", "Evidence bundle prepared for reviewer.", {"decision": decision_str}))

    audit = state.get("audit_events", [])

    if status == "failed":
        ext = state.get("extension", "")
        if val is None:
            val = ValidationReport(status="invalid", file_format=ext.removeprefix("."), warnings=["EEG file could not be read or processed."])
        err_msg = f"Pipeline failed at stage '{failed_stage}'."
        report = JobReport(
            job_id=job_id,
            status="processing_failed",
            input_filename=filename,
            input_hash=digest,
            validation=val,
            warnings=[err_msg],
            audit_events=audit,
            agent_updates=agent_updates_models,
        )
    elif status == "invalid":
        evidence = build_evidence(job_id, filename, digest, val, {}, {}, None, None, outcome)
        report = JobReport(
            job_id=job_id,
            status="invalid_input",
            input_filename=filename,
            input_hash=digest,
            validation=val,
            decision=outcome.decision if outcome else None,
            evidence=evidence,
            warnings=val.warnings if val else [],
            audit_events=audit,
            agent_updates=agent_updates_models,
        )
    else:
        evidence = build_evidence(job_id, filename, digest, val, prep, ep, qual, cls_rep, outcome)
        warnings = []
        if qual and qual.warnings:
            warnings.extend(qual.warnings)
        if cls_rep and cls_rep.warnings:
            warnings.extend(cls_rep.warnings)
        report = JobReport(
            job_id=job_id,
            status="completed",
            input_filename=filename,
            input_hash=digest,
            validation=val,
            decision=outcome.decision if outcome else None,
            preprocessing=prep,
            epoching=ep,
            quality=qual,
            classification=cls_rep,
            evidence=evidence,
            warnings=warnings,
            audit_events=audit,
            agent_updates=agent_updates_models,
        )

    return {"report": report}


def route_after_validation(state: PipelineState) -> Literal["signal", "decision"]:
    if state.get("status") in ("invalid", "failed"):
        return "decision"
    return "signal"


def route_after_signal(state: PipelineState) -> Literal["prediction", "decision"]:
    if state.get("status") == "failed":
        return "decision"
    return "prediction"


def build_graph(mode: str = "rules"):
    if mode != "rules":
        raise NotImplementedError(f"Mode '{mode}' is reserved for Phase 2.")

    g = StateGraph(PipelineState)
    g.add_node("validation", stage("validation")(validation_node))
    g.add_node("signal", stage("signal")(signal_node))
    g.add_node("prediction", stage("prediction")(prediction_node))
    g.add_node("decision", stage("decision")(decision_node))
    g.add_node("report", stage("report")(report_node))

    g.add_edge(START, "validation")
    g.add_conditional_edges("validation", route_after_validation, {"signal": "signal", "decision": "decision"})
    g.add_conditional_edges("signal", route_after_signal, {"prediction": "prediction", "decision": "decision"})
    g.add_edge("prediction", "decision")
    g.add_edge("decision", "report")
    g.add_edge("report", END)

    return g.compile()


def stream_pipeline(job_id: str, filename: str, input_path: Path, digest: str) -> Generator[AgentUpdate | JobReport, None, None]:
    graph = build_graph(mode="rules")
    ctx = RunContext()
    init_state: PipelineState = {
        "job_id": job_id,
        "filename": filename,
        "extension": input_path.suffix.lower(),
        "digest": digest,
        "input_path": input_path,
        "status": "ok",
        "agent_updates": [],
        "audit_events": [_event("uploaded", "Original file stored unchanged.")],
    }
    config = {"configurable": {"ctx": ctx}}

    emitted_count = 0
    final_report: Optional[JobReport] = None

    try:
        for mode, chunk in graph.stream(init_state, config, stream_mode=["updates", "custom"]):
            if mode == "custom":
                if isinstance(chunk, dict) and "agent" in chunk:
                    yield AgentUpdate(**chunk)
            elif mode == "updates":
                if isinstance(chunk, dict):
                    for _node_name, node_update in chunk.items():
                        if isinstance(node_update, dict):
                            if "agent_updates" in node_update and isinstance(node_update["agent_updates"], list):
                                for upd_dict in node_update["agent_updates"][emitted_count:]:
                                    yield AgentUpdate(**upd_dict)
                                emitted_count = len(node_update["agent_updates"])
                            if "report" in node_update:
                                final_report = node_update["report"]
    finally:
        ctx.raw = None
        ctx.epochs = None
        ctx.cleaned = None

    if final_report:
        yield final_report


def run_pipeline(job_id: str, filename: str, input_path: Path, digest: str) -> JobReport:
    final_report = None
    for item in stream_pipeline(job_id, filename, input_path, digest):
        if isinstance(item, JobReport):
            final_report = item
    if final_report is None:
        raise RuntimeError("Pipeline failed to produce a JobReport.")
    return final_report
