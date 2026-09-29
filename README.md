# NeuroAgent

NeuroAgent is an EEG motor-imagination review pipeline. A recording is uploaded, five bounded agents read it in order (validation, signal, prediction, decision, report), a fixed deterministic policy authorizes the verdict, and a human reviewer records a separate decision that never overwrites the automated one. Every number in the report is traceable to the stage that produced it.

The backend is FastAPI. The frontend is now a single React 19 + Vite + Tailwind v4 app that builds into `frontend/` and is served by the API itself at `/` (landing) and `/app/` (the review workspace). The old vanilla tool is parked at `/legacy/` until the React tool is verified in its place.

## Quick start

```powershell
python -m venv .venv
.\\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000`. The landing page is at `/`; the review workspace is at `/app/`.

To run the tool with a real recording, upload an event-marked `.edf`, `.fif`, or `.bdf` on `/app/`. The upload is stored unchanged and hashed before any stage reads it, then `POST /live-jobs` starts the pipeline and the workspace polls `GET /live-jobs/{job_id}` every 450 ms until the job leaves `processing`. The finished report is also available at `GET /jobs/{job_id}`.

## What the workspace shows

`/app/` renders five panels:

- **Upload** — multipart `POST /live-jobs` (field name `file`), busy state during the upload, status line driven by the live-polled state.
- **Evidence checks** — one row per stage (validation, signal, prediction, decision, report), each showing the stage's latest `AgentUpdate` (status, message, duration, timestamp). A "settled" count excludes stages still `working`. When the report exists, deterministic gates are rendered beneath the rows.
- **Agent pipeline** — five absolute nodes in a row, SVG connectors between them, per-node drag (pointer capture, clamped centre fractions), and a hand-off token animated between nodes when a stage reports its update. Click a finished box for its raw `AgentUpdate`.
- **Signal Lens** — requests `GET /jobs/{job_id}/signal-preview` once the signal stage has actually settled, draws C3/Cz/C4 traces when a cleaned signal file is persisted, and shows a quality block when the report has a `quality` section. There is no illustrative fallback trace.
- **Review brief** — shows the automated verdict (`decision`), the FBCNet probabilities and calibration, the evidence signals and warnings, and a reviewer form (`approve` / `mark_uncertain` / `override`). `POST /jobs/{job_id}/review` records the reviewer decision next to the policy verdict.

Resume a finished or running job with `/app/?job={job_id}`. The workspace attaches to it (polling a live entry, or falling back to `GET /jobs/{job_id}` when the in-memory live entry is gone) and renders the report as it would appear at the end of a live run.

## Routes

The API has these paths (all pre-existing except `signal-preview`, which is additive and read-only):

- `GET /health` → `{status, phase, pipeline_version, policy_version}`
- `POST /jobs` → 201 `JobReport` (multipart `file`; blocking; built-in pipeline runs in-process unless `N8N_WEBHOOK_URL` hands it off)
- `POST /live-jobs` → 202 `LiveJobStatus` (multipart `file`; non-blocking; the workspace polls this)
- `GET /live-jobs/{job_id}` → `LiveJobStatus` (polling target; stops when status leaves `processing`)
- `GET /jobs/{job_id}` → `JobReport`
- `POST /jobs/{job_id}/review` → `JobReport` (body `{action, reviewer_comment, approved_label?}`)
- `GET /jobs/{job_id}/signal-preview` → `{job_id, channels, sample_rate_effective, duration_seconds, unit, requested, matched, source}`

`POST /jobs/{job_id}/review` refuses `approve` when there is no prediction (409 `"A prediction is required before it can be approved."`) and refuses an `override` without a label (422). `approve` sets the reviewed label to the predicted label; `mark_uncertain` sets it to `"uncertain"`; `override` sets it to the supplied `left_hand` or `right_hand`. The automated `classification.predicted_label` is preserved beside the reviewer's decision in the stored report.

## Signal Lens and the cleaned-signal file

`GET /jobs/{job_id}/signal-preview` reads the cleaned continuous recording the signal stage wrote (`runtime/{job_id}/internal/signal_cleaned_eeg.fif`, 8-30 Hz band-pass, average reference, decimated for transport). It returns C3, Cz, C4 when the montage exposes them by name, or the first three EEG channels with `matched: false` when it does not. For jobs processed through the `/internal/*` steps (the n8n-driven path) that file is persisted and the trace endpoint returns real values; for jobs processed by the built-in pipeline (the in-process path, which is what `/app/` uses when `N8N_WEBHOOK_URL` is empty), the cleaned recording is kept in memory only and the endpoint returns a 404 with a clear reason. That 404 is intentional and is shown in the workspace as the honest "no cleaned signal persisted" state.

## The merged-review-vs-policy note

The workspace's review form and the API both keep two things side by side: the automated `classification.predicted_label` (what FBCNet produced) and the reviewer's `approved_label` (what the human recorded). The policy verdict (`decision`) is never rewritten by a review. The brief displays the automated verdict as the decision; the reviewer panel displays the human decision next to it. That ordering is deliberate: reviewers act on evidence, and the audit trail keeps the automated and human outcomes distinct.

## Built-in pipeline and the live update asymmetry

The built-in `stream_pipeline` publishes a `working` update for each stage but does not, by itself, publish the `completed` updates for signal, prediction, decision, and report to the live store — only the final `JobReport.agent_updates` carries all five completed updates with durations. The workspace merges the report's completed updates into its polled list when the live entry settles, so the pipeline panel and evidence rows resolve to the real completed state instead of staying stuck on `working`. The live store itself is unchanged (that is the honest server-side state); the merge is a client-side reconciliation.

## Step-wise HTTP API for n8n

`backend/main.py` also exposes the pipeline as five step-wise routes, so an orchestrator such as n8n can drive one stage at a time instead of waiting on a single blocking upload. Each route runs the *same* node function the LangGraph pipeline runs, so a job driven this way produces the same `JobReport` as the in-process pipeline. Only the job id, small JSON metrics, and paths cross the wire; MNE objects are never serialized into a response but kept on disk under `runtime/{job_id}/internal/`. Because `save_upload` records what each job id refers to, every route needs nothing but the id:

| Route | Body | Response |
| --- | --- | --- |
| `POST /internal/validate` | `{job_id}` | `{job_id, status, validation_report}` |
| `POST /internal/signal` | `{job_id}` | `{job_id, status, quality_report}` |
| `POST /internal/predict` | `{job_id}` | `{job_id, status, classification_report}` |
| `POST /internal/decide` | `{job_id, validation_report?, quality_report?, classification_report?}` | `{job_id, status, policy_outcome}` |
| `POST /internal/report` | `{job_id, policy_outcome?}` | the full `JobReport` |

`status` is `valid` or `invalid` for `validate` (mirroring `validation_report.status`), `ok` for `signal`, `predict`, and `decide`, and `failed` — accompanied by a `failed_stage` field — when a stage raised. No route returns a 500 for a pipeline problem: a rejected or unreadable recording still reaches `/internal/report`, which produces the `invalid_input` or `processing_failed` report. `/internal/decide` deliberately reproduces the graph's routing, so when the recording was rejected or an earlier stage failed the signal and prediction evidence is treated as *absent* rather than taken from the request body.

Call the routes in order with a job id from `POST /jobs` or `POST /live-jobs`. They append `AgentUpdate`s to the same live-job store the UI already polls, so the existing frontend shows this path's progress with no changes.

### Importing the n8n workflows

Import the two exported workflows — the pipeline one and the reviewer one — with **Workflows -> Import from File** in the n8n editor, then activate them. The pipeline webhook is `POST /webhook/neuroagent-run` and takes `{ "job_id": "..." }`; it walks `validate -> signal -> predict -> decide -> report`. The reviewer workflow's webhook takes `{job_id, action, note}` and forwards it to the existing `POST /jobs/{job_id}/review`, so human decisions land in the same audit trail as the automated verdict.

Point the HTTP Request nodes at the backend with the `NEUROAGENT_BACKEND_URL` environment variable on the n8n process, which n8n expressions read as `$env.NEUROAGENT_BACKEND_URL`. Use `http://127.0.0.1:8000` for a backend on the same host, and note that n8n must be able to reach it — `127.0.0.1` resolves to the *n8n* container when n8n runs in Docker.

```powershell
$env:NEUROAGENT_BACKEND_URL = "http://127.0.0.1:8000"
```

## Frontend build

The web app lives in `web/` and is built with `npm run build`, which emits into `frontend/` (the same directory the API serves statically). The build script cleans only what it emits (`frontend/assets`, `frontend/index.html`, `frontend/app`) and never touches `frontend/legacy/`. Do not hand-edit files under `frontend/assets/` — rerun the build instead. The served tree is `/` (landing), `/app/` (tool), `/legacy/` (vanilla tool), plus the API routes and the static assets under `/assets/`.

```powershell
cd web
npm install
cd ..
npm run build
uvicorn backend.main:app --reload
```

The development server binds to `http://localhost:5173` (IPv6 only in the current Vite config); the production server is the FastAPI process on `http://127.0.0.1:8000`.

## Classification

Every valid upload is also scored by the trained FBCNet checkpoint (`models/fbcnet_v1.pt`, override with `NEUROAGENT_MODEL_PATH`). The `classification` block of the report carries the predicted hand, mean per-class probabilities, the per-trial vote counts, and the share of trials agreeing with the verdict.

Inference reproduces the training preprocessing exactly — 160 Hz, 8-30 Hz band-pass, average reference, and 0.5-3.5 s trials locked to `T1`/`T2` annotations. Anything it cannot honour is reported as `classification.status = "unavailable"` with a reason rather than failing the job: no trained checkpoint, no `T1`/`T2` annotations, no trial that fits inside the recording, or a recording whose channel or sample count the checkpoint was not trained on. The checkpoint stores no channel names, so a recording is only comparable to it if its EEG channels are already in the training order (the 64-channel PhysioNet montage).

## Acceptance policy and the evidence bundle

The report's `decision` field is produced by `backend/policy.py`, which is deterministic: no language model is consulted, and every threshold is a frozen constant recorded in `evidence.provenance.thresholds` so no agent can quietly redefine "acceptable". Agents may *recommend*; only the policy *authorizes*.

`POST /jobs/{job_id}/review` records a human `approve`, `mark_uncertain`, or `override` decision and comment. It never rewrites the original automated decision, which preserves the audit trail. This is in-memory prototype storage; restart the API and jobs/reviews are cleared (the workspace's resume path reads `GET /jobs/{job_id}` for finished jobs, so a review recorded before a restart is still reachable through that endpoint until the process is restarted).

`evidence` is the single artifact a reviewer agent consumes — provenance, flat measurements, plain-language signals, every gate with what it observed, the remaining recheck budget, and the actions the verdict permits. `authorize()` refuses any action absent from that allowlist, and refuses `recheck_quality` once the budget is spent.

Order of precedence: invalid input is rejected; poor quality triggers a bounded recheck and then escalates to human review once the budget is gone; a missing or unusable prediction returns uncertain; otherwise the job is accepted, with a warning whenever a soft gate fails. Quality is measured per event-locked epoch: more than 25% of epochs above 300 uV, or any flat EEG channel, is poor quality; isolated amplitude artifacts become warnings. Note that the recheck *budget* is enforced here, but nothing yet performs a recheck run — that is the orchestration step, so `REQUEST_HUMAN_REVIEW` currently arises only when a spent budget is supplied.

## Checks

```powershell
python -m unittest discover -s tests
```

## Prepare motor-imagery training data

The PhysioNet EEG Motor Movement/Imagery data is prepared from `R04`, `R08`, and `R12` only. `T1` becomes `left_hand`; `T2` becomes `right_hand`; `T0` is excluded.

```powershell
python -m training.prepare_dataset `
  --dataset-root "C:\Users\punee\Desktop\NeuroAgent\eeg-motor-movementimagery-dataset-1.0.0\files" `
  --output-dir data\processed\motor_imagery
```

## Train FBCNet

FBCNet training requires an NVIDIA CUDA GPU. It stops immediately if CUDA is unavailable, so a CPU run cannot silently produce a slower, non-comparable checkpoint. Verify your environment first:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

```powershell
python -m training.train_fbcnet --data-dir data\processed\motor_imagery
```

## Prepare and tune all eight actions

The multiclass model keeps execution and imagery separate: actual/imagined left hand, right hand, both fists, and both feet. It is trained independently and never replaces the binary hand-imagery checkpoint.

```powershell
python -m training.prepare_all_actions `
  --dataset-root "C:\Users\punee\Desktop\NeuroAgent\eeg-motor-movementimagery-dataset-1.0.0\files"
python -m training.tune_all_actions_fbcnet
```

## Accuracy experiments

Each command saves metrics to its model JSON file. Use the same data split for every comparison.

```powershell
# Up to 100 GPU epochs; early stopping keeps the best validation checkpoint.
python -m training.tune_all_actions_fbcnet --epochs 100

# Find which simple decision is difficult.
python -m training.train_fbcnet --data-dir data\processed\all_actions --task movement_type --model-path models\movement_type.pt --epochs 100
python -m training.train_fbcnet --data-dir data\processed\all_actions --task effector --model-path models\effector.pt --epochs 100
python -m training.train_fbcnet --data-dir data\processed\all_actions --task hand_side --model-path models\hand_side.pt --epochs 100

# Make a separate data variant, then train it. Do not overwrite the original.
python -m training.prepare_all_actions --dataset-root "C:\Users\punee\Desktop\NeuroAgent\eeg-motor-movementimagery-dataset-1.0.0\files" --output-dir data\processed\all_actions_mu --low-hz 8 --high-hz 13
python -m training.train_fbcnet --data-dir data\processed\all_actions_mu --model-path models\all_actions_mu.pt --epochs 100

# Compare the simple CSP + LDA baseline on the same task and split.
python -m training.train_csp_lda --data-dir data\processed\all_actions --task all_actions --model-path models\csp_lda_all_actions.pkl
```

## Train the CSP + LDA baseline

This baseline uses exactly the same prepared epochs and subject-wise split as FBCNet.

```powershell
python -m training.train_csp_lda --data-dir data\processed\motor_imagery
```

## Measure and calibrate confidence

Run this after training so the policy has evidence about whether the model's probability is worth trusting. The artifact is optional at inference time: without it probabilities stay uncalibrated and the policy refuses to claim acceptance.

```powershell
python -m training.calibrate_fbcnet --data-dir data\processed\motor_imagery
```
