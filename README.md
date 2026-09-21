# NeuroAgent

NeuroAgent is an EEG motor-imagery pipeline. It uses FBCNet for left/right-hand classification and a structured verification layer for reliability decisions. Implemented today: safe upload, validation, deterministic preprocessing, event epoching, basic quality metrics, left/right-hand prediction, calibrated confidence, a deterministic acceptance policy, an evidence bundle for reviewer agents, and a JSON report.

## Prototype setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000/docs` and use `POST /jobs` to upload a `.fif`, `.edf`, or `.bdf` recording. The result is a job id; fetch the report with `GET /jobs/{job_id}`.

## Classification

Every valid upload is also scored by the trained FBCNet checkpoint (`models/fbcnet_v1.pt`, override with `NEUROAGENT_MODEL_PATH`). The `classification` block of the report carries the predicted hand, mean per-class probabilities, the per-trial vote counts, and the share of trials agreeing with the verdict.

Inference reproduces the training preprocessing exactly — 160 Hz, 8-30 Hz band-pass, average reference, and 0.5-3.5 s trials locked to `T1`/`T2` annotations. Anything it cannot honour is reported as `classification.status = "unavailable"` with a reason rather than failing the job: no trained checkpoint, no `T1`/`T2` annotations, no trial that fits inside the recording, or a recording whose channel or sample count the checkpoint was not trained on. The checkpoint stores no channel names, so a recording is only comparable to it if its EEG channels are already in the training order (the 64-channel PhysioNet montage).

## Acceptance policy and the evidence bundle

The report's `decision` field is produced by `backend/policy.py`, which is deterministic: no language model is consulted, and every threshold is a frozen constant recorded in `evidence.provenance.thresholds` so no agent can quietly redefine "acceptable". Agents may *recommend*; only the policy *authorizes*.

`POST /jobs/{job_id}/review` records a human `approve`, `mark_uncertain`, or `override` decision and comment. It never rewrites the original automated decision, which preserves the audit trail. This is in-memory prototype storage; restart the API and jobs/reviews are cleared.

`evidence` is the single artifact a reviewer agent consumes — provenance, flat measurements, plain-language signals, every gate with what it observed, the remaining recheck budget, and the actions the verdict permits. `authorize()` refuses any action absent from that allowlist, and refuses `recheck_quality` once the budget is spent.

Order of precedence: invalid input is rejected; poor quality triggers a bounded recheck and then escalates to human review once the budget is gone; a missing or unusable prediction returns uncertain; otherwise the job is accepted, with a warning whenever a soft gate fails. Quality is measured per event-locked epoch: more than 25% of epochs above 300 uV, or any flat EEG channel, is poor quality; isolated amplitude artifacts become warnings. Note that the recheck *budget* is enforced here, but nothing yet performs a recheck run — that is the orchestration step, so `REQUEST_HUMAN_REVIEW` currently arises only when a spent budget is supplied.

### Why confidence cannot gate acceptance

`models/fbcnet_v1_calibration.json` records what a held-out split says about the model's confidence, and the result is decisive: `confidence_discriminative` is **false**. The most confident quartile of trials is only 0.024 more accurate than the base rate, and the high confidence band is *less* accurate (0.576) than the low band (0.599). Temperature scaling still matters — it turns a misleading 0.76 average confidence into an honest 0.55, halving the calibration error from 0.215 to 0.111 — but an honest number is not an informative one.

So plain `ACCEPT` is deliberately unreachable for this checkpoint, and the policy refuses to gate on confidence. A stronger model (or an agent reasoning over quality evidence instead) is what has to earn `ACCEPT` back.

## Phase 3 scope

The Phase 3 prototype includes upload UI, FBCNet inference, per-epoch signal-quality checks, a deterministic policy, evidence records, and human review. Input recordings must contain events; without them the job is reported as `invalid_input` rather than pretending to create task epochs. ICA, persistence, authentication, and automated recheck orchestration remain subsequent work. Predictions are reported as-is: at ~60% cross-subject test accuracy the model is a working end-to-end path, not a trustworthy verdict.

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
