let currentJobId = null;
let pollTimer = null;
let agentEvents = new Map();
let seenEvents = new Set();
const $ = (id) => document.getElementById(id);
const agentNames = { validation: "Recording check", signal: "Signal check", prediction: "Model read", decision: "Safety check", report: "Review brief" };
const readable = (value) => String(value).replaceAll("_", " ");

function setLive(text, running = false) {
  $("live-indicator").classList.toggle("running", running);
  $("live-indicator").lastElementChild.textContent = text;
  $("analysis-field").classList.toggle("is-processing", running);
}
function setCore(label, value, detail) { $("core-label").textContent = label; $("core-value").textContent = value; $("core-detail").textContent = detail; }
function node(agent) { return document.querySelector(`[data-agent="${agent}"]`); }
function renderAgent(update) {
  agentEvents.set(update.agent, update);
  const item = node(update.agent); if (!item) return;
  item.className = `agent-node ${update.agent} ${update.status}`;
  item.querySelector(".node-state").textContent = update.status === "working" ? "Working" : update.status;
  $("team-count").textContent = `${document.querySelectorAll(".agent-node.completed, .agent-node.blocked, .agent-node.failed").length} / 5`;
  if (update.status === "working") setCore("LIVE ANALYSIS", agentNames[update.agent], update.message);
}
function addActivity(update) {
  const key = `${update.timestamp}-${update.agent}-${update.status}`;
  if (seenEvents.has(key)) return; seenEvents.add(key);
  const feed = $("activity-feed"); feed.querySelector(".empty-log")?.remove();
  const entry = document.createElement("li"); const title = document.createElement("strong"); const detail = document.createElement("span");
  title.textContent = agentNames[update.agent]; detail.textContent = `${update.message}${update.duration_ms ? ` · ${update.duration_ms} ms` : ""}`;
  entry.append(title, detail); feed.prepend(entry);
}
function inspectAgent(agent) {
  document.querySelectorAll(".agent-node").forEach((item) => item.setAttribute("aria-pressed", String(item.dataset.agent === agent)));
  const update = agentEvents.get(agent); $("inspector-title").textContent = agentNames[agent]; $("inspector-message").textContent = update?.message || "This agent has not started yet.";
  const findings = $("inspector-findings"); findings.replaceChildren();
  Object.entries(update?.findings || {}).forEach(([key, value]) => { const row = document.createElement("div"); const term = document.createElement("dt"); const description = document.createElement("dd"); term.textContent = readable(key); description.textContent = Array.isArray(value) ? value.map(readable).join(", ") : readable(value); row.append(term, description); findings.append(row); });
}
function renderLive(state) {
  state.agent_updates.forEach((update) => { renderAgent(update); addActivity(update); });
  const last = state.agent_updates.at(-1); if (last && last.status !== "working") setCore("EVIDENCE TRACE", agentNames[last.agent], last.message);
  if (state.status !== "processing" && state.report) finish(state.report);
}
function metric(label, value) { const card = document.createElement("div"); const name = document.createElement("small"); const score = document.createElement("strong"); card.className = "metric"; name.textContent = label; score.textContent = value; card.append(name, score); return card; }
function finish(job) {
  clearInterval(pollTimer); pollTimer = null; $("submit").disabled = false;
  const prediction = job.classification?.predicted_label ? readable(job.classification.predicted_label) : "Uncertain";
  const confidence = job.classification?.probabilities?.[job.classification?.predicted_label];
  setLive(job.status === "completed" ? "Review ready" : "Review needs attention"); setCore("REVIEW STATE", prediction, readable(job.decision || job.status));
  $("field-status").textContent = job.status === "completed" ? "Evidence is ready for human review." : "The recording needs attention."; $("verdict").textContent = prediction; $("decision").textContent = readable(job.decision || job.status);
  $("metrics").replaceChildren(metric("model confidence", confidence == null ? "—" : `${Math.round(confidence * 100)}%`), metric("trial agreement", job.classification?.agreement == null ? "—" : `${Math.round(job.classification.agreement * 100)}%`), metric("valid trials", job.classification?.trial_count ?? "—"), metric("signal quality", job.quality?.status ?? "unavailable"));
  const signals = $("signals"); signals.replaceChildren(); [...(job.evidence?.signals || []), ...(job.warnings || [])].forEach((signal) => { const item = document.createElement("li"); item.textContent = signal; signals.append(item); }); $("results").classList.remove("hidden");
}
async function pollLiveJob() { const response = await fetch(`/live-jobs/${currentJobId}`); const state = await response.json(); if (!response.ok) throw new Error(state.detail || "Could not read live review status."); renderLive(state); }

$("recording").addEventListener("change", (event) => { $("file-name").textContent = event.target.files[0]?.name || "Choose an EEG recording"; });
$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault(); if (!$("recording").files[0]) return;
  clearInterval(pollTimer); agentEvents = new Map(); seenEvents = new Set(); $("results").classList.add("hidden"); $("submit").disabled = true;
  document.querySelectorAll(".agent-node").forEach((item) => { item.className = `agent-node ${item.dataset.agent}`; item.querySelector(".node-state").textContent = "Waiting"; }); $("activity-feed").replaceChildren(Object.assign(document.createElement("li"), { className: "empty-log", textContent: "No agent activity yet." }));
  $("status").textContent = "Uploading recording…"; $("field-status").textContent = "Opening a live evidence trace."; setLive("Receiving recording", true); setCore("INPUT", "Recording received", "Waiting for the first agent update.");
  try {
    const response = await fetch("/live-jobs", { method: "POST", body: new FormData($("upload-form")) }); const state = await response.json(); if (!response.ok) throw new Error(state.detail || "The recording could not be processed.");
    currentJobId = state.job_id; $("status").textContent = "Agents are reviewing the recording."; renderLive(state); await pollLiveJob(); if (!pollTimer) pollTimer = setInterval(() => pollLiveJob().catch((error) => { $("status").textContent = error.message; clearInterval(pollTimer); }), 450);
  } catch (error) { $("status").textContent = error.message; $("submit").disabled = false; setLive("Upload needs attention"); }
});
document.querySelectorAll(".agent-node").forEach((item) => item.addEventListener("click", () => inspectAgent(item.dataset.agent)));
$("review-action").addEventListener("change", () => $("override-field").classList.toggle("hidden", $("review-action").value !== "override"));
$("review-form").addEventListener("submit", async (event) => { event.preventDefault(); if (!currentJobId) return; const action = $("review-action").value; const body = { action, reviewer_comment: $("review-comment").value }; if (action === "override") body.approved_label = $("override-label").value; const response = await fetch(`/jobs/${currentJobId}/review`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }); const job = await response.json(); $("review-status").textContent = response.ok ? `Review recorded: ${readable(job.human_review.approved_label)}.` : (job.detail || "Could not record review."); });
