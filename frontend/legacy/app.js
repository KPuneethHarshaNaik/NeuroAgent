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
const pipelineAgents = ["validation", "signal", "prediction", "decision", "report"];
const pipelineCaptions = { validation: "Format, channels, markers", signal: "Filter, epochs, artifacts", prediction: "Trial-level classification", decision: "Policy and evidence gates", report: "Reviewer-ready summary" };
const handoffTravel = 700;
const handoffHold = 400;
const handoffInstant = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let pipelineNodes = [];
let pipelineLinks = [];
let handoffQueue = [];
let handoffRunning = false;
let pipelineCursor = 0;
let pipelineGeneration = 0;
let pipelineDragged = false;
const wait = (delay) => new Promise((resolve) => setTimeout(resolve, delay));
const pipelineCanvas = () => $("pipeline-canvas");
const nodeFor = (agent) => pipelineNodes.find((item) => item.key === agent);
const nodeCenter = (item) => ({ x: item.fx * pipelineCanvas().clientWidth, y: item.fy * pipelineCanvas().clientHeight });
const canvasPoint = (event) => { const box = pipelineCanvas().getBoundingClientRect(); return { x: event.clientX - box.left, y: event.clientY - box.top }; };
function edgePoint(item, target) {
  const center = nodeCenter(item);
  const half = { x: (item.el.offsetWidth || 150) / 2, y: (item.el.offsetHeight || 92) / 2 };
  const dx = target.x - center.x, dy = target.y - center.y;
  if (!dx && !dy) return center;
  const scale = Math.min(dx ? half.x / Math.abs(dx) : Infinity, dy ? half.y / Math.abs(dy) : Infinity);
  return { x: center.x + dx * scale, y: center.y + dy * scale };
}
function buildPipeline() {
  const specs = [{ key: "origin", step: "↑", title: "Upload", caption: "start · webhook", agent: null }].concat(pipelineAgents.map((agent, index) => ({ key: agent, step: String(index + 1).padStart(2, "0"), title: agentNames[agent], caption: pipelineCaptions[agent], agent })));
  pipelineNodes = specs.map((spec) => {
    const item = { ...spec, status: "idle", update: null, fx: 0, fy: 0 };
    const el = document.createElement("article"); el.className = `pipeline-node ${spec.key} idle`; el.dataset.pipelineAgent = spec.key; el.setAttribute("role", "button"); el.setAttribute("tabindex", "0"); el.setAttribute("aria-label", `${spec.title}: idle`);
    item.el = el; item.state = document.createElement("em"); item.state.className = "pipeline-state"; item.state.textContent = "idle"; item.message = document.createElement("p"); item.message.className = "pipeline-message"; item.duration = document.createElement("span"); item.duration.className = "pipeline-duration";
    const head = document.createElement("div"); head.className = "pipeline-head"; const step = document.createElement("span"); step.className = "pipeline-step"; step.textContent = spec.step; const title = document.createElement("strong"); title.textContent = spec.title; head.append(step, title);
    const foot = document.createElement("div"); foot.className = "pipeline-foot"; foot.append(item.state, item.duration);
    item.message.textContent = spec.caption;
    el.append(head, foot, item.message); el.addEventListener("pointerdown", (event) => startDrag(event, item)); el.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); showRawUpdate(item); } });
    pipelineCanvas().append(el); return item;
  });
  pipelineLinks = pipelineNodes.slice(0, -1).map(() => ({ el: null }));
}
function layoutPipeline() {
  if (pipelineDragged) return;
  const box = pipelineCanvas(), width = box.clientWidth, columns = Math.max(1, Math.min(pipelineNodes.length, Math.floor(width / 172)));
  const rows = Math.ceil(pipelineNodes.length / columns);
  box.style.height = `${Math.max(240, rows * 122)}px`;
  pipelineNodes.forEach((item, index) => {
    const row = Math.floor(index / columns), inRow = Math.min(columns, pipelineNodes.length - row * columns);
    item.fx = (index % columns + 0.5) / inRow; item.fy = (row + 0.5) / rows; item.el.style.left = `${item.fx * 100}%`; item.el.style.top = `${item.fy * 100}%`;
  });
  drawLinks();
}
function drawLinks() {
  pipelineLinks.forEach((link, index) => {
    const from = pipelineNodes[index], to = pipelineNodes[index + 1]; if (!from || !to) return;
    if (!link.el) { link.el = document.createElementNS("http://www.w3.org/2000/svg", "line"); $("pipeline-links").append(link.el); }
    const start = edgePoint(from, nodeCenter(to)), end = edgePoint(to, nodeCenter(from));
    link.el.setAttribute("x1", start.x); link.el.setAttribute("y1", start.y); link.el.setAttribute("x2", end.x); link.el.setAttribute("y2", end.y);
    link.el.classList.toggle("done", to.status === "done"); link.el.classList.toggle("active", to.status === "working");
  });
}
function setNodeState(item, status, message) {
  item.status = status; item.el.className = `pipeline-node ${item.key} ${status}${item.el.classList.contains("selected") ? " selected" : ""}`;
  item.state.textContent = status === "working" ? "working" : status;
  item.duration.textContent = item.update?.duration_ms ? `${item.update.duration_ms} ms` : "";
  item.el.setAttribute("aria-label", `${item.title}: ${status === "idle" ? "idle" : item.state.textContent}`);
  if (message !== undefined) item.message.textContent = message;
}
function applyNode(item, status, message) { setNodeState(item, status, message); drawLinks(); }
function handoffLabel(update) {
  const findings = update.findings || {};
  const labels = {
    validation: findings.event_markers != null ? `${findings.event_markers} markers` : findings.eeg_channels != null ? `${findings.eeg_channels} channels` : null,
    signal: findings.epochs != null ? `${findings.epochs} epochs${findings.quality ? ` · ${readable(findings.quality)}` : ""}` : null,
    prediction: findings.prediction ? `${readable(findings.prediction)}${findings.agreement == null ? "" : ` · ${Math.round(findings.agreement * 100)}%`}` : null,
    decision: findings.decision ? readable(findings.decision).toLowerCase() : null,
    report: findings.decision ? `brief · ${readable(findings.decision).toLowerCase()}` : null,
  };
  const label = labels[update.agent] || update.message || update.status;
  return label.length > 30 ? `${label.slice(0, 29)}…` : label;
}
async function playHandoff(update) {
  const generation = pipelineGeneration;
  const target = nodeFor(update.agent); if (!target) return;
  target.update = update;
  const status = update.status === "completed" ? "done" : update.status === "working" ? "working" : "failed";
  let source = pipelineNodes[pipelineNodes.indexOf(target) - 1];
  if (source && source.key !== "origin" && source.status === "idle") source = pipelineNodes[0];
  if (!source || source === target || target.status === "working" || handoffInstant) { applyNode(target, status, update.message); return; }
  const token = document.createElement("span"); token.className = `handoff-token${status === "failed" ? " failed" : ""}`; token.textContent = handoffLabel(update); pipelineCanvas().append(token);
  const startedAt = performance.now(); let live = true;
  // The frame loop is what makes this look like a hand-off, but a hidden or uncomposited tab never
  // fires requestAnimationFrame. Racing it against the wall clock keeps the queue draining regardless.
  const animated = new Promise((resolve) => {
    const frame = (now) => {
      if (!live) return resolve();
      const progress = Math.min(1, (now - startedAt) / handoffTravel); const eased = progress < .5 ? 2 * progress * progress : 1 - ((-2 * progress + 2) ** 2) / 2;
      const start = edgePoint(source, nodeCenter(target)), end = edgePoint(target, nodeCenter(source));
      token.style.left = `${start.x + (end.x - start.x) * eased}px`; token.style.top = `${start.y + (end.y - start.y) * eased}px`;
      if (progress < 1) requestAnimationFrame(frame); else resolve();
    };
    requestAnimationFrame(frame);
  });
  await Promise.race([animated, wait(handoffTravel + 500)]); live = false;
  token.remove(); if (generation !== pipelineGeneration) return;
  applyNode(target, "working", update.message);
  if (status !== "working") { await wait(handoffHold); if (generation !== pipelineGeneration) return; applyNode(target, status, update.message); }
}
async function drainHandoffs() {
  handoffRunning = true;
  while (handoffQueue.length) await playHandoff(handoffQueue.shift());
  handoffRunning = false;
}
function queueHandoff(updates) {
  if (!updates?.length) { pipelineCursor = 0; return; }
  if (updates.length < pipelineCursor) pipelineCursor = 0;
  const fresh = updates.slice(pipelineCursor); pipelineCursor = updates.length;
  if (!fresh.length) return;
  handoffQueue.push(...fresh); if (!handoffRunning) drainHandoffs();
}
function startDrag(event, item) {
  if (event.pointerType === "mouse" && event.button !== 0) return;
  event.preventDefault();
  const point = canvasPoint(event), center = nodeCenter(item);
  const offset = { x: center.x - point.x, y: center.y - point.y }; let moved = false;
  const onMove = (moveEvent) => {
    const next = canvasPoint(moveEvent); if (Math.hypot(next.x - point.x, next.y - point.y) > 3) moved = true;
    const box = pipelineCanvas(); const half = { x: (item.el.offsetWidth || 150) / 2 / box.clientWidth, y: (item.el.offsetHeight || 92) / 2 / box.clientHeight };
    const clamp = (value, margin) => margin >= .5 ? .5 : Math.min(1 - margin, Math.max(margin, value));
    item.fx = clamp((next.x + offset.x) / box.clientWidth, half.x); item.fy = clamp((next.y + offset.y) / box.clientHeight, half.y);
    item.el.style.left = `${item.fx * 100}%`; item.el.style.top = `${item.fy * 100}%`; drawLinks();
  };
  const onUp = () => {
    item.el.removeEventListener("pointermove", onMove); item.el.removeEventListener("pointerup", onUp); item.el.removeEventListener("pointercancel", onUp);
    if (moved) pipelineDragged = true; else if (item.update) showRawUpdate(item);
  };
  item.el.addEventListener("pointermove", onMove); item.el.addEventListener("pointerup", onUp); item.el.addEventListener("pointercancel", onUp);
  try { item.el.setPointerCapture(event.pointerId); } catch { /* a drag still works while the pointer stays over the box */ }
}
function showRawUpdate(item) {
  const panel = $("pipeline-json-panel");
  if (!item.update) return;
  const reopen = !panel.classList.contains("hidden") && panel.dataset.agent === item.key;
  pipelineNodes.forEach((other) => other.el.classList.toggle("selected", !reopen && other === item));
  if (reopen) { panel.classList.add("hidden"); return; }
  panel.dataset.agent = item.key; $("pipeline-json-title").textContent = `${agentNames[item.key] || item.title} · raw AgentUpdate`;
  $("pipeline-json-body").textContent = JSON.stringify(item.update, null, 2); panel.classList.remove("hidden"); panel.scrollIntoView({ block: "nearest" });
}
function resetPipeline() {
  pipelineGeneration += 1; handoffQueue = []; pipelineCursor = 0;
  pipelineCanvas().querySelectorAll(".handoff-token").forEach((token) => token.remove());
  pipelineNodes.forEach((item) => { item.update = null; applyNode(item, "idle", item.caption); });
  pipelineNodes.forEach((item) => item.el.classList.remove("selected"));
  $("pipeline-json-panel").classList.add("hidden"); delete $("pipeline-json-panel").dataset.agent;
  drawLinks();
}
function renderLive(state) {
  state.agent_updates.forEach((update) => { renderAgent(update); addActivity(update); });
  queueHandoff(state.agent_updates);
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
  clearInterval(pollTimer); agentEvents = new Map(); seenEvents = new Set(); resetPipeline(); $("results").classList.add("hidden"); $("submit").disabled = true;
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
buildPipeline(); layoutPipeline();
window.addEventListener("resize", () => { layoutPipeline(); drawLinks(); });
$("pipeline-json-close").addEventListener("click", () => { $("pipeline-json-panel").classList.add("hidden"); document.querySelectorAll(".pipeline-node.selected").forEach((item) => item.classList.remove("selected")); });
