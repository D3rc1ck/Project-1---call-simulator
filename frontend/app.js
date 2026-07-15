"use strict";

const $ = (id) => document.getElementById(id);

const pickerView = $("picker-view");
const callView = $("call-view");
const scenarioList = $("scenario-list");
const voiceSelect = $("voice-select");
const transcriptEl = $("transcript");
const statusEl = $("status");
const recordBtn = $("record-btn");
const textInput = $("text-input");
const textSendBtn = $("text-send-btn");
const endCallBtn = $("end-call-btn");

let sessionId = null;
let mediaRecorder = null;
let recordedChunks = [];
let recording = false;
let busy = false;

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

async function init() {
  try {
    const [scenarios, voices] = await Promise.all([
      fetchJSON("/api/scenarios"),
      fetchJSON("/api/voices"),
    ]);
    renderScenarios(scenarios);
    for (const v of voices) {
      const opt = document.createElement("option");
      opt.value = v.id;
      opt.textContent = v.name;
      voiceSelect.appendChild(opt);
    }
  } catch (err) {
    scenarioList.innerHTML =
      `<p class="error">Could not reach the backend: ${err.message}</p>`;
  }
}

function renderScenarios(scenarios) {
  scenarioList.innerHTML = "";
  for (const s of scenarios) {
    const card = document.createElement("button");
    card.className = "scenario-card";
    card.innerHTML =
      `<strong>${escapeHtml(s.title)}</strong>` +
      `<span>${escapeHtml(s.description)}</span>` +
      `<em>You play: ${escapeHtml(s.role)}</em>`;
    card.addEventListener("click", () => startCall(s));
    scenarioList.appendChild(card);
  }
}

// ---------------------------------------------------------------------------
// Call lifecycle
// ---------------------------------------------------------------------------

async function startCall(scenario) {
  setStatus("Dialing…");
  pickerView.classList.add("hidden");
  callView.classList.remove("hidden");
  $("call-title").textContent = scenario.title;
  $("call-role").textContent = scenario.role;
  transcriptEl.innerHTML = "";

  try {
    const res = await fetchJSON("/api/call/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scenario_id: scenario.id,
        voice: voiceSelect.value || null,
      }),
    });
    sessionId = res.session_id;
    addBubble("caller", res.reply_text);
    await playAudio(res.audio_b64);
    setStatus("Your turn — click the mic or type a reply.");
    recordBtn.disabled = false;
  } catch (err) {
    setStatus(`Failed to start call: ${err.message}`, true);
  }
}

async function endCall() {
  if (sessionId) {
    fetch("/api/call/end", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    }).catch(() => {});
  }
  sessionId = null;
  stopRecorder();
  recordBtn.disabled = true;
  callView.classList.add("hidden");
  pickerView.classList.remove("hidden");
}

function finishCall() {
  sessionId = null;
  recordBtn.disabled = true;
  setStatus("Call ended. 👋");
}

// ---------------------------------------------------------------------------
// Recording (Whisper STT)
// ---------------------------------------------------------------------------

function pickMimeType() {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"];
  return candidates.find((t) => MediaRecorder.isTypeSupported(t)) || "";
}

async function toggleRecording() {
  if (busy) return;
  if (recording) {
    mediaRecorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = pickMimeType();
    mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    recordedChunks = [];
    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      recording = false;
      recordBtn.classList.remove("recording");
      recordBtn.textContent = "🎤 Click to talk";
      const blob = new Blob(recordedChunks, {
        type: mediaRecorder.mimeType || "audio/webm",
      });
      if (blob.size > 0) sendAudioTurn(blob);
    };
    mediaRecorder.start();
    recording = true;
    recordBtn.classList.add("recording");
    recordBtn.textContent = "⏹ Stop & send";
    setStatus("Recording — click again when you're done.");
  } catch (err) {
    setStatus(`Microphone unavailable: ${err.message}`, true);
  }
}

function stopRecorder() {
  if (mediaRecorder && recording) {
    mediaRecorder.onstop = null;
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach((t) => t.stop());
    recording = false;
    recordBtn.classList.remove("recording");
    recordBtn.textContent = "🎤 Click to talk";
  }
}

// ---------------------------------------------------------------------------
// Turns
// ---------------------------------------------------------------------------

async function sendAudioTurn(blob) {
  if (!sessionId) return;
  setBusy(true);
  setStatus("Transcribing with Whisper…");
  const form = new FormData();
  form.append("session_id", sessionId);
  const ext = blob.type.includes("mp4") ? "mp4" : blob.type.includes("ogg") ? "ogg" : "webm";
  form.append("audio", blob, `turn.${ext}`);
  try {
    const res = await fetchJSON("/api/call/turn", { method: "POST", body: form });
    await handleTurnResponse(res);
  } catch (err) {
    setStatus(`Turn failed: ${err.message}`, true);
  } finally {
    setBusy(false);
  }
}

async function sendTextTurn() {
  const text = textInput.value.trim();
  if (!text || !sessionId || busy) return;
  textInput.value = "";
  setBusy(true);
  setStatus("Thinking…");
  try {
    const res = await fetchJSON("/api/call/text_turn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, text }),
    });
    await handleTurnResponse(res, text);
  } catch (err) {
    setStatus(`Turn failed: ${err.message}`, true);
  } finally {
    setBusy(false);
  }
}

async function handleTurnResponse(res, typedText) {
  const userText = typedText ?? res.transcript;
  if (userText) {
    addBubble("user", userText);
  } else {
    addBubble("user", "(unintelligible)");
  }
  addBubble("caller", res.reply_text);
  await playAudio(res.audio_b64);
  if (res.ended) {
    finishCall();
  } else {
    setStatus("Your turn — click the mic or type a reply.");
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch (_) { /* not json */ }
    throw new Error(detail);
  }
  return res.json();
}

function playAudio(b64) {
  return new Promise((resolve) => {
    const audio = new Audio(`data:audio/wav;base64,${b64}`);
    setStatus("Caller is speaking…");
    audio.onended = resolve;
    audio.onerror = resolve;
    audio.play().catch(resolve);
  });
}

function addBubble(who, text) {
  const div = document.createElement("div");
  div.className = `bubble ${who}`;
  div.textContent = text;
  transcriptEl.appendChild(div);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("error", isError);
}

function setBusy(value) {
  busy = value;
  recordBtn.disabled = value || !sessionId;
  textSendBtn.disabled = value;
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

// ---------------------------------------------------------------------------

recordBtn.addEventListener("click", toggleRecording);
textSendBtn.addEventListener("click", sendTextTurn);
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendTextTurn();
});
endCallBtn.addEventListener("click", endCall);

init();
