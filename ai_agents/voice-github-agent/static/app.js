const recordBtn = document.getElementById("record-btn");
const statusLine = document.getElementById("status-line");
const errorBanner = document.getElementById("error-banner");
const resultSection = document.getElementById("result");
const transcriptText = document.getElementById("transcript-text");
const activityFeed = document.getElementById("activity-feed");
const summaryText = document.getElementById("summary-text");

let mediaRecorder = null;
let mediaStream = null;
let chunks = [];
let isRecording = false;

recordBtn.addEventListener("click", () => {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
});

async function startRecording() {
  hideError();

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    showError("Microphone access was denied or is unavailable.");
    return;
  }

  chunks = [];
  mediaRecorder = new MediaRecorder(mediaStream);
  mediaRecorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) chunks.push(event.data);
  };
  mediaRecorder.onstop = handleStop;
  mediaRecorder.start();

  isRecording = true;
  recordBtn.classList.add("is-recording");
  recordBtn.setAttribute("aria-pressed", "true");
  recordBtn.setAttribute("aria-label", "Stop recording");
  statusLine.textContent = "Recording — click to stop";
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
  }
  isRecording = false;
  recordBtn.classList.remove("is-recording");
  recordBtn.setAttribute("aria-pressed", "false");
  recordBtn.setAttribute("aria-label", "Start recording");
}

async function handleStop() {
  statusLine.textContent = "Encoding audio...";

  const recordedBlob = new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" });
  const arrayBuffer = await recordedBlob.arrayBuffer();

  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  const audioCtx = new AudioContextCtor();

  let audioBuffer;
  try {
    audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  } catch (err) {
    showError("Could not process that recording — try again.");
    statusLine.textContent = "Click to speak";
    return;
  } finally {
    audioCtx.close();
  }

  const wavBlob = encodeWav(audioBuffer);
  await sendToServer(wavBlob);
}

/** Encodes an AudioBuffer as a 16-bit PCM mono WAV Blob — AssemblyAI's Sync API needs WAV. */
function encodeWav(audioBuffer) {
  const samples = downmixToMono(audioBuffer);
  const sampleRate = audioBuffer.sampleRate;
  const bytesPerSample = 2;
  const dataSize = samples.length * bytesPerSample;

  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true); // byte rate
  view.setUint16(32, bytesPerSample, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

function downmixToMono(audioBuffer) {
  if (audioBuffer.numberOfChannels === 1) {
    return audioBuffer.getChannelData(0);
  }
  const left = audioBuffer.getChannelData(0);
  const right = audioBuffer.getChannelData(1);
  const out = new Float32Array(left.length);
  for (let i = 0; i < left.length; i++) {
    out[i] = (left[i] + right[i]) / 2;
  }
  return out;
}

function writeString(view, offset, str) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

async function sendToServer(wavBlob) {
  statusLine.textContent = "Thinking...";
  recordBtn.disabled = true;
  resultSection.hidden = true;

  const formData = new FormData();
  formData.append("audio", wavBlob, "instruction.wav");

  try {
    const response = await fetch("/api/run", { method: "POST", body: formData });
    const data = await response.json();

    if (!response.ok) {
      showError(data.error || "Something went wrong.");
      return;
    }

    renderResult(data);
  } catch (err) {
    showError("Could not reach the server.");
  } finally {
    statusLine.textContent = "Click to speak";
    recordBtn.disabled = false;
  }
}

function renderResult(data) {
  resultSection.hidden = false;

  // Transcript, with a briefly blinking caret to sell the "lands at the cursor" feel.
  transcriptText.textContent = data.transcript || "";
  const caret = document.createElement("span");
  caret.className = "caret";
  transcriptText.appendChild(caret);
  setTimeout(() => caret.remove(), 2200);

  activityFeed.innerHTML = "";
  const toolCalls = data.tool_calls || [];

  if (toolCalls.length === 0) {
    const li = document.createElement("li");
    li.className = "activity-item activity-item-empty";
    li.textContent = "No tools were needed for this one.";
    activityFeed.appendChild(li);
  } else {
    toolCalls.forEach((call, index) => {
      const li = document.createElement("li");
      li.className = "activity-item";
      li.style.animationDelay = `${index * 120}ms`;
      li.innerHTML = renderMarkdownInline(describeToolCall(call));
      activityFeed.appendChild(li);
    });
  }

  summaryText.innerHTML = renderMarkdownBlock(data.summary || "");
}

/**
 * The summary and tool-call descriptions come from the model, so before
 * handing them to marked we escape the characters that would otherwise let
 * raw HTML (a <script> tag, an onerror= attribute, ...) slip into innerHTML.
 * Markdown syntax itself (*, `, #, -, etc.) doesn't use < or >, so this
 * doesn't affect legitimate formatting — it only neutralizes literal HTML.
 */
function escapeHtml(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/** Full block-level markdown (paragraphs, numbered/bulleted lists, etc.) for the summary panel. */
function renderMarkdownBlock(text) {
  return marked.parse(escapeHtml(text));
}

/** Inline-only markdown (bold, code, links) for single-line activity feed entries. */
function renderMarkdownInline(text) {
  return marked.parseInline(escapeHtml(text));
}

/** Turns a raw tool call into a plain-language activity line for the feed. */
function describeToolCall(call) {
  const name = call.name;
  const args = call.args || {};
  const result = call.result || {};

  switch (name) {
    case "list_recent_commits":
      return `Checked the last ${args.count || 5} commit${(args.count || 5) === 1 ? "" : "s"}`;
    case "get_commit_diff":
      return `Looked at the diff for commit ${(args.sha || "").slice(0, 7)}`;
    case "list_open_issues":
      return "Checked the open issues";
    case "create_issue":
      return result.number ? `Opened issue #${result.number}` : `Opened a new issue: "${args.title || ""}"`;
    case "add_comment":
      return `Added a comment on issue #${args.issue_number}`;
    default:
      return `Ran ${name}`;
  }
}

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.hidden = false;
}

function hideError() {
  errorBanner.hidden = true;
  errorBanner.textContent = "";
}
