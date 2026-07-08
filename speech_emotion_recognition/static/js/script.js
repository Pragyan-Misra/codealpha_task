// ---------------------------------------------------------------------
// Speech Emotion Recognition — Frontend logic
// Handles: tab switching, file upload (drag/drop), mic recording,
// calling /predict, and rendering results.
// ---------------------------------------------------------------------

const EMOTION_EMOJI = {
  neutral: "😐", calm: "😌", happy: "😄", sad: "😢",
  angry: "😠", fearful: "😨", disgust: "🤢", surprised: "😲", boredom: "🥱",
};

let selectedFile = null;      // File object from upload OR recorded blob
let mediaRecorder = null;
let recordedChunks = [];
let recordTimerInterval = null;
let recordSeconds = 0;

const $ = (id) => document.getElementById(id);

// ---------------- Tabs ----------------
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
    btn.classList.add("active");
    $(`${btn.dataset.tab}-tab`).classList.add("active");
    resetSelection();
  });
});

// ---------------- File upload / drag & drop ----------------
const dropzone = $("dropzone");
const fileInput = $("fileInput");

dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) handleFileSelected(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length) handleFileSelected(e.target.files[0]);
});

$("clearFile").addEventListener("click", () => resetSelection());

function handleFileSelected(file) {
  selectedFile = file;
  $("fileName").textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  $("fileInfo").style.display = "flex";
  $("analyzeBtn").disabled = false;
}

// ---------------- Recording ----------------
const recordBtn = $("recordBtn");

recordBtn.addEventListener("click", async () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    stopRecording();
  } else {
    await startRecording();
  }
});

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
      const blob = new Blob(recordedChunks, { type: "audio/webm" });
      selectedFile = new File([blob], "recording.webm", { type: "audio/webm" });
      const audioUrl = URL.createObjectURL(blob);
      $("recordedAudio").src = audioUrl;
      $("recordedAudio").style.display = "block";
      $("analyzeBtn").disabled = false;
      stream.getTracks().forEach((t) => t.stop());
    };

    mediaRecorder.start();
    recordBtn.classList.add("recording");
    $("recordIcon").textContent = "⏹️";
    $("recordStatus").textContent = "Recording… tap to stop";

    recordSeconds = 0;
    updateTimer();
    recordTimerInterval = setInterval(updateTimer, 1000);
  } catch (err) {
    showError("Microphone access denied or unavailable: " + err.message);
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
  }
  recordBtn.classList.remove("recording");
  $("recordIcon").textContent = "🎤";
  $("recordStatus").textContent = "Recording complete";
  clearInterval(recordTimerInterval);
}

function updateTimer() {
  const mins = String(Math.floor(recordSeconds / 60)).padStart(2, "0");
  const secs = String(recordSeconds % 60).padStart(2, "0");
  $("recordTimer").textContent = `${mins}:${secs}`;
  recordSeconds++;
}

// ---------------- Analyze ----------------
$("analyzeBtn").addEventListener("click", async () => {
  if (!selectedFile) return;

  hideError();
  $("resultSection").style.display = "none";
  $("loadingSection").style.display = "block";
  $("analyzeBtn").disabled = true;

  const formData = new FormData();
  formData.append("audio", selectedFile);

  try {
    const res = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Something went wrong.");
    }
    renderResult(data);
  } catch (err) {
    showError(err.message);
  } finally {
    $("loadingSection").style.display = "none";
    $("analyzeBtn").disabled = false;
  }
});

function renderResult(data) {
  $("resultSection").style.display = "block";
  $("resultEmoji").textContent = data.emoji || EMOTION_EMOJI[data.emotion] || "🎙️";
  $("resultEmotion").textContent = data.emotion;
  $("resultConfidence").textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

  const container = $("scoresContainer");
  container.innerHTML = "";
  Object.entries(data.all_scores).forEach(([emotion, score]) => {
    const row = document.createElement("div");
    row.className = "score-row";
    row.innerHTML = `
      <span class="score-label">${emotion}</span>
      <div class="score-bar-bg"><div class="score-bar-fill" style="width:${score * 100}%"></div></div>
      <span class="score-pct">${(score * 100).toFixed(1)}%</span>
    `;
    container.appendChild(row);
  });

  $("resultSection").scrollIntoView({ behavior: "smooth", block: "center" });
}

function showError(msg) {
  const box = $("errorBox");
  box.textContent = "⚠️ " + msg;
  box.style.display = "block";
}
function hideError() {
  $("errorBox").style.display = "none";
}

function resetSelection() {
  selectedFile = null;
  $("fileInfo").style.display = "none";
  $("fileInput").value = "";
  $("recordedAudio").style.display = "none";
  $("recordStatus").textContent = "Tap to start recording";
  $("recordTimer").textContent = "00:00";
  $("analyzeBtn").disabled = true;
  $("resultSection").style.display = "none";
  hideError();
}
