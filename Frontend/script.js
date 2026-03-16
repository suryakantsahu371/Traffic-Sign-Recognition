// ================= DOM =================
const imageInput = document.getElementById("image-input");
const chooseBtn = document.getElementById("choose-image-btn");
const openCameraBtn = document.getElementById("open-camera-btn");
const captureBtn = document.getElementById("capture-btn");

const previewImage = document.getElementById("preview-image");
const noImageText = document.getElementById("no-image-text");
const video = document.getElementById("camera-stream");

const predictBtn = document.getElementById("predict-btn");
const predictedClass = document.getElementById("predicted-class");

const probCnnBar = document.getElementById("prob-cnn");
const probRnnBar = document.getElementById("prob-rnn");
const probLstmBar = document.getElementById("prob-lstm");

const probCnnText = document.getElementById("prob-cnn-text");
const probRnnText = document.getElementById("prob-rnn-text");
const probLstmText = document.getElementById("prob-lstm-text");

// ================= STATE =================
let stream = null;
let usingCamera = false;
let intervalId = null;

// ================= IMAGE UPLOAD =================
chooseBtn.onclick = () => {
  stopCamera();
  usingCamera = false;
  stopRealtime();
  imageInput.click();
};

imageInput.onchange = e => {
  const file = e.target.files[0];
  if (!file) return;

  usingCamera = false;
  stopCamera();
  stopRealtime();

  const reader = new FileReader();
  reader.onload = ev => {
    previewImage.src = ev.target.result;
    previewImage.style.display = "block";
    noImageText.style.display = "none";
    video.style.display = "none";
  };
  reader.readAsDataURL(file);
};

// ================= START CAMERA =================
openCameraBtn.onclick = async () => {
  try {
    usingCamera = true;

    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;

    video.style.display = "block";
    previewImage.style.display = "none";
    noImageText.style.display = "none";

    captureBtn.style.display = "inline-block";

  } catch (err) {
    alert("Camera permission denied");
    console.error(err);
  }
};

// ================= STOP CAMERA =================
function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  video.style.display = "none";
  captureBtn.style.display = "none";
}

// ================= CAPTURE FRAME =================
captureBtn.onclick = () => {
  if (!usingCamera) return;

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  ctx.drawImage(video, 0, 0);

  const dataURL = canvas.toDataURL("image/jpeg");

  previewImage.src = dataURL;
  previewImage.style.display = "block";
  video.style.display = "none";

  stopCamera();
  usingCamera = false;
};

// ================= FRAME SENDER =================
async function sendFrameToBackend() {
  if (!usingCamera) return;

  try {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0);

    const base64 = canvas.toDataURL("image/jpeg").split(",")[1];

    const res = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ image: base64 })
    });

    const data = await res.json();

    predictedClass.textContent = data.cnn.class;

    updateBars(
      data.cnn.confidence,
      data.rnn.confidence,
      data.lstm.confidence
    );

  } catch (err) {
    console.error("Frame error:", err);
  }
}

// ================= START REALTIME =================
function startRealtime() {
  stopRealtime(); // avoid duplicate loops

  intervalId = setInterval(() => {
    sendFrameToBackend();
  }, 500); // 🔥 every 0.5 sec
}

// ================= STOP REALTIME =================
function stopRealtime() {
  if (intervalId) {
    clearInterval(intervalId);
    intervalId = null;
  }
}

// ================= PREDICT BUTTON =================
predictBtn.onclick = async () => {
  try {
    if (usingCamera) {
      // 🔥 START CONTINUOUS STREAM
      startRealtime();
      return;
    }

    // ================= IMAGE PREDICT =================
    if (!imageInput.files[0]) {
      alert("Please upload image or use camera");
      return;
    }

    const formData = new FormData();
    formData.append("image", imageInput.files[0]);

    const res = await fetch("http://localhost:5000/predict", {
      method: "POST",
      body: formData
    });

    const data = await res.json();

    predictedClass.textContent = data.cnn.class;

    updateBars(
      data.cnn.confidence,
      data.rnn.confidence,
      data.lstm.confidence
    );

  } catch (err) {
    alert("Prediction failed");
    console.error(err);
  }
};

// ================= UI UPDATE =================
function updateBars(cnn, rnn, lstm) {
  probCnnBar.style.width = cnn + "%";
  probRnnBar.style.width = rnn + "%";
  probLstmBar.style.width = lstm + "%";

  probCnnText.textContent = cnn + "%";
  probRnnText.textContent = rnn + "%";
  probLstmText.textContent = lstm + "%";
}