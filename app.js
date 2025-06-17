// Fetch and display metrics
fetch('https://pneumonia-xray-cam.onrender.com/metrics')
  .then(res => res.json())
  .then(data => {
    document.getElementById('metrics').innerHTML = `
      <div><div class='font-bold'>Accuracy</div><div>${data.accuracy}%</div></div>
      <div><div class='font-bold'>Precision</div><div>${data.precision}%</div></div>
      <div><div class='font-bold'>Recall</div><div>${data.recall}%</div></div>
      <div><div class='font-bold'>F1 Score</div><div>${data.f1_score}%</div></div>
    `;
  });

const form = document.getElementById('upload-form');
const imageInput = document.getElementById('image-input');
const resultsDiv = document.getElementById('results');
const origImg = document.getElementById('original-img');
const camImg = document.getElementById('cam-img');
const labelSpan = document.getElementById('label');
const confidenceSpan = document.getElementById('confidence');

form.onsubmit = async (e) => {
  e.preventDefault();
  const file = imageInput.files[0];
  if (!file) return;
  const formData = new FormData();
  formData.append('file', file);
  labelSpan.textContent = 'Predicting...';
  confidenceSpan.textContent = '';
  resultsDiv.classList.remove('hidden');
  origImg.src = '';
  camImg.src = '';
  try {
    const res = await fetch('https://pneumonia-xray-cam.onrender.com/predict', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    origImg.src = 'data:image/png;base64,' + data.original_image;
    camImg.src = data.cam_image ? 'data:image/png;base64,' + data.cam_image : '';
    labelSpan.textContent = `Prediction: ${data.predicted_label}`;
    confidenceSpan.textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
  } catch (err) {
    labelSpan.textContent = 'Prediction failed.';
    confidenceSpan.textContent = '';
  }
}; 