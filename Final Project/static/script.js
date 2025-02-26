// Select elements
const inputVideo = document.getElementById('inputVideo');
const processButton = document.getElementById('processButton');
const outputVideo = document.getElementById('outputVideo');
const videoPreview = document.getElementById('videoPreview');

let inputVideoFile = null;

// When a video file is selected
inputVideo.addEventListener('change', function () {
  inputVideoFile = this.files[0];
  const fileURL = URL.createObjectURL(inputVideoFile);
  videoPreview.src = fileURL;
  videoPreview.style.display = 'block';
});

// When the Process button is clicked
processButton.addEventListener('click', function () {
  if (!inputVideoFile) {
    alert('Please select a video file first.');
    return;
  }

  const formData = new FormData();
  formData.append('video', inputVideoFile);

  // Send the video to the backend using a relative URL
  fetch('/process', {
    method: 'POST',
    body: formData,
  })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        alert(`Error: ${data.error}`);
      } else {
        // Display the processed video
        const processedVideoUrl = `/uploads/${data.processed_video}?t=${new Date().getTime()}`; // Append timestamp to prevent caching
        console.log(`Processed video URL: ${processedVideoUrl}`);
        outputVideo.src = processedVideoUrl;
        outputVideo.style.display = 'block';

        // Display the prediction result
        document.getElementById('predictionLabel').textContent = data.predicted_label;
        document.getElementById('predictionProbability').textContent = `${data.prediction_probability.toFixed(1)}%`;
        document.getElementById('resultSection').style.display = 'block';
      }
    })
    .catch(error => {
      console.error('Error:', error);
    });
});
