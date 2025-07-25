document.getElementById('upload-form').addEventListener('submit', function(e) {
    const fileInput = document.getElementById('file');
    const submitButton = document.querySelector('button[type="submit"]');
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');

    // Validate file input
    if (!fileInput.files || !fileInput.files[0]) {
        e.preventDefault();
        alert('Please select a video file.');
        return;
    }

    const file = fileInput.files[0];
    const validTypes = ['video/mp4', 'video/avi', 'video/quicktime'];
    if (!validTypes.includes(file.type)) {
        e.preventDefault();
        alert('Please upload a valid video file (MP4, AVI, MOV).');
        return;
    }

    // Show loading spinner and disable button
    loading.classList.remove('hidden');
    result.classList.add('hidden');
    submitButton.disabled = true;
    submitButton.textContent = 'Processing...';

    // Update loading message with estimated time
    setTimeout(() => {
        const loadingText = loading.querySelector('p');
        if (loadingText) {
            loadingText.textContent = 'Analyzing video frames...';
        }
    }, 2000);
});

// Reset form on page load
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('upload-form');
    const submitButton = document.querySelector('button[type="submit"]');
    form.reset();
    submitButton.disabled = false;
    submitButton.textContent = 'Upload and Analyze';
});