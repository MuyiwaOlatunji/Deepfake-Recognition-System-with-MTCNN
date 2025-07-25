document.getElementById('uploadForm').addEventListener('submit', function() {
    // Show loading spinner
    document.getElementById('loading').classList.remove('hidden');
    // Hide result until new one loads
    document.getElementById('result').classList.add('hidden');
});