document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('username-form');
    const usernameInput = document.getElementById('username');
    const loadingSteps = document.getElementById('loading-steps');
    const stepText = document.getElementById('step');
    const resultDiv = document.getElementById('result');
    const resultUsername = document.getElementById('result-username');
    const resultType = document.getElementById('result-type');
    const resultProbability = document.getElementById('result-probability');
    const collageImage = document.getElementById('collage-image');
    const detectGenderButton = document.getElementById('detect-gender-btn');
    const detectRaceButton = document.getElementById('detect-race-btn');

    let detectionType = 'gender'; // Default detection type

    detectGenderButton.addEventListener('click', function(e) {
        detectionType = 'gender';
    });

    detectRaceButton.addEventListener('click', function(e) {
        detectionType = 'race';
        // Manually submit the form
        form.dispatchEvent(new Event('submit'));
    });

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        const username = usernameInput.value.trim();
        if (username === '') {
            alert('Please enter a username');
            return;
        }

        // Disable inputs during processing
        usernameInput.disabled = true;
        detectGenderButton.disabled = true;
        detectRaceButton.disabled = true;

        // Hide previous results and show loading steps
        resultDiv.style.display = 'none';
        loadingSteps.style.display = 'block';

        const steps = [
            "Fetching images...",
            "Creating collage...",
            "Analyzing images..."
        ];
        let currentStep = 0;
        stepText.textContent = steps[currentStep];

        const interval = setInterval(function() {
            currentStep++;
            if (currentStep < steps.length) {
                stepText.textContent = steps[currentStep];
            } else {
                // Keep "Analyzing images..." displayed
                currentStep = steps.length - 1;
            }
        }, 2000); // Update every 2 seconds

        let endpoint = detectionType === 'gender' ? '/detect_gender' : '/detect_race';

        fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 'username': username })
        })
        .then(response => response.json())
        .then(data => {
            clearInterval(interval);
            if (data.error) {
                stepText.textContent = data.error;
                // Re-enable inputs
                usernameInput.disabled = false;
                detectGenderButton.disabled = false;
                detectRaceButton.disabled = false;
                return;
            }

            // Display the result
            loadingSteps.style.display = 'none';
            resultDiv.style.display = 'block';
            resultUsername.textContent = data.username;
            resultType.textContent = detectionType === 'gender' ? `Gender: ${data.result}` : `Race: ${data.result}`;
            resultProbability.textContent = (data.probability * 100).toFixed(2) + '%';

            // Set the collage image source using data URL
            collageImage.src = 'data:image/jpeg;base64,' + data.collage_image;

            // Re-enable inputs
            usernameInput.disabled = false;
            // usernameInput.value = '';
            detectGenderButton.disabled = false;
            detectRaceButton.disabled = false;
        })
        .catch(error => {
            clearInterval(interval);
            console.error('Error:', error);
            stepText.textContent = 'An error occurred during processing.';
            // Re-enable inputs
            usernameInput.disabled = false;
            detectGenderButton.disabled = false;
            detectRaceButton.disabled = false;
        });
    });
});
