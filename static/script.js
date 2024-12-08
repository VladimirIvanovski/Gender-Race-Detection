document.addEventListener('DOMContentLoaded', function () {
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
    const analyzeVideoButton = document.getElementById('analyze-video-btn');

    let detectionType = 'gender'; // Default detection type

    detectGenderButton.addEventListener('click', () => {
        detectionType = 'gender';
    });

    detectRaceButton.addEventListener('click', () => {
        detectionType = 'race';
        form.dispatchEvent(new Event('submit'));
    });

    analyzeVideoButton.addEventListener('click', () => {
        alert("Analyze Video Content feature coming soon!");
    });

    form.addEventListener('submit', function (e) {
        e.preventDefault();
        const username = usernameInput.value.trim();
        if (!username) {
            alert('Please enter a username');
            return;
        }

        usernameInput.disabled = true;
        detectGenderButton.disabled = true;
        detectRaceButton.disabled = true;

        resultDiv.style.display = 'none';
        loadingSteps.style.display = 'block';

        const steps = [
            "Fetching data...",
            "Analyzing images...",
            "Finalizing results...",
        ];

        let currentStep = 0;
        stepText.textContent = steps[currentStep];

        const interval = setInterval(() => {
            currentStep++;
            if (currentStep < steps.length) {
                stepText.textContent = steps[currentStep];
            }
        }, 2000);

        fetch(`/detect_${detectionType}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username }),
        })
            .then((response) => response.json())
            .then((data) => {
                clearInterval(interval);
                loadingSteps.style.display = 'none';
                resultDiv.style.display = 'block';
                resultUsername.textContent = data.username || 'N/A';
                resultType.textContent = `${detectionType.toUpperCase()}: ${data.result || 'N/A'}`;
                resultProbability.textContent = `${(data.probability * 100).toFixed(2) || 'N/A'}%`;
                collageImage.src = data.collage_image ? `data:image/jpeg;base64,${data.collage_image}` : '';
            })
            .catch((err) => {
                clearInterval(interval);
                stepText.textContent = 'An error occurred.';
            })
            .finally(() => {
                usernameInput.disabled = false;
                detectGenderButton.disabled = false;
                detectRaceButton.disabled = false;
            });
    });
});
