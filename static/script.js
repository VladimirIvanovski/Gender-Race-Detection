document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('username-form');
    const usernameInput = document.getElementById('username');
    const loadingSteps = document.getElementById('loading-steps');
    const stepText = document.getElementById('step');
    const resultDiv = document.getElementById('result');
    const resultUsername = document.getElementById('result-username');
    const resultType = document.getElementById('result-type');
    const resultProbabilityContainer = document.getElementById('result-probability-container');
    const resultProbability = document.getElementById('result-probability');
    const collageImage = document.getElementById('collage-image');
    const videoCollageImage = document.getElementById('video-collage-image');
    const detectGenderButton = document.getElementById('detect-gender-btn');
    const detectRaceButton = document.getElementById('detect-race-btn');
    const analyzeVideoButton = document.getElementById('analyze-video-btn');
    const videoAnalysisDiv = document.getElementById('video-analysis');
    const videoSummary = document.getElementById('video-summary');
    const videoHashtags = document.getElementById('video-hashtags');
    const videoNiches = document.getElementById('video-niches');

    let detectionType = ''; // To store the type of detection

    // Event Listeners for Buttons
    detectGenderButton.addEventListener('click', () => {
        detectionType = 'gender';
        submitForm();
    });

    detectRaceButton.addEventListener('click', () => {
        detectionType = 'race';
        submitForm();
    });

    analyzeVideoButton.addEventListener('click', () => {
        detectionType = 'analyze_video';
        submitForm();
    });

    // Function to Handle Form Submission
    function submitForm() {
        const username = usernameInput.value.trim();
        if (!username) {
            alert('Please enter a username');
            return;
        }

        // Disable buttons and input to prevent multiple submissions
        usernameInput.disabled = true;
        detectGenderButton.disabled = true;
        detectRaceButton.disabled = true;
        analyzeVideoButton.disabled = true;

        // Reset previous results
        resultDiv.style.display = 'none';
        loadingSteps.style.display = 'block';
        stepText.textContent = 'Starting analysis...';

        // Define steps for loading animation
        const steps = [
            "Fetching data...",
            "Analyzing content...",
            "Processing results...",
            "Finalizing..."
        ];

        let currentStep = 0;
        stepText.textContent = steps[currentStep];

        const interval = setInterval(() => {
            currentStep++;
            if (currentStep < steps.length) {
                stepText.textContent = steps[currentStep];
            } else {
                clearInterval(interval);
            }
        }, 2000);

        // Determine the endpoint based on detection type
        let endpoint = '';
        if (detectionType === 'gender') {
            endpoint = '/detect_gender';
        } else if (detectionType === 'race') {
            endpoint = '/detect_race';
        } else if (detectionType === 'analyze_video') {
            endpoint = '/detect_analyze_video';
        } else {
            alert('Invalid detection type.');
            return;
        }

        // Make the POST request
        fetch(endpoint, {
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

            if (detectionType === 'analyze_video') {
                // Hide attribute results
                document.getElementById('attribute-result').style.display = 'none';
                // Show video analysis
                videoAnalysisDiv.style.display = 'block';
                videoSummary.textContent = data.summary || 'N/A';
                videoHashtags.textContent = data.hashtags || 'N/A';
                videoNiches.textContent = data.niches || 'N/A';
                // Set video collage image
                if (data.collage_image) {
                    videoCollageImage.src = `data:image/jpeg;base64,${data.collage_image}`;
                    videoCollageImage.style.display = 'block';
                } else {
                    videoCollageImage.style.display = 'none';
                }
                // Hide regular collage image
                collageImage.style.display = 'none';
            } else {
                // Show attribute results
                document.getElementById('attribute-result').style.display = 'block';
                // Hide video analysis
                videoAnalysisDiv.style.display = 'none';
                // Set detection type and probability
                const upperType = detectionType.charAt(0).toUpperCase() + detectionType.slice(1);
                resultType.textContent = `${upperType}: ${data.result || 'N/A'}`;
                if (data.probability !== undefined && data.probability !== null) {
                    resultProbability.textContent = `${(data.probability * 100).toFixed(2)}%`;
                    resultProbabilityContainer.style.display = 'block';
                } else {
                    resultProbabilityContainer.style.display = 'none';
                }
                // Set regular collage image
                if (data.collage_image) {
                    collageImage.src = `data:image/jpeg;base64,${data.collage_image}`;
                    collageImage.style.display = 'block';
                } else {
                    collageImage.style.display = 'none';
                }
                // Hide video collage image
                videoCollageImage.style.display = 'none';
            }
        })
        .catch((err) => {
            console.error(err);
            clearInterval(interval);
            loadingSteps.style.display = 'none';
            alert('An error occurred during the analysis. Please try again.');
        })
        .finally(() => {
            // Re-enable buttons and input
            usernameInput.disabled = false;
            detectGenderButton.disabled = false;
            detectRaceButton.disabled = false;
            analyzeVideoButton.disabled = false;
        });
    }
});
