from flask import Flask, render_template, request, jsonify
import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from transformers import CLIPProcessor, CLIPModel
import threading
import time
import base64
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Load the model at startup
print("Launching the model...")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
print("Model launched.")

def extract_image_urls(username):
    page_url = "https://flufi.me"
    try:
        response = requests.get(f"https://flufi.me/profile/{username}")
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tags = soup.find_all('img', src=True)
        image_urls = []
        for img in img_tags:
            img_url = img.get('src')
            if img_url and "_next/image" in img_url:
                full_url = urljoin(page_url, img_url)
                image_urls.append(full_url)
        return image_urls[2:]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the webpage: {e}")
        return []

def fetch_image(url):
    try:
        response = requests.get(url,timeout=5)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the image: {e}")
        return None

def fetch_images_concurrently(image_urls):
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(fetch_image, image_urls))
    return images

def create_collage(image_urls, grid_size=(3, 3), image_size=(500, 500), spacing=10):
    rows, cols = grid_size
    img_width, img_height = image_size
    collage_width = cols * img_width + (cols + 1) * spacing
    collage_height = rows * img_height + (rows + 1) * spacing
    collage = Image.new('RGB', (collage_width, collage_height), color=(255, 255, 255))

    # Fetch images concurrently
    images = fetch_images_concurrently(image_urls)
    images = [img.resize(image_size) for img in images if img is not None]

    for idx, img in enumerate(images):
        row, col = divmod(idx, cols)
        x = col * img_width + (col + 1) * spacing
        y = row * img_height + (row + 1) * spacing
        collage.paste(img, (x, y))
    return collage

def classify_image(image, labels):
    inputs = processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True
    )
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    best_match_idx = probs.argmax().item()
    best_label = labels[best_match_idx]
    best_prob = probs[0][best_match_idx].item()
    return {
        "best_label": best_label,
        "best_probability": best_prob,
        "all_probabilities": {label: prob.item() for label, prob in zip(labels, probs[0])}
    }

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detect_gender', methods=['POST'])
def detect_gender_route():
    return detect_attribute_route('gender')

@app.route('/detect_race', methods=['POST'])
def detect_race_route():
    return detect_attribute_route('race')

def detect_attribute_route(attribute):
    data = request.get_json()
    username = str(data.get('username')).replace("@","").lower()
    if not username:
        return jsonify({'error': 'No username provided'}), 400

    steps = []
    steps.append("Fetching images...")
    image_urls = extract_image_urls(username)[:9]
    if not image_urls:
        return jsonify({'error': 'No images found for this user'}), 404

    steps.append("Creating collage...")
    collage = create_collage(image_urls, grid_size=(3, 3), image_size=(500, 500), spacing=15)

    steps.append("Analyzing images...")
    if attribute == 'gender':
        labels = [
            "Images of Male people",
            "Images of Female people",
            "Images of objects"
        ]
    elif attribute == 'race':
        labels = [
            "White race",
            "Indian race",
            "Asian race",
            "Black race",
            "Objects"
        ]
    else:
        return jsonify({'error': 'Invalid attribute'}), 400

    result = classify_image(collage, labels)
    steps.append("Processing complete.")
    best_label = result['best_label']
    best_probability = result['best_probability']
    detected_attribute = best_label.replace('Images of', '').strip()

    # Encode the collage image to base64
    buffered = BytesIO()
    collage.save(buffered, format="JPEG")
    collage_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # if best_probability < 0.75:
    #     detected_attribute = "Objects"

    return jsonify({
        'username': username,
        'result': detected_attribute,
        'probability': best_probability,
        'steps': steps,
        'collage_image': collage_base64
    })

if __name__ == '__main__':
    app.run(debug=True)
