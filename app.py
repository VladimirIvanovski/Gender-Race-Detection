from flask import Flask, render_template, request, jsonify
import requests
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import base64
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Load the model at startup
print("Launching the model...")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
print("Model launched.")
load_dotenv()

# Access environment variables
PROXY_URL = os.getenv("PROXY_URL")
INSTAGRAM_API = os.getenv("INSTAGRAM_API")

def extract_image_urls(username):

    proxies = {'https': PROXY_URL}
    max_retries = 2
    counter = 0
    while counter < max_retries:
        try:
            image_links = []
            video_links = []
            headers = {
                "accept": "*/*",
                "accept-language": "en-US,en;q=0.9",
                "x-ig-app-id": "936619743392459",
            }
            response = requests.get(f"{INSTAGRAM_API}/?username={username}", proxies=proxies,headers=headers)

            if response.status_code == 200:

                allData = response.json()['data']['user']
                allPosts = allData['edge_owner_to_timeline_media']['edges']

                for i in range(0, len(allPosts)):
                    if allPosts[i]['node']['is_video']:
                        image_links.append(allPosts[i]['node']['display_url'])
                        video_links.append(allPosts[i]['node']['video_url'])
                    else:
                        image_links.append(allPosts[i]['node']['display_url'])

                profile_photo_url = allData.get('profile_pic_url_hd', None)

                # Print or store the profile photo URL
                if profile_photo_url:
                    image_links[0] = profile_photo_url

                return image_links
        except Exception as e:
            print("error",e)
        counter+=1
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

def create_collage(image_urls, grid_size=(3, 3), image_size=(300, 300), spacing=10):
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
            "Images of Female people"
        ]
    elif attribute == 'race':
        labels = [
            "White race",
            "Indian race",
            "Asian race",
            "Black race"
        ]
    else:
        return jsonify({'error': 'Invalid attribute'}), 400

    result = classify_image(collage, labels)
    steps.append("Processing complete.")
    best_label = result['best_label']
    best_probability = result['best_probability']
    detected_attribute = best_label.replace('Images of', '').strip()
    if best_probability < 0.65:
        detected_attribute = "Not sure, guessing " + detected_attribute

    buffered = BytesIO()
    collage.save(buffered, format="JPEG")
    collage_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({
        'username': username,
        'result': detected_attribute,
        'probability': best_probability,
        'steps': steps,
        'collage_image': collage_base64
    })

if __name__ == '__main__':
    app.run(debug=True)
