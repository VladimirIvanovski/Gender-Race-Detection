import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


from flask import Flask, render_template, request, jsonify
import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import base64
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import os
import whisper
import warnings
from openai import OpenAI
from pydantic import BaseModel
import subprocess
from io import BytesIO
from pydub import AudioSegment
import numpy as np

app = Flask(__name__)
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Load the model at startup
print("Launching the model...")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
transcription_model = whisper.load_model("tiny")
print("Model launched.")
load_dotenv()

class Analyzing(BaseModel):
    summary: str
    hashtags: str
    niches: str

# Access environment variables
PROXIES = os.getenv("PROXY_URL").split(",") if os.getenv("PROXY_URL") else []
INSTAGRAM_API = os.getenv("INSTAGRAM_API")
open_ai_key = os.getenv("OPENAI_KEY")
client = OpenAI(api_key=open_ai_key)

def extract_image_urls(username):

    max_retries = 3
    counter = 0
    which_proxy = 0
    while counter < max_retries:
        try:
            if which_proxy == 0:
                which_proxy = 1
            else:
                which_proxy = 0
            proxies = {'https': PROXIES[which_proxy]}
            image_links = []
            video_links = []
            headers = {
                "accept": "*/*",
                "accept-language": "en-US,en;q=0.9",
                "x-ig-app-id": "936619743392459",
            }
            response = requests.get(f"{INSTAGRAM_API}/?username={username}", proxies=proxies,headers=headers)
            if response.status_code == 200:
                data = response.json()
                allData = response.json()['data']['user']
                allPosts = allData['edge_owner_to_timeline_media']['edges']
                is_priv = data.get('data', {}).get('user',{}).get('is_private')
                if is_priv:
                    return [], True, []

                reel_links = []
                media = data["data"]["user"].get("edge_owner_to_timeline_media", {}).get("edges", [])
                for item in media:
                    node = item.get("node", {})
                    if node.get("__typename") == "GraphVideo":  # Videos, including reels
                        shortcode = node.get("shortcode")
                        if shortcode:
                            reel_links.append(f"https://www.instagram.com/reel/{shortcode}/")

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

                return image_links[:9], False, reel_links
        except Exception as e:
            print("error",e)
        counter+=1
    return None, None, None


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


def extract_audio(reel_url):
    """Extract audio from a reel URL."""
    try:
        yt_process = subprocess.Popen(
            ["yt-dlp", "-f", "bestaudio", "-o", "-", reel_url],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        ff_process = subprocess.Popen(
            [
                "ffmpeg", "-i", "pipe:0",
                "-f", "wav",
                "-ar", "16000",  # Set sample rate to 16kHz
                "-ac", "1",      # Mono
                "-acodec", "pcm_s16le",
                "-"
            ],
            stdin=yt_process.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        yt_process.stdout.close()
        wav_data, ff_err = ff_process.communicate()

        if ff_process.returncode != 0 or len(wav_data) == 0:
            # print(f"FFmpeg error for URL {reel_url}: {ff_err.decode('utf-8', errors='ignore')}")
            return None  # Return None if there's an error

        return wav_data
    except Exception as e:
        # print(f"Error extracting audio for URL {reel_url}: {e}")
        return None


def transcribe_audio(wav_data):
    """Transcribe audio data using Whisper."""
    try:
        audio_segment = AudioSegment.from_file(BytesIO(wav_data), format="wav")
        if len(audio_segment) == 0:
            # print("Error: Extracted audio is empty.")
            return None

        samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0
        # print(f"Audio length: {len(audio_segment)} ms, Samples shape: {samples.shape}")

        result = transcription_model.transcribe(audio=samples)
        return result["text"]
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None


def process_reel(reel_url):
    """Process a single reel URL: extract audio and transcribe."""
    wav_data = extract_audio(reel_url)
    if not wav_data:
        # print(f"Skipping reel {reel_url}: No audio data extracted.")
        return None

    transcription = transcribe_audio(wav_data)
    if not transcription:
        print(f"Skipping reel {reel_url}: Transcription failed.")
        return None

    return transcription


def process_reels_with_limit(reel_urls, required_transcriptions=2):
    """
    Process reels one by one until at least 'required_transcriptions' transcriptions are obtained.
    Combine transcriptions with newlines and return the result.
    """
    transcriptions = []
    for reel_url in reel_urls:
        print(f"Processing reel: {reel_url}")
        transcription = process_reel(reel_url)
        if transcription:
            transcriptions.append(transcription)
            print(f"Transcription added. Total so far: {len(transcriptions)}")

        # Stop processing if the required number of transcriptions is reached
        if len(transcriptions) >= required_transcriptions:
            print(f"Required transcriptions reached ({required_transcriptions}). Stopping.")
            break

    # Combine all transcriptions with newlines
    combined_transcriptions = "\n".join(transcriptions)
    return combined_transcriptions


def analyze_influencer_content(transcription):
    prompt = (
        "Analyze the following transcription to provide:\n"
        "1. A concise summary and simple to understand (maximum 25 words).\n"
        "2. Five relevant hashtags.\n"
        "3. Three relevant niches.\n\n"
        f"Transcription: {transcription}"
    )

    if len(str(transcription).split(" ")) < 50:
        return None,None,None

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system","content": "You are an expert in analyzing text from Influencers or Creators"},
            {"role": "user", "content": prompt}
        ],
        response_format=Analyzing,
        max_tokens=900  # Adjust this value based on your needs
    )

    response_message = completion.choices[0].message

    # Check if the model refused to respond
    if response_message.refusal:
        print(response_message.refusal)
        return None, None, None
    else:
        parsed_response = response_message.parsed
        summary = parsed_response.summary
        hashtags = parsed_response.hashtags  # Convert list to comma-separated string
        niches = parsed_response.niches     # Convert list to comma-separated string
        return summary, hashtags, niches



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detect_gender', methods=['POST'])
def detect_gender_route():
    return detect_attribute_route('gender')

@app.route('/detect_race', methods=['POST'])
def detect_race_route():
    return detect_attribute_route('race')

@app.route('/detect_analyze_video', methods=['POST'])
def analyze_video_content():
    data = request.get_json()
    username = str(data.get('username')).replace("@","").lower()
    if not username:
        return jsonify({'error': 'No username provided'}), 400

    steps = []
    steps.append("Fetching images...")

    image_urls, is_private, reels = extract_image_urls(username)

    if is_private:
        return jsonify({"error": "Profile is private or unavailable"})

    if not image_urls:
        return jsonify({'error': 'No images found for this user'}), 404

    results = process_reels_with_limit(reels)
    summary, hashtags, niches = analyze_influencer_content(results)
    collage = create_collage(image_urls, grid_size=(3, 3), image_size=(500, 500), spacing=15)

    # Convert collage image to base64
    buffered = BytesIO()
    collage.save(buffered, format="JPEG")
    collage_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({
        'username': username,
        'summary': summary,
        'hashtags': hashtags,
        'niches': niches,
        'collage_image': collage_base64
    })

def detect_attribute_route(attribute):
    data = request.get_json()
    username = str(data.get('username')).replace("@","").lower()
    if not username:
        return jsonify({'error': 'No username provided'}), 400

    steps = []
    steps.append("Fetching images...")

    image_urls, is_private, reels = extract_image_urls(username)

    if is_private:
        return jsonify({"error": "Profile is private or unavailable"})

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
