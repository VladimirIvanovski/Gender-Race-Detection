import asyncio
import os
import time
from typing import Optional, List, Any

import torch
import subprocess
import threading
import warnings
import base64
import requests
import io
import numpy as np
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from PIL import Image
from pydantic import BaseModel
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI
from faster_whisper import WhisperModel  # Ensure this is installed: pip install faster-whisper
import soundfile as sf

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ----------------------- Singleton Classes -----------------------

class CLIPModelSingleton:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    print("Loading CLIPModel...")
                    cls._instance = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
                    print("CLIPModel loaded.")
        return cls._instance

class CLIPProcessorSingleton:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    print("Loading CLIPProcessor...")
                    cls._instance = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
                    print("CLIPProcessor loaded.")
        return cls._instance

class FasterWhisperSingleton:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, model_size="tiny", device="cpu", compute_type="int8"):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    print(f"Loading Faster-Whisper model '{model_size}' on device '{device}' with compute type '{compute_type}'...")
                    cls._instance = WhisperModel(model_size, device=device, compute_type=compute_type)
                    print("Faster-Whisper model loaded.")
        return cls._instance

class OpenAIClientSingleton:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, api_key):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    print("Initializing OpenAI client...")
                    cls._instance = OpenAI(api_key=api_key)
                    print("OpenAI client initialized.")
        return cls._instance

# ----------------------- Flask Application -----------------------

app = Flask(__name__)
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Load environment variables
load_dotenv()

# Initialize Singleton models
clip_model = None
clip_processor = None

trans_model = FasterWhisperSingleton.get_instance(model_size="tiny", device="cpu", compute_type="int8")
# Access environment variables
PROXIES = os.getenv("PROXY_URL").split(",") if os.getenv("PROXY_URL") else []
INSTAGRAM_API = os.getenv("INSTAGRAM_API")
open_ai_key = os.getenv("OPENAI_KEY")

if not open_ai_key:
    raise ValueError("OPENAI_KEY environment variable is not set.")

client = OpenAIClientSingleton.get_instance(api_key=open_ai_key)

class Analyzing(BaseModel):
    summary: str
    hashtags: str
    niches: str

# ----------------------- Helper Functions -----------------------

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
            response = requests.get(f"{INSTAGRAM_API}/?username={username}", proxies=proxies, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                allData = data['data']['user']
                allPosts = allData['edge_owner_to_timeline_media']['edges']
                is_priv = allData.get('is_private', False)
                if is_priv:
                    return [], True, []

                reel_links = []
                media = allPosts
                for item in media:
                    node = item.get("node", {})
                    if node.get("__typename") == "GraphVideo":  # Videos, including reels
                        shortcode = node.get("shortcode")
                        if shortcode:
                            reel_links.append(f"https://www.instagram.com/reel/{shortcode}/")

                for post in allPosts:
                    node = post.get('node', {})
                    if node.get('is_video'):
                        image_links.append(node.get('display_url'))
                        video_links.append(node.get('video_url'))
                    else:
                        image_links.append(node.get('display_url'))

                profile_photo_url = allData.get('profile_pic_url_hd', None)

                # Replace the first image with the profile photo if available
                if profile_photo_url and image_links:
                    image_links[0] = profile_photo_url

                return image_links[:9], False, reel_links[:8]
        except Exception as e:
            print(f"Error fetching images (Attempt {counter+1}/{max_retries}): {e}")
        counter += 1
    return None, None, None

def fetch_image(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the image from {url}: {e}")
        return None

def fetch_images_concurrently(image_urls):
    with ThreadPoolExecutor(max_workers=8) as executor:
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
        if idx >= rows * cols:
            break  # Limit to grid size
        row, col = divmod(idx, cols)
        x = col * img_width + (col + 1) * spacing
        y = row * img_height + (row + 1) * spacing
        collage.paste(img, (x, y))
    return collage

def classify_image(image, labels):
    inputs = clip_processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        outputs = clip_model(**inputs)
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

def extract_audio_sync(reel_url: str, proxy: str = None) -> Optional[bytes]:
    """
    Synchronously extract audio from a reel URL using yt-dlp and FFmpeg.

    Args:
        reel_url (str): The URL of the Instagram reel.
        proxy (Optional[str]): Proxy URL if required.

    Returns:
        Optional[bytes]: The extracted WAV audio data or None if extraction failed.
    """
    try:
        yt_dlp_command = ["yt-dlp", "-f", "bestaudio", "-o", "-"]
        if proxy:
            yt_dlp_command.extend(["--proxy", proxy])
        yt_dlp_command.append(reel_url)

        ffmpeg_command = [
            "ffmpeg",
            "-i", "pipe:0",
            "-f", "wav",
            "-ar", "16000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-loglevel", "quiet",
            "-"
        ]

        # Start yt-dlp process
        yt_process = subprocess.Popen(
            yt_dlp_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Start FFmpeg process
        ff_process = subprocess.Popen(
            ffmpeg_command,
            stdin=yt_process.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Close yt_process stdout in the parent to allow FFmpeg to receive EOF
        yt_process.stdout.close()

        # Get FFmpeg output
        wav_data, ff_err = ff_process.communicate()

        # Get yt-dlp stderr
        yt_stdout, yt_err = yt_process.communicate()

        # Check yt-dlp's return code
        if yt_process.returncode != 0:
            print(f"yt-dlp failed for {reel_url}. Error: {yt_err.decode('utf-8').strip()}")
            return None

        # Check FFmpeg's return code and wav_data
        if ff_process.returncode != 0 or not wav_data:
            print(f"FFmpeg failed for {reel_url}. FFmpeg stderr: {ff_err.decode('utf-8').strip()}")
            return None

        return wav_data

    except Exception as e:
        print(f"Unexpected error during audio extraction for {reel_url}: {e}")
        return None

async def extract_audio_async(reel_url: str, proxy: str = None) -> Optional[bytes]:
    """
    Asynchronously extract audio by running the synchronous extraction in a thread.

    Args:
        reel_url (str): The URL of the Instagram reel.
        proxy (Optional[str]): Proxy URL if required.

    Returns:
        Optional[bytes]: The extracted WAV audio data or None if extraction failed.
    """
    return await asyncio.to_thread(extract_audio_sync, reel_url, proxy)

async def extract_audio_from_reels(reel_urls: List[str], proxy: Optional[str] = None) -> tuple[Any]:
    """
    Extract audio from multiple reel URLs asynchronously using threads.

    Args:
        reel_urls (List[str]): A list of Instagram reel URLs.
        proxy (Optional[str]): Proxy URL if required.

    Returns:
        List[Optional[bytes]]: A list containing WAV audio data or None for each URL.
    """
    tasks = [extract_audio_async(url, proxy) for url in reel_urls]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return results


def transcribe_in_memory_wav(reels, required_transcriptions=3, language=None):
    """
    Transcribes multiple in-memory WAV data using Faster Whisper until the required number of transcriptions is obtained.

    Parameters:
    - reels (List[bytes]): A list of WAV audio data as bytes.
    - required_transcriptions (int): The number of successful transcriptions to obtain.
    - language (str, optional): The language of the audio. If known, specify to speed up transcription.

    Returns:
    - Optional[str]: The combined transcribed text separated by newlines if at least `required_transcriptions` are obtained.
                      Returns the combined transcriptions available if fewer are successful.
                      Returns None if no transcriptions are successful.
    """
    global trans_model
    transcriptions = []

    # Initialize Faster-Whisper model (Singleton)
    trans_model = FasterWhisperSingleton.get_instance(model_size="tiny", device="cpu", compute_type="int8")
    idx = 0
    # print(reels)
    for wav_data in reels:
        try:
            if not wav_data or len(wav_data) == 0:
                print(f"Reel {idx}: Received empty audio data for transcription. Skipping.")
                continue

            # Load audio from in-memory bytes
            with io.BytesIO(wav_data) as audio_buffer:
                speech, sample_rate = sf.read(audio_buffer)

            # Ensure correct sampling rate
            if sample_rate != 16000:
                print(f"Reel {idx}: Resampling audio from {sample_rate} Hz to 16000 Hz.")
                # Resample using torchaudio if sample rate is incorrect
                import torch
                import torchaudio.transforms as T

                resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
                speech_tensor = torch.tensor(speech).float()
                speech_resampled = resampler(speech_tensor).numpy()
                speech = speech_resampled
                sample_rate = 16000  # Update sample rate after resampling

            # Perform transcription
            segments, info = trans_model.transcribe(
                speech,
                beam_size=1,  # Greedy decoding for speed
                language=language,  # Specify if known, e.g., "en" for English
                word_timestamps=False,  # Disable word-level timestamps
                best_of=1,  # Keep only the best hypothesis
                without_timestamps=True  # Return only text
            )

            # Combine all segments into a single transcription
            transcription = " ".join([segment.text for segment in segments]).strip()
            if transcription:
                transcriptions.append(transcription)
                print(f"Reel {idx}: Transcription obtained.")
            else:
                print(f"Reel {idx}: Empty transcription received.")

            # Check if required number of transcriptions is reached
            if len(transcriptions) >= required_transcriptions:
                print(f"Required transcriptions ({required_transcriptions}) obtained. Stopping transcription.")
                break

        except Exception as e:
            print(f"Reel {idx}: Error during transcription: {e}")
            continue

    if len(transcriptions) >= required_transcriptions:
        combined_transcription = "\n".join(transcriptions[:required_transcriptions])
        return combined_transcription
    elif transcriptions:
        # Optionally, return whatever transcriptions were obtained
        combined_transcription = "\n".join(transcriptions)
        print(f"Only {len(transcriptions)} transcription(s) obtained.")
        return combined_transcription
    else:
        print("No successful transcriptions obtained.")
        return None


def analyze_influencer_content(transcription):
    prompt = (
        "Analyze the following transcription to provide:\n"
        "1. A concise and overall summary of the Influencer (maximum 25 words).\n"
        "2. Five relevant hashtags.\n"
        "3. Three relevant niches.\n\n"
        f"Transcription: {transcription}"
    )

    if len(str(transcription).split(" ")) < 50:
        print("Transcription is too short for analysis.")
        return None, None, None

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing text from Influencers or Creators"},
                {"role": "user", "content": prompt}
            ],
            response_format=Analyzing,
            max_tokens=900  # Adjust this value based on your needs
        )

        response_message = completion.choices[0].message

        # Check if the model refused to respond
        if response_message.refusal:
            print(f"Model refusal: {response_message.refusal}")
            return None, None, None
        else:
            parsed_response = response_message.parsed
            summary = parsed_response.summary
            hashtags = parsed_response.hashtags  # Convert list to comma-separated string
            niches = parsed_response.niches     # Convert list to comma-separated string
            return summary, hashtags, niches
    except Exception as e:
        print(f"Error during content analysis: {e}")
        return None, None, None

async def main_test(reels,proxy):
    # results = []
    results = await extract_audio_from_reels(reels, proxy=proxy)
    # print(results)
    return results


# ----------------------- Flask Routes -----------------------

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
    global PROXIES
    now = time.time()

    data = request.get_json()
    username = str(data.get('username')).replace("@","").lower()
    if not username:
        return jsonify({'error': 'No username provided'}), 400

    print(f"Analyzing video content for username: {username}")
    image_urls, is_private, reels = extract_image_urls(username)

    if is_private:
        return jsonify({"error": "Profile is private or unavailable"}), 403

    if not image_urls:
        return jsonify({'error': 'No images found for this user'}), 404

    wave_datas = asyncio.run(main_test(reels,PROXIES[0]))

    results = transcribe_in_memory_wav(wave_datas,required_transcriptions=2)

    openaitime=time.time()
    summary, hashtags, niches = analyze_influencer_content(results)
    print("openai time:",time.time() - openaitime)
    collage = create_collage(image_urls, grid_size=(3, 3), image_size=(500, 500), spacing=15)

    # Convert collage image to base64
    buffered = BytesIO()
    collage.save(buffered, format="JPEG")
    collage_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    print(f"Processing Time: {time.time() - now:.2f} seconds")

    return jsonify({
        'username': username,
        'summary': summary,
        'hashtags': hashtags,
        'niches': niches,
        'collage_image': collage_base64
    })

def detect_attribute_route(attribute):
    global clip_model, clip_processor
    clip_model = CLIPModelSingleton.get_instance()
    clip_processor = CLIPProcessorSingleton.get_instance()
    data = request.get_json()
    username = str(data.get('username')).replace("@","").lower()
    if not username:
        return jsonify({'error': 'No username provided'}), 400

    print(f"Detecting {attribute} for username: {username}")
    image_urls, is_private, reels = extract_image_urls(username)

    if is_private:
        return jsonify({"error": "Profile is private or unavailable"}), 403

    if not image_urls:
        return jsonify({'error': 'No images found for this user'}), 404

    collage = create_collage(image_urls, grid_size=(3, 3), image_size=(500, 500), spacing=15)

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
    best_label = result['best_label']
    best_probability = result['best_probability']
    detected_attribute = best_label.replace('Images of', '').strip()
    if best_probability < 0.65:
        detected_attribute = "Not sure, guessing " + detected_attribute

    # Convert collage image to base64
    buffered = BytesIO()
    collage.save(buffered, format="JPEG")
    collage_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({
        'username': username,
        'result': detected_attribute,
        'probability': best_probability,
        'collage_image': collage_base64
    })

# ----------------------- Main Entry Point -----------------------

if __name__ == '__main__':
    # Ensure that the Flask app runs with a production-ready server in a real deployment.
    # The built-in Flask server is not suitable for production.
    app.run(debug=True)
