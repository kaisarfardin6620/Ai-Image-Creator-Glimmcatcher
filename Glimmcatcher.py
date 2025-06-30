from dotenv import load_dotenv
import os
import openai
import json
import sys
import sounddevice as sd
import scipy.io.wavfile as wav
from tempfile import NamedTemporaryFile
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image
import urllib.request

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ Error: OPENAI_API_KEY not found in environment variables.")
    exit(1)

client = OpenAI(api_key=api_key)

# System prompt for DALL·E image generation chatbot
SYSTEM_PROMPT = (
    "You are Glimmcatcher, a highly creative DALL·E-powered image generation assistant. "
    "Your role is to interpret user inputs and craft vivid, detailed, and imaginative image descriptions for DALL·E to generate stunning visuals. "
    "Enhance vague or simple requests with rich details, such as mood, lighting, style (e.g., photorealistic, surreal, watercolor), and context, while staying true to the user's intent. "
    "If the input is unclear, generate a refined prompt with creative suggestions, but always confirm alignment with the user's vision. "
    "Provide concise, engaging responses, and ensure the generated images are safe, ethical, and visually compelling. "
    "If the request is not about generating images, simply reply appropriately and do not attempt image generation."
)

# Prompt and response cache for efficiency
prompt_cache = {}
response_cache = {}

# Audio recording and transcription
def record_audio(duration=5, samplerate=44100):
    print(f"🎙️ Recording for {duration} seconds...")
    try:
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
        sd.wait()
        temp_file = NamedTemporaryFile(delete=False, suffix=".wav")
        wav.write(temp_file.name, samplerate, recording)
        print("✅ Recording saved.")
        return temp_file.name
    except Exception as e:
        print(f"❌ Error recording audio: {str(e)}")
        return None

def transcribe_audio(audio_path):
    print("🧠 Transcribing audio...")
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )
        return transcript.text
    except Exception as e:
        print(f"❌ Error transcribing audio: {str(e)}")
        return None

# Image generation with DALL·E
@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=5))
def generate_image(description):
    if not description.strip():
        return "❗ Please provide a valid image description."
    if description in response_cache:
        return response_cache[description]

    try:
        print("🖼️ Generating image with DALL·E...")
        response = client.images.generate(
            model="dall-e-3",
            prompt=description,
            size="1024x1024",
            quality="standard",
            n=1
        )
        image_url = response.data[0].url
        response_cache[description] = image_url
        return image_url
    except openai.OpenAIError as e:
        return f"❌ OpenAI API Error: {str(e)}"
    except Exception as e:
        return f"❌ Unexpected Error: {str(e)}"

# Save image locally
def save_image(image_url, filename="generated_image.png"):
    try:
        urllib.request.urlretrieve(image_url, filename)
        print(f"✅ Image saved as {filename}")
        return filename
    except Exception as e:
        print(f"❌ Error saving image: {str(e)}")
        return None

# Chatbot logic with DALL·E
@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=5))
def ask_image_assistant(user_input):
    if not SYSTEM_PROMPT:
        return "System prompt could not be loaded."
    if not user_input.strip():
        return "❗ Please provide a valid description."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]

    try:
        # First, get a refined description from the chat model
        response = stream_chat_response(messages)
        if "❌" in response:
            return response
        
        # Generate image based on the refined description
        image_url = generate_image(response)
        if "❌" in image_url:
            return image_url
        
        # Save the image locally
        saved_image = save_image(image_url)
        if saved_image:
            return f"✅ Image generated and saved as {saved_image}. URL: {image_url}"
        else:
            return f"✅ Image generated. URL: {image_url}"
    except Exception as e:
        return f"❌ Unexpected Error: {str(e)}"

def stream_chat_response(conversation, model="gpt-4-turbo", temperature=0.7, max_tokens=800):
    assistant_reply = ""
    try:
        response_stream = client.chat.completions.create(
            model=model,
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        for chunk in response_stream:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    print(content, end="", flush=True)
                    assistant_reply += content
        print()  # New line after streaming
        return assistant_reply
    except Exception as e:
        return f"❌ Error generating response: {e}"

# CLI with text/voice support
def run_chat_loop():
    while True:
        try:
            mode = input("\nType 'text' to type a description, 'voice' to speak, or 'exit' to quit: ").strip().lower()
            if mode == "exit":
                print("👋 Goodbye!")
                break
            elif mode == "voice":
                audio_path = record_audio(duration=5)
                if audio_path:
                    user_input = transcribe_audio(audio_path)
                    if user_input:
                        print("🗣️ You said:", user_input)
                    else:
                        print("❌ Failed to transcribe audio.")
                        continue
                else:
                    print("❌ Failed to record audio.")
                    continue
            elif mode == "text":
                user_input = input("Type your image description: ")
            else:
                print("⚠️ Invalid input. Choose 'text', 'voice', or 'exit'.")
                continue

            response = ask_image_assistant(user_input)
            print("\n🤖 Image Assistant:", response)

        except KeyboardInterrupt:
            print("\n❗ Interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

# Main
if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        answer = ask_image_assistant(user_input)
        print("Image Assistant:", answer)
    else:
        run_chat_loop() 
















# 🚀 Modern Sci‑Fi
# “Ultra‑detailed futuristic city skyline at night, neon reflections on water, sleek flying vehicles, cinematic lighting, 8K resolution”

# A humanoid AI robot walking through a transparent tunnel inside a space station, soft ambient blue glows, hyper‑realistic

# Alien jungle biome under a giant ringed planet sky, towering bioluminescent flora, misty and ethereal lighting, wide angle

# 🌍 Cultural & Ethereal
# Traditional Japanese tea ceremony in a zen garden at dawn, soft pastel tones, ultra‑realistic watercolor style

# Colorful Maasai village celebration at sunset, dancers in ornate beaded jewelry, warm golden sunlight, impressionist painting feel

# Ancient Mayan market scene, merchants in traditional garments, earthy tones and rich textures, oil painting

# 🎨 Bold & Graphic
# High‑contrast pop‑art portrait of a jazz musician, vibrant primary colors, thick outlines, retro 1950s style

# Bold geometric abstraction in neon pink, electric blue, and black shapes, inspired by Bauhaus, stark and dynamic

# Surreal digital collage of floating eyes and lips in vivid red and gold, maximalist style, high saturation

# 📜 Old‑Vintage & Nostalgic
# 1920s Paris street café scene, black‑and‑white film grain, soft focus, high contrast shadows

# Retro travel poster for the Orient Express, muted pastel shades, stylized Art Deco typeface and layout

# Turn‑of‑the‑century explorer’s sketchbook page: hand‑drawn jungle plants, sepia tones, ink & watercolor

# ✨ Tips to Enhance Results
# Add art‑style descriptors: e.g. “oil painting,” “photo‑realistic,” “watercolor,” “digital hyper‑realism.”

# Define composition: like “cinematic lighting,” “wide angle,” “intricate detail,” or “soft focus.”

# Specify resolution or aspect: “8K,” “ultra‑high definition,” “square format,” or “16:9 cinematic ratio.”
