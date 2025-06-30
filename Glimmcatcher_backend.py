import os
import logging
import re
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI
from dotenv import load_dotenv
import urllib.request

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment variables.")

client = OpenAI(api_key=api_key)

# System prompt for DALL·E image generation assistant
SYSTEM_PROMPT = (
    "You are Glimmcatcher, a highly creative DALL·E-powered image generation assistant. "
    "Your role is to interpret user inputs and craft vivid, detailed, and imaginative image descriptions for DALL·E to generate stunning visuals. "
    "Enhance vague or simple requests with rich details, such as mood, lighting, style (e.g., photorealistic, surreal, watercolor), and context, while staying true to the user's intent. "
    "If the input is unclear, generate a refined prompt with creative suggestions, but always confirm alignment with the user's vision. "
    "Provide concise, engaging responses, and ensure the generated images are safe, ethical, and visually compelling. "
    "If the request is not about generating images, simply reply appropriately and do not attempt image generation."
)

# Intent detection keywords
IMAGE_INTENT_KEYWORDS = [
    "generate image", "draw", "visualize", "illustrate", "render", "create art", "paint", "design",
    "make an image", "show me a picture", "picture of", "scene of", "photo of", "image of"
]

# Caches
prompt_cache = {}
response_cache = {}
conversation_history = {}

def sanitize_input(text: str) -> str:
    """Removes special characters and excessive whitespace from input."""
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return ' '.join(text.split())

def is_image_intent(text: str) -> bool:
    """Checks if input likely implies an image generation request."""
    lowered = text.lower()
    return any(keyword in lowered for keyword in IMAGE_INTENT_KEYWORDS)

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=5))
def _stream_chat_response(conversation, model="gpt-4-turbo", temperature=0.7, max_tokens=800):
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
                    assistant_reply += content
        return assistant_reply
    except Exception as e:
        logger.error(f"Error generating chat response: {e}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=5))
def _generate_image(description):
    if not description.strip():
        return None, "Please provide a valid image description."
    if description in response_cache:
        return response_cache[description], None
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=description,
            size="1024x1024",
            quality="standard",
            n=1
        )
        image_url = response.data[0].url
        response_cache[description] = image_url
        return image_url, None
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return None, f"Failed to generate image: {str(e)}"

def save_image(image_url, filename="generated_image.png"):
    try:
        urllib.request.urlretrieve(image_url, filename)
        logger.info(f"Image saved as {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")
        return None

def get_image_assistant_response(user_input: str, user_id: str = None) -> dict:
    if not user_input or not user_input.strip():
        logger.warning("Empty or invalid input received.")
        return {"status": "error", "message": "Please provide a valid image description or question."}

    question = sanitize_input(user_input)
    if not question:
        logger.warning("Input is empty after sanitization.")
        return {"status": "error", "message": "Please provide a valid image description or question."}

    # Check cache
    if question in response_cache:
        logger.info(f"Cache hit for input: {question}")
        return {
            "status": "success",
            "image_url": response_cache[question],
            "cached": True,
            "follow_up_prompt": "Was this what you wanted? If not, feel free to refine your request!"
        }

    # Prepare chat messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if user_id and user_id in conversation_history:
        messages.extend(conversation_history[user_id])
        logger.info(f"Appended conversation history for user_id: {user_id}")
    messages.append({"role": "user", "content": question})

    try:
        refined_description = _stream_chat_response(messages)
        if not refined_description or "❌" in refined_description:
            return {"status": "error", "message": "Failed to generate refined description."}

        # Advanced image intent detection
        if not is_image_intent(question) and not is_image_intent(refined_description):
            logger.info("Input interpreted as general chat.")
            return {"status": "chat", "message": refined_description}

        # Generate image
        image_url, error = _generate_image(refined_description)
        if error:
            return {"status": "error", "message": error}
        
        saved_image = save_image(image_url) if image_url else None

        # Update conversation history
        if user_id:
            if user_id not in conversation_history:
                conversation_history[user_id] = []
            conversation_history[user_id].extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": refined_description}
            ])
            conversation_history[user_id] = conversation_history[user_id][-10:]

        logger.info(f"Generated image for input: {question}")
        return {
            "status": "success",
            "image_url": image_url,
            "refined_description": refined_description,
            "saved_image": saved_image,
            "cached": False,
            "follow_up_prompt": "Was this what you wanted? If not, feel free to refine your request!"
        }
    except Exception as e:
        logger.error(f"Error in image assistant response: {e}")
        return {"status": "error", "message": f"Failed to generate image: {str(e)}"}
