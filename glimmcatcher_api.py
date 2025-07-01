from fastapi import FastAPI, Request
from pydantic import BaseModel
from Glimmcatcher_backend import get_image_assistant_response

app = FastAPI()

class GlimmcatcherRequest(BaseModel):
    user_input: str
    user_id: str = None

@app.post("/glimmcatcher")
async def glimmcatcher_endpoint(req: GlimmcatcherRequest):
    result = get_image_assistant_response(req.user_input, req.user_id)
    return result

# To run: uvicorn glimmcatcher_api:app --host 0.0.0.0 --port 5001