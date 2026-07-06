import base64
from io import BytesIO
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx  # Using httpx for async HTTP requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request format required by the grader
class DataRequest(BaseModel):
    image_base64: str
    question: str

@app.post("/answer-image")
async def answer_image(payload: DataRequest):
    try:
        # 1. Clean the base64 string
        header_cleanup = payload.image_base64.split(",")[-1]
        
        # 2. Set up the strict system guidelines for the assignment rules
        system_instruction = (
            "You are an exact data extraction tool. Answer the user's question based strictly on the image. "
            "If the answer is a numeric value, return ONLY the raw number. Do not include currency symbols, "
            "commas, units, letters, or punctuation. Example: return '4089.35' instead of '$4,089.35 USD'."
        )
        combined_prompt = f"{system_instruction}\n\nQuestion: {payload.question}"

        # 3. Format the payload for Google Gemini's API structure via AI Pipe
        ai_pipe_payload = {
            "contents": [
                {
                    "parts": [
                        {"text": combined_prompt},
                        {
                            "inlineData": {
                                "mimeType": "image/png",  # Standard fallback mimeType
                                "data": header_cleanup
                            }
                        }
                    ]
                }
            ]
        }

        # 4. Fetch the AI Pipe Token from environment variables
        AI_PIPE_TOKEN = os.getenv("AI_PIPE_TOKEN") # Securely loaded via Render
        
        headers = {
            "Authorization": f"Bearer {AI_PIPE_TOKEN}",
            "Content-Type": "application/json"
        }

        # 5. Route the request through the specific AI Pipe Gemini proxy endpoint
        # Using Gemini 1.5 Flash as requested for fast document processing
        url = "https://aipipe.org/geminiv1beta/models/gemini-1.5-flash:generateContent"

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=ai_pipe_payload, headers=headers, timeout=30.0)
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            
            # 6. Parse Gemini's standard JSON response layout
            result_json = response.json()
            raw_answer = result_json['candidates'][0]['content']['parts'][0]['text']
            
            return {"answer": raw_answer.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

