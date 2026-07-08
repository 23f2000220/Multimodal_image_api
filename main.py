import os
import re
import base64
import binascii
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("answer-image-api")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")  # optional, for OpenAI-compatible proxies
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set. Requests will fail until it is configured.")

client_kwargs = {"api_key": OPENAI_API_KEY}
if OPENAI_BASE_URL:
    client_kwargs["base_url"] = OPENAI_BASE_URL

client = OpenAI(**client_kwargs)

app = FastAPI(title="Answer Image API")

# CORS: allow any origin (grader calls from a Cloudflare Worker, origin unknown ahead of time)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnswerImageRequest(BaseModel):
    image_base64: str
    question: str


class AnswerImageResponse(BaseModel):
    answer: str


def _clean_base64(data: str) -> str:
    """Strip data-URL prefix if present, e.g. 'data:image/png;base64,....'"""
    if "," in data and data.strip().lower().startswith("data:"):
        return data.split(",", 1)[1]
    return data.strip()


def _detect_mime(raw: bytes) -> str:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"  # sensible default


def clean_answer(text: str) -> str:
    """Post-process model output into a bare answer string.
    - Strips quotes/whitespace/newlines.
    - For numeric-looking answers, removes currency symbols, units, commas.
    """
    text = text.strip().strip('"').strip("'").strip()
    text = text.split("\n")[0].strip()

    # Only treat as numeric if the text is ALREADY just a number plus optional
    # currency symbol / thousands separators / trailing unit-ish word or %.
    # This avoids mangling genuine text answers like "Q4 Sales".
    match = re.fullmatch(
        r"[$₹€£]?\s*(-?\d[\d,]*\.?\d*)\s*%?(?:\s+[a-zA-Z]+)?",
        text,
    )
    if match:
        number = match.group(1).replace(",", "")
        return number

    return text


SYSTEM_PROMPT = (
    "You are a precise visual data-extraction assistant. You will be shown an image "
    "(which may be a chart, receipt, invoice, table, or pie chart) and asked a question about it. "
    "Respond with ONLY the answer value, nothing else — no explanation, no full sentences, "
    "no currency symbols, no units, no commas in numbers, no surrounding quotes. "
    "If the answer is a number, output just the number (e.g. 4089.35). "
    "If the answer is text, output just that text as concisely as possible."
)


@app.get("/")
def root():
    return {"status": "ok", "service": "answer-image-api"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/answer-image", response_model=AnswerImageResponse)
def answer_image(payload: AnswerImageRequest):
    if not payload.image_base64 or not payload.question:
        raise HTTPException(status_code=400, detail="image_base64 and question are required")

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: OPENAI_API_KEY not set")

    b64_clean = _clean_base64(payload.image_base64)

    try:
        raw = base64.b64decode(b64_clean, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="image_base64 is not valid base64")

    mime = _detect_mime(raw)
    data_url = f"data:{mime};base64,{b64_clean}"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": payload.question},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            max_tokens=100,
            temperature=0,
        )
        raw_answer = response.choices[0].message.content or ""
    except Exception as exc:
        logger.exception("Error calling multimodal model")
        raise HTTPException(status_code=502, detail=f"Upstream model error: {exc}")

    answer = clean_answer(raw_answer)
    return AnswerImageResponse(answer=answer)