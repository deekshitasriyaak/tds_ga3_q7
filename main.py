from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
import os
import json
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    video_url: str
    topic: str


def normalize_timestamp(ts: str) -> str:
    parts = ts.strip().split(":")
    if len(parts) == 2:
        return f"00:{parts[0].zfill(2)}:{parts[1].zfill(2)}"
    elif len(parts) == 3:
        return f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:{parts[2].zfill(2)}"
    return "00:00:00"


@app.post("/ask")
async def ask(request: AskRequest):
    if not request.video_url or not request.topic:
        raise HTTPException(status_code=422, detail="video_url and topic are required")

    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        prompt = f"""You are analyzing a YouTube video.
Video URL: {request.video_url}

Find the FIRST timestamp in the video where the topic "{request.topic}" is spoken or discussed.

Return ONLY a JSON object with the timestamp in HH:MM:SS format.
Example: {{"timestamp": "00:05:47"}}"""

        response = client.models.generate_content(
            model="gemini-1.5-flash-latest",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "timestamp": types.Schema(
                            type=types.Type.STRING,
                            description="Timestamp in HH:MM:SS format"
                        )
                    },
                    required=["timestamp"]
                )
            )
        )

        result = json.loads(response.text)
        timestamp = normalize_timestamp(result.get("timestamp", "00:00:00"))

        return JSONResponse(content={
            "timestamp": timestamp,
            "video_url": request.video_url,
            "topic": request.topic
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")