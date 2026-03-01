from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import json
import re
from youtube_transcript_api import YouTubeTranscriptApi
from google import genai
from google.genai import types

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


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


def seconds_to_hhmmss(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_transcript(video_id: str) -> list:
    """Fetch transcript from YouTube."""
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return transcript


def find_timestamp_with_gemini(transcript: list, topic: str) -> str:
    """Use Gemini to find when a topic is mentioned in the transcript."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    # Format transcript as text with timestamps
    transcript_text = "\n".join([
        f"[{seconds_to_hhmmss(entry['start'])}] {entry['text']}"
        for entry in transcript
    ])

    prompt = f"""You are given a YouTube video transcript with timestamps in [HH:MM:SS] format.

Find the FIRST timestamp where the topic "{topic}" is spoken or discussed.

TRANSCRIPT:
{transcript_text[:15000]}

Return the exact timestamp from the transcript where "{topic}" first appears."""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "timestamp": types.Schema(
                        type=types.Type.STRING,
                        description="Timestamp in HH:MM:SS format e.g. 00:05:47"
                    )
                },
                required=["timestamp"]
            )
        )
    )

    result = json.loads(response.text)
    ts = result.get("timestamp", "00:00:00").strip()

    # Ensure HH:MM:SS format
    parts = ts.split(":")
    if len(parts) == 2:
        ts = f"00:{parts[0].zfill(2)}:{parts[1].zfill(2)}"
    elif len(parts) == 3:
        ts = f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:{parts[2].zfill(2)}"
    else:
        ts = "00:00:00"

    return ts


@app.post("/ask")
async def ask(request: AskRequest):
    if not request.video_url or not request.topic:
        raise HTTPException(status_code=422, detail="video_url and topic are required")

    try:
        # Step 1: Extract video ID
        video_id = extract_video_id(request.video_url)

        # Step 2: Get transcript (no download needed!)
        transcript = get_transcript(video_id)

        # Step 3: Ask Gemini to find the timestamp
        timestamp = find_timestamp_with_gemini(transcript, request.topic)

        return JSONResponse(content={
            "timestamp": timestamp,
            "video_url": request.video_url,
            "topic": request.topic
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")