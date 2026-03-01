from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
import os
import json
import re
import urllib.request
import urllib.parse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

class AskRequest(BaseModel):
    video_url: str
    topic: str


def extract_video_id(url: str) -> str:
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
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_transcript_via_apify(video_url: str) -> list:
    """Use Apify to fetch YouTube transcript bypassing IP blocks."""
    apify_token = os.environ.get("APIFY_TOKEN")

    # Start the Apify actor run
    actor_url = f"https://api.apify.com/v2/acts/topaz~youtube-transcript-scraper/run-sync-get-dataset-items?token={apify_token}&timeout=60"

    payload = json.dumps({
        "urls": [video_url],
        "outputFormat": "captions",
        "maxRetries": 3,
        "channelHandleBoolean": False,
        "channelNameBoolean": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        actor_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=90) as response:
        result = json.loads(response.read().decode())

    if not result or len(result) == 0:
        raise ValueError("No transcript returned from Apify")

    # Extract captions with timestamps
    captions = result[0].get("captions", [])
    return [{"start": c.get("start", 0), "text": c.get("text", "")} for c in captions]


def find_timestamp_with_groq(transcript: list, topic: str) -> str:
    transcript_text = "\n".join([
        f"[{seconds_to_hhmmss(entry['start'])}] {entry['text']}"
        for entry in transcript
    ])[:12000]

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a transcript analyzer. Given a transcript with timestamps, "
                    "find the FIRST timestamp where the given topic is spoken or discussed. "
                    "Respond ONLY with a JSON object like: {\"timestamp\": \"00:05:47\"} "
                    "Use HH:MM:SS format. No extra text."
                )
            },
            {
                "role": "user",
                "content": f"Topic to find: \"{topic}\"\n\nTRANSCRIPT:\n{transcript_text}"
            }
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    result = json.loads(response.choices[0].message.content)
    ts = result.get("timestamp", "00:00:00").strip()

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
        transcript = get_transcript_via_apify(request.video_url)
        timestamp = find_timestamp_with_groq(transcript, request.topic)

        return JSONResponse(content={
            "timestamp": timestamp,
            "video_url": request.video_url,
            "topic": request.topic
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")