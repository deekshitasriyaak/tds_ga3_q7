from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
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


def get_transcript(video_id: str) -> list:
    ytt = YouTubeTranscriptApi()
    transcript = ytt.fetch(video_id)
    return [{"start": entry.start, "text": entry.text} for entry in transcript]


def find_timestamp_by_search(transcript: list, topic: str) -> str:
    topic_lower = topic.lower()
    topic_words = [w for w in topic_lower.split() if len(w) > 2]

    # Try exact phrase match first
    for entry in transcript:
        if topic_lower in entry["text"].lower():
            return seconds_to_hhmmss(entry["start"])

    # Fall back to best keyword match
    best_entry = None
    best_score = 0
    for entry in transcript:
        text_lower = entry["text"].lower()
        score = sum(1 for word in topic_words if word in text_lower)
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry and best_score > 0:
        return seconds_to_hhmmss(best_entry["start"])

    return "00:00:00"


@app.post("/ask")
async def ask(request: AskRequest):
    if not request.video_url or not request.topic:
        raise HTTPException(status_code=422, detail="video_url and topic are required")

    try:
        video_id = extract_video_id(request.video_url)
        transcript = get_transcript(video_id)
        timestamp = find_timestamp_by_search(transcript, request.topic)

        return JSONResponse(content={
            "timestamp": timestamp,
            "video_url": request.video_url,
            "topic": request.topic
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")