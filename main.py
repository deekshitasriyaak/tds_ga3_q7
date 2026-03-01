from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import re
import time
import tempfile
import subprocess
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


def download_audio(video_url: str, output_path: str) -> str:
    """Download audio-only using yt-dlp."""
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-x",                          # extract audio only
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "-o", output_path,
        video_url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")
    return output_path


def upload_and_wait(client: genai.Client, audio_path: str):
    """Upload audio to Gemini Files API and wait for ACTIVE state."""
    uploaded = client.files.upload(path=audio_path)
    # Poll until ACTIVE
    for _ in range(30):
        file_info = client.files.get(name=uploaded.name)
        if file_info.state.name == "ACTIVE":
            return file_info
        elif file_info.state.name == "FAILED":
            raise RuntimeError("Gemini file processing failed")
        time.sleep(3)
    raise RuntimeError("Timeout waiting for file to become ACTIVE")


def ask_gemini_for_timestamp(client: genai.Client, audio_file, topic: str) -> str:
    """Ask Gemini to find when a topic is spoken in the audio."""
    prompt = f"""Listen to this audio carefully and find the FIRST moment where the topic "{topic}" is spoken or discussed.

Return ONLY the timestamp in HH:MM:SS format (e.g. "00:05:47").
If you cannot find it, return the closest relevant moment.
Return just the timestamp string, nothing else."""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Content(parts=[
                types.Part(text=prompt),
                types.Part(file_data=types.FileData(
                    mime_type=audio_file.mime_type,
                    file_uri=audio_file.uri
                ))
            ])
        ],
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

    import json
    result = json.loads(response.text)
    ts = result.get("timestamp", "00:00:00")

    # Ensure HH:MM:SS format
    parts = ts.strip().split(":")
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

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    # Use temp file for audio
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")

        try:
            # Step 1: Download audio
            download_audio(request.video_url, audio_path)

            # Step 2: Upload to Gemini Files API
            audio_file = upload_and_wait(client, audio_path)

            # Step 3: Ask Gemini for timestamp
            timestamp = ask_gemini_for_timestamp(client, audio_file, request.topic)

            # Step 4: Clean up uploaded file
            try:
                client.files.delete(name=audio_file.name)
            except Exception:
                pass

            return JSONResponse(content={
                "timestamp": timestamp,
                "video_url": request.video_url,
                "topic": request.topic
            })

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")