#!/usr/bin/env python3
"""
Tiny MCP server exposing:
• GET /split_segments          → ["resource://segment_1", ...]
• GET /resource/{segment_id}   → raw text of that segment
Run:  uvicorn mcp:app --reload          # :contentReference[oaicite:5]{index=5}
"""
import os, re, pathlib
from fastapi import FastAPI, HTTPException                  # :contentReference[oaicite:6]{index=6}
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

load_dotenv()

INTERVIEW_FILE = os.getenv("INTERVIEW_PATH", "data/interivew_results.md")
app = FastAPI(title="Interview-MCP-Server")

# eager load & split once at start-up -----------------------
print(pathlib.Path(INTERVIEW_FILE).resolve())
text = pathlib.Path(INTERVIEW_FILE).resolve().read_text(encoding="utf-8")
segments = [s.strip() for s in re.split(r"\n\s*\n", text) if s.strip()]  # :contentReference[oaicite:7]{index=7}

@app.get("/split_segments")
def split_segments():
    return [f"resource://segment_{i}" for i in range(1, len(segments)+1)]

@app.get("/resource/{segment_id}", response_class=PlainTextResponse)
def get_resource(segment_id: str):
    m = re.match(r"segment_(\d+)", segment_id)
    if not m:
        raise HTTPException(404, "Bad resource id")
    idx = int(m.group(1)) - 1
    if idx >= len(segments):
        raise HTTPException(404, "Segment not found")
    return segments[idx]
