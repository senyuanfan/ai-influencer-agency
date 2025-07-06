#!/usr/bin/env python3
"""
Minimal Planning-Agent runner:
• Talks to a local MCP server for interview segments.
• Sends each segment to OpenAI o3 with built-in Web Search tool.
• Writes k=3 plans/segment into shot_plans.md.
Requires: openai>=1.3, requests, python>=3.9
"""
import os, json, logging, requests, textwrap
import openai
from dotenv import load_dotenv

load_dotenv()

from prompts import PLANNER_PROMPT

# --- basic setup ------------------------------------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")             # :contentReference[oaicite:0]{index=0}
MCP_BASE = os.getenv("MCP_BASE_URL", "http://localhost:8000")
OUTPUT_FILE = "shot_plans.md"
K_VARIANTS = 3

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(message)s")

# --- helper -----------------------------------------------------------------
def fetch_segments() -> list[str]:
    resp = requests.get(f"{MCP_BASE}/split_segments")
    resp.raise_for_status()                              # :contentReference[oaicite:1]{index=1}
    return resp.json()                                   # list of URIs

def plan_segment(uri: str) -> list[dict]:
    messages = [
        {"role": "system", "content": PLANNER_PROMPT.format(k=K_VARIANTS)},
        {"role": "user", "content":
            {"type": "resource", "resource": {"uri": uri}}}
    ]
    response = openai.chat.completions.create(           # :contentReference[oaicite:2]{index=2}
        model="o3",
        messages=messages,
        tools=["web-search"]                             # uses built-in Web Search  :contentReference[oaicite:3]{index=3}
    )
    return json.loads(response.choices[0].message.content)

# --- main -------------------------------------------------------------------
def main():
    segments = fetch_segments()
    md_lines = ["# Shot plan variants\n"]
    for seg_id, uri in enumerate(segments, 1):
        try:
            plans = plan_segment(uri)
            md_lines.append(f"\n## Segment {seg_id}\n")
            for idx, plan in enumerate(plans, 1):
                md_lines.append(f"### Variant {idx}\n")
                md_lines.append("```json")
                md_lines.append(json.dumps(plan, ensure_ascii=False, indent=2))
                md_lines.append("```\n")
            logging.info("Segment %s → %d plans", seg_id, len(plans))
        except Exception as exc:
            logging.error("Segment %s failed: %s", seg_id, exc)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:  # :contentReference[oaicite:4]{index=4}
        fh.write("\n".join(md_lines))
    logging.info("All done → %s", OUTPUT_FILE)

if __name__ == "__main__":
    main()
