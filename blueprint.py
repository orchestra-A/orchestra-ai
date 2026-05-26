"""Generate a structured project blueprint from a raw app idea using Gemini."""

import json # for parsing the JSON response
import os # for reading environment variables
import re # for regular expressions
import sys # for command line arguments

from dotenv import load_dotenv # for loading .env files
from google import genai # for generating the blueprint
from google.genai import types # for typed generation config

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-1.5-flash-8b"

PROMPT_TEMPLATE = """You are a senior software architect. Break the following app idea
into a structured project blueprint.

App idea:
\"\"\"{idea}\"\"\"

Return ONLY a single valid JSON object. Do not include markdown fences, prose,
comments, or any text outside the JSON. The JSON must match this exact schema:

{{
  "project_name": "string",
  "tasks": [
    {{
      "id": "T1",
      "title": "string",
      "track": "frontend" | "backend" | "AI",
      "description": "string",
      "dependencies": ["T0", "..."]
    }}
  ]
}}

Rules:
- Use short stable ids like T1, T2, T3 ...
- "track" must be exactly one of: "frontend", "backend", "AI".
- "dependencies" is an array of task ids this task depends on (use [] if none).
- Cover the full stack needed to ship the idea (UI, APIs, data, AI/ML pieces).
- Output JSON only. No explanations."""


def extract_json(text: str) -> str:
    """Strip code fences or stray prose and return the JSON substring."""
    cleaned = text.strip()

    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model response.")
    return cleaned[start : end + 1]


def generate_blueprint(idea: str) -> dict:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to a .env file in the project root."
        )

    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=PROMPT_TEMPLATE.format(idea=idea),
        config=types.GenerateContentConfig(
            temperature=0.4,
            response_mime_type="application/json",
        ),
    )
    raw = response.text or ""
    payload = extract_json(raw)
    return json.loads(payload)


def main() -> None:
    if len(sys.argv) > 1:
        idea = " ".join(sys.argv[1:])
    else:
        idea = "A mobile app that uses AI to suggest recipes from a photo of your fridge."

    blueprint = generate_blueprint(idea)
    print(json.dumps(blueprint, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
