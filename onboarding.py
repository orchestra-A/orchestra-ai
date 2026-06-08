"""Generate a developer profile from GitHub public repos using Gemini."""

import json
import os
import sys

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

MODEL_NAME = "gemini-2.5-flash-lite"
GITHUB_API = "https://api.github.com"

PROMPT_TEMPLATE = """You are a technical recruiter analyzing a developer's GitHub activity.

GitHub username: {username}
Display name: {display_name}

Aggregated language usage across public repositories (language: total bytes):
{languages}

Based on this data, infer the developer's likely role and top skills.

Return ONLY a valid JSON object with this exact shape:
{{
  "name": "string",
  "github": "string",
  "role": "string",
  "skills": ["string"]
}}

Rules:
- "name" must be the display name provided above (or username if display name is empty).
- "github" must be exactly the GitHub username provided.
- "role" should be one concise label like "frontend", "backend", "fullstack", "devops", "mobile", or "data".
- "skills" must be a list of 3 to 5 skill tags (e.g. ["Python", "React", "Neo4j"]).
- Base skills on languages and typical stack associations from the repo data.
- Output JSON only. No markdown, no prose, no extra keys.
"""


def fetch_user(username: str) -> dict:
    """Fetch GitHub user profile."""
    response = requests.get(f"{GITHUB_API}/users/{username}", timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_repos(username: str) -> list[dict]:
    """Fetch all public repositories for a user."""
    repos: list[dict] = []
    page = 1

    while True:
        response = requests.get(
            f"{GITHUB_API}/users/{username}/repos",
            params={"per_page": 100, "page": page},
            timeout=30,
        )
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        repos.extend(batch)
        page += 1

    return repos


def fetch_languages(username: str, repo_name: str) -> dict[str, int]:
    """Fetch language byte counts for one repository."""
    response = requests.get(
        f"{GITHUB_API}/repos/{username}/{repo_name}/languages",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def aggregate_languages(username: str, repos: list[dict]) -> dict[str, int]:
    """Sum language bytes across all public repos."""
    totals: dict[str, int] = {}

    for repo in repos:
        repo_name = repo.get("name", "")
        if not repo_name:
            continue
        try:
            languages = fetch_languages(username, repo_name)
        except requests.HTTPError:
            continue
        for language, byte_count in languages.items():
            totals[language] = totals.get(language, 0) + byte_count

    return totals


def analyze_profile(
    username: str, display_name: str, languages: dict[str, int], api_key: str
) -> dict:
    """Send aggregated language data to Gemini and return a developer profile."""
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=PROMPT_TEMPLATE.format(
            username=username,
            display_name=display_name or username,
            languages=json.dumps(languages, indent=2, ensure_ascii=False),
        ),
        config=types.GenerateContentConfig(
            temperature=0.3,
            response_mime_type="application/json",
        ),
    )

    return json.loads((response.text or "").strip())


def build_profile(username: str, api_key: str) -> dict:
    """Fetch GitHub data and generate a developer profile."""
    user = fetch_user(username)
    display_name = user.get("name") or username
    repos = fetch_repos(username)

    if not repos:
        raise RuntimeError(f"No public repositories found for user '{username}'.")

    languages = aggregate_languages(username, repos)
    if not languages:
        raise RuntimeError(f"No language data found for user '{username}'.")

    return analyze_profile(username, display_name, languages, api_key)


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to a .env file in the project root."
        )

    if len(sys.argv) > 1:
        username = sys.argv[1].strip()
    else:
        username = input("GitHub username: ").strip()

    if not username:
        raise RuntimeError("Username cannot be empty.")

    profile = build_profile(username, api_key)
    print(json.dumps(profile, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
