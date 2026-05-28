# Orchestra + Clover

Orchestra + Clover is a Python AI service that takes a raw app idea and turns it into a structured JSON engineering roadmap.

It generates:
- A project name
- A list of tasks with:
  - `id`
  - `title`
  - `track` (`frontend`, `backend`, or `AI`)
  - `description`
  - `dependencies`

## What It Does

Given a plain-English app concept, the service sends a prompt to Gemini and returns clean JSON you can directly use for planning and execution.

The roadmap is designed to help teams break down work into clear, dependency-aware tasks across frontend, backend, and AI tracks.

## Tech Stack

- Python
- [`google-genai`](https://pypi.org/project/google-genai/)
- [`python-dotenv`](https://pypi.org/project/python-dotenv/)

## Project Structure

- `blueprint.py` - Main script that:
  - Loads `GEMINI_API_KEY` from `.env`
  - Sends the raw app idea to Gemini (`gemini-1.5-flash`)
  - Enforces JSON-only output
  - Parses and pretty-prints the final roadmap JSON

## Setup

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install google-genai python-dotenv
```

3. Add your API key to `.env`:

```env
GEMINI_API_KEY=your_api_key_here
```

## Usage

Run with your app idea as an argument:

```bash
python blueprint.py "An app that converts meeting recordings into actionable sprint tasks."
```

If no argument is provided, `blueprint.py` uses its built-in sample idea.

## Output Format

The service returns JSON in this shape:

```json
{
  "project_name": "string",
  "tasks": [
    {
      "id": "T1",
      "title": "string",
      "track": "frontend",
      "description": "string",
      "dependencies": []
    }
  ]
}
```

`track` is constrained to one of:
- `frontend`
- `backend`
- `AI`

## Current Status

- Week 1: Complete
