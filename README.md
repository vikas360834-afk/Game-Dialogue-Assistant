# Game Dialogue Assistant 2.0

A GitHub-ready NPC dialogue prototyping toolkit with a local Add-1 smoothed n-gram model, optional LLM enhancement, CLI and Streamlit dashboard.

## Features

- **N-gram model from scratch:** tokenization, n-gram counts, Laplace/Add-1 smoothing, temperature sampling and held-out perplexity.
- **Hybrid generation:** uses a configured OpenAI-compatible chat endpoint (Ollama, Groq, Gemini, etc.) when available; safely falls back to the local n-gram model.
- **Character controls:** warrior, merchant, wizard, villager, villain and royal archetypes; seven emotional tones.
- **World context:** location, quest/world notes and conversation history inputs.
- **CLI:** train, generate multiple options, interactive chat, evaluate, save and export.
- **Web dashboard:** dialogue studio, NPC profiles, saved dialogue library, model diagnostics, TXT corpus import and JSON/TXT export.
- **Persistence:** local JSON store; no database server required.
- **Tests and CI:** pytest suite and GitHub Actions workflow.

> Academic note: personality/context prompts condition the LLM backend. The pure n-gram model is statistical and does not deeply understand personality or long-range story context; its output quality depends on the corpus. The dashboard labels the engine and fallback clearly.

## Setup (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

## Run the CLI

```powershell
python -m app.cli hello
python -m app.cli train
python -m app.cli generate "The kingdom is in danger" --personality warrior --emotion angry --location castle --options 3 --length 45 --engine ngram
python -m app.cli generate "The old wizard" --personality wizard --emotion mysterious --engine hybrid
python -m app.cli interactive
python -m app.cli evaluate
python -m app.cli save greeting "Welcome, traveler."
python -m app.cli export --format json --output exports/dialogues
```

Use `--engine ngram` for the deterministic academic model, `--engine hybrid` for optional LLM with local fallback, or `--engine llm` to require a configured endpoint.

## Run the web dashboard (Windows - easiest method)

Double-click `run_web.bat` in the extracted project folder. It creates a virtual environment, installs the required packages (including Streamlit), and starts the dashboard automatically.

Or use PowerShell from the project root:

```powershell
py -3 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app/web.py
```

Streamlit prints a local URL, usually `http://localhost:8501`.

## Optional LLM configuration

Edit `.env` (never commit it). Example for local Ollama:

```dotenv
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=
LLM_MODEL=qwen2.5:7b
```

Start Ollama and pull the model first. For hosted providers, set their OpenAI-compatible base URL, API key and model name. Hybrid mode will use the local n-gram model if the LLM is not configured or the request fails. LLM mode reports the error instead.

## Project structure

```text
app/
  core.py       n-gram model, preprocessing, dialogue controls
  llm.py        optional OpenAI-compatible LLM adapter and fallback
  cli.py        command-line interface
  web.py        Streamlit dashboard
  storage.py    local JSON persistence
data/
  game_dialogues.txt
tests/
  test_core.py
  test_app.py
```

## Tests

```powershell
pytest -v
```

## Limitations and responsible use

- The included corpus is a small original demonstration corpus, not a licensed commercial game-script dataset.
- N-gram output can repeat words, produce awkward transitions or fail to maintain plot continuity.
- Add-1 smoothing assigns probability mass to unseen tokens; it is a simple teaching baseline, not state-of-the-art language modeling.
- Hybrid LLM output quality and availability depend on the configured provider/model and network access.
- Do not upload proprietary scripts unless you have permission; avoid putting API keys in source control.

## Milestones

- [x] Collect and preprocess a dialogue corpus.
- [x] Build an Add-1 smoothed n-gram model.
- [x] Implement seeded, temperature-controlled dialogue generation.
- [x] Build a CLI for generation and interactive use.
- [x] Add optional hybrid LLM adapter with offline fallback.
- [x] Add character archetypes, emotions and world context controls.
- [x] Add NPC profile and saved dialogue persistence.
- [x] Add Streamlit web dashboard and export tools.
- [x] Add model evaluation, tests and CI workflow.
