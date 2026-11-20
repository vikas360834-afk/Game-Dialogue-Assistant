"""Optional OpenAI-compatible LLM backend (Ollama, Groq, Gemini, etc.)."""
import os
from pathlib import Path
import httpx

def _load_env():
    p=Path(".env")
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line=line.strip()
            if line and not line.startswith("#") and "=" in line:
                k,v=line.split("=",1); os.environ.setdefault(k.strip(),v.strip())

def settings():
    _load_env()
    return (os.getenv("LLM_BASE_URL") or "http://localhost:11434/v1",
            os.getenv("LLM_API_KEY",""), os.getenv("LLM_MODEL",""))

def is_configured():
    return bool(settings()[2])

def chat(messages, **options):
    base,key,model=settings()
    if not model: raise RuntimeError("LLM is not configured. Set LLM_MODEL in .env.")
    headers={"Authorization":f"Bearer {key}"} if key else {}
    response=httpx.post(f"{base.rstrip('/')}/chat/completions", headers=headers, timeout=90,
                        json={"model":model,"messages":messages,**options})
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

def hybrid_generate(model, prompt, personality="villager", emotion="neutral", location="village",
                    context="", history=None, length=50, engine="hybrid", temperature=.8):
    """LLM-first in hybrid mode; safely fall back to n-gram when no LLM is configured/unavailable."""
    from .core import build_prompt
    system = ("You write original, game-ready NPC dialogue. Follow the character, emotion and world context. "
              f"Keep the response under approximately {length} words. Do not explain your process.")
    user = build_prompt(prompt, personality, emotion, location, context, history)
    if engine in {"hybrid","llm"} and is_configured():
        try:
            return chat([{"role":"system","content":system},{"role":"user","content":user}],
                        temperature=temperature, max_tokens=max(40,length*2))
        except (httpx.HTTPError, RuntimeError, KeyError, ValueError):
            if engine == "llm": raise
    if engine == "llm":
        raise RuntimeError("LLM mode selected but no configured/available LLM. Configure .env or choose hybrid.")
    from .core import generate_dialogue
    return generate_dialogue(model, prompt, personality, emotion, location, context, history,
                             max(8,length), 1, temperature)[0]
