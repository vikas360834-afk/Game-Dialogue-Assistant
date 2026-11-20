"""N-gram dialogue engine with personality, emotion, context and evaluation."""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import math, random, re

TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|[.!?]+|[,;:]|\d+")
DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "data" / "game_dialogues.txt"

PERSONALITIES = {
    "warrior": "Speak boldly, directly and protectively. Use short decisive sentences.",
    "merchant": "Speak persuasively and warmly. Mention value, trade and opportunity.",
    "wizard": "Speak mysteriously and thoughtfully. Use arcane imagery and measured phrasing.",
    "villager": "Speak plainly, warmly and helpfully. Sound like an ordinary local.",
    "villain": "Speak menacingly and confidently. Use controlled threats, never cartoonish excess.",
    "royal": "Speak formally and diplomatically. Use courteous, elevated language.",
}
EMOTIONS = {
    "neutral": "Keep the tone balanced.",
    "happy": "Sound cheerful, optimistic and welcoming.",
    "angry": "Sound frustrated, sharp and impatient.",
    "sad": "Sound subdued, regretful and vulnerable.",
    "afraid": "Sound nervous, cautious and worried.",
    "sarcastic": "Use dry wit and playful irony.",
    "mysterious": "Sound cryptic and suspenseful.",
}

def tokenize(text: str) -> list[str]:
    return [t.lower().replace("’", "'") for t in TOKEN_RE.findall(text)]

def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([.!?,;:])", r"\1", text)
    text = re.sub(r"\s+(['’])", r"\1", text)
    return text[:1].upper() + text[1:] if text else text

def load_corpus(path: str | Path = DEFAULT_CORPUS) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dialogue corpus not found: {p}")
    raw = p.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Dialogue corpus is empty: {p}")
    return tokenize(raw)

@dataclass
class NGramModel:
    n: int = 3
    alpha: float = 1.0
    def __post_init__(self):
        if self.n < 2: raise ValueError("n must be at least 2")
        if self.alpha <= 0: raise ValueError("alpha must be greater than zero")
        self.vocabulary: set[str] = set()
        self.context_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.total_tokens = 0
    def fit(self, tokens: list[str]):
        if not tokens: raise ValueError("Cannot train on an empty token list")
        self.vocabulary = set(tokens) | {"<s>", "</s>", "<unk>"}
        self.unigram_counts = Counter(tokens)
        self.total_tokens = len(tokens)
        self.context_counts = defaultdict(Counter)
        padded = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
        for i in range(self.n - 1, len(padded)):
            self.context_counts[tuple(padded[i-self.n+1:i])][padded[i]] += 1
        return self
    def probabilities(self, context):
        if not self.vocabulary: raise RuntimeError("Train the model first")
        ctx = tuple(context[-(self.n-1):])
        counts = self.context_counts.get(ctx, Counter())
        denom = sum(counts.values()) + self.alpha * len(self.vocabulary)
        return {w: (counts.get(w, 0)+self.alpha)/denom for w in sorted(self.vocabulary)}
    def next_token(self, context, rng=None, temperature=1.0):
        if temperature <= 0: raise ValueError("temperature must be greater than zero")
        rng = rng or random.Random()
        probs = self.probabilities(context)
        candidates = [(w, p ** (1.0/temperature)) for w,p in probs.items() if w not in {"<s>", "<unk>"}]
        words, weights = zip(*candidates)
        return rng.choices(words, weights=weights, k=1)[0]
    def generate(self, prompt="", max_tokens=35, seed=None, temperature=.85):
        if not self.vocabulary: raise RuntimeError("Train the model first")
        if max_tokens < 1: raise ValueError("max_tokens must be at least 1")
        if temperature <= 0: raise ValueError("temperature must be greater than zero")
        rng = random.Random(seed)
        out = tokenize(prompt)
        ctx = (["<s>"] * max(0, self.n-1-len(out)) + out)[-(self.n-1):]
        for _ in range(max_tokens):
            token = self.next_token(ctx, rng, temperature)
            if token == "</s>":
                if out: break
                continue
            out.append(token); ctx = (ctx + [token])[-(self.n-1):]
        return detokenize(out)
    def evaluate_perplexity(self, tokens):
        if not tokens: raise ValueError("Evaluation tokens cannot be empty")
        padded = ["<s>"]*(self.n-1)+tokens
        logs=[]
        for i in range(self.n-1, len(padded)):
            p=self.probabilities(padded[i-self.n+1:i]).get(padded[i], self.alpha/len(self.vocabulary))
            logs.append(math.log(p))
        return math.exp(-sum(logs)/len(logs))

def train_default(n=3, corpus_path=DEFAULT_CORPUS):
    return NGramModel(n=n).fit(load_corpus(corpus_path))

def build_prompt(prompt, personality="villager", emotion="neutral", location="village", context="", history=None):
    parts = [f"Character archetype: {personality}. {PERSONALITIES.get(personality, PERSONALITIES['villager'])}",
             f"Emotion: {emotion}. {EMOTIONS.get(emotion, EMOTIONS['neutral'])}",
             f"Location: {location}."]
    if context: parts.append(f"World context: {context}.")
    if history: parts.append("Conversation so far: " + " ".join(history[-4:]))
    parts.append("Continue naturally from: " + prompt)
    return "\n".join(parts)

def generate_dialogue(model, prompt="", personality="villager", emotion="neutral",
                      location="village", context="", history=None, length=35,
                      options=1, temperature=.85, seed=None):
    """Generate variations; conditioning nudges the corpus prompt using a compact style vocabulary."""
    if options < 1: raise ValueError("options must be at least 1")
    style_words = {
        "warrior": "honor battle defend sword", "merchant": "gold trade price goods",
        "wizard": "magic spell ancient stars", "villager": "friend home village",
        "villain": "power fear destroy darkness", "royal": "kingdom duty noble"
    }.get(personality, "")
    emotion_words = {
        "happy": "glad joy welcome wonderful", "angry": "fury rage enough", "sad": "sorry loss remember",
        "afraid": "danger careful hide", "sarcastic": "oh splendid amusing", "mysterious": "perhaps secret shadow"
    }.get(emotion, "")
    enriched = " ".join(x for x in [prompt, style_words, emotion_words] if x).strip()
    results=[]
    for i in range(options):
        text = model.generate(prompt or enriched, max_tokens=length, seed=None if seed is None else seed+i,
                              temperature=temperature)
        if text and text not in results: results.append(text)
    while len(results) < options:
        results.append(model.generate(prompt, max_tokens=length, seed=None if seed is None else seed+len(results)+17,
                                      temperature=min(1.5, temperature+.15)))
    return results

def split_tokens(tokens, ratio=.8):
    if not .5 <= ratio < 1: raise ValueError("train ratio must be >= .5 and < 1")
    cut=int(len(tokens)*ratio)
    if cut < 3 or cut >= len(tokens): raise ValueError("Corpus too small for evaluation split")
    return tokens[:cut], tokens[cut:]
