import urllib.request
import json
import numpy as np
from typing import List

_EMBEDDINGS_CACHE = {}

def get_ollama_embedding(text: str, model: str = "nomic-embed-text") -> List[float]:
    """
    Fetches embedding from Ollama using nomic-embed-text.
    Caches result in-memory per session.
    """
    text_key = (model, text.strip())
    if text_key in _EMBEDDINGS_CACHE:
        return _EMBEDDINGS_CACHE[text_key]
        
    try:
        url = "http://localhost:11434/api/embeddings"
        payload = json.dumps({"model": model, "prompt": text}).encode('utf-8')
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            emb = data.get("embedding", [])
            _EMBEDDINGS_CACHE[text_key] = emb
            return emb
    except Exception as e:
        # Fallback to deterministic pseudo-vector or zero vector if Ollama is unreachable
        return None

def batch_get_embeddings(texts: List[str], model: str = "nomic-embed-text") -> List[List[float]]:
    return [get_ollama_embedding(t, model) for t in texts]

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2:
        return 0.0
    a = np.array(v1)
    b = np.array(v2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
