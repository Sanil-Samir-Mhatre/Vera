import json
import urllib.request
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from src.embeddings import get_ollama_embedding, cosine_similarity
from src.report_ingestion import segment_report

# ── tunables ──────────────────────────────────────────────────────────────────
_LLM_TIMEOUT       = 20          # seconds per phi4-mini call (was 30)
_EMBED_SIM_THRESH  = 0.18        # below this → TF-IDF fallback, no LLM call
_MAX_PARALLEL      = 4           # concurrent phi4-mini threads
_CONTEXT_CHAR_LIM  = 500         # truncate each chunk to keep prompts short
# ─────────────────────────────────────────────────────────────────────────────

def query_phi4_mini(prompt: str, system_prompt: str = "") -> str:
    """Sends a query to local Ollama phi4-mini (short timeout, capped tokens)."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "phi4-mini",
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 80}   # cap output tokens
    }
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=_LLM_TIMEOUT) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data.get("response", "").strip()
    except Exception as e:
        return f"__TIMEOUT_OR_ERROR__: {e}"


# ── TF-IDF fallback (no LLM) ─────────────────────────────────────────────────
def _tfidf_classify(q_text: str, chunks: List[str], is_asterisk: bool) -> str:
    """Fast keyword/TF-IDF heuristic when embedding similarity is too low to call LLM."""
    combined = " ".join(chunks).lower()
    neg_phrases = [
        "not compliant", "non-compliant", "not found", "not maintained",
        "absent", "missing", "no record", "not available", "failed",
        "not provided", "lacking", "deficient", "inadequate", "violation"
    ]
    partial_phrases = [
        "partially", "some", "occasional", "not always", "insufficient",
        "needs improvement", "under review", "in progress"
    ]
    if any(p in combined for p in neg_phrases):
        return "NC"
    if not is_asterisk and any(p in combined for p in partial_phrases):
        return "PC"
    return "C"


# ── Per-item LLM call (runs inside thread pool) ───────────────────────────────
def _classify_item_llm(it: Dict[str, Any], top_chunks: List[str]) -> Tuple[str, str, str]:
    """Returns (item_id, status, quote). Called concurrently."""
    it_id   = it['item_id']
    q_text  = it['question']
    is_ast  = it.get('asterisk_critical', False)

    context_str = "\n".join([f"- {c[:_CONTEXT_CHAR_LIM]}" for c in top_chunks])
    prompt = (
        f'Item [{it_id}]: "{q_text}"\n'
        f'Critical: {is_ast}\n'
        f'Observations:\n{context_str}\n\n'
        'Reply ONLY JSON: {"status":"C"|"PC"|"NC"|"NA","quoted_sentence":"exact quote or N/A"}'
    )
    response_text = query_phi4_mini(prompt)

    if "__TIMEOUT_OR_ERROR__" in response_text:
        return it_id, _tfidf_classify(q_text, top_chunks, is_ast), top_chunks[0]

    try:
        s = response_text.find('{')
        e = response_text.rfind('}') + 1
        if s >= 0 and e > s:
            res    = json.loads(response_text[s:e])
            status = res.get('status', 'NC').upper()
            quote  = res.get('quoted_sentence', '')
        else:
            raise ValueError("no JSON")
    except Exception:
        status = _tfidf_classify(q_text, top_chunks, is_ast)
        quote  = top_chunks[0] if top_chunks else "N/A"

    if is_ast and status == 'PC':
        status = 'NC'

    return it_id, status, quote


# ── Main engine ───────────────────────────────────────────────────────────────
def run_rag_engine(
    checklist: List[Dict[str, Any]],
    report_text: str,
    top_k: int = 3
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Architecture 3: RAG Engine with phi4-mini.
    Optimisations vs. original:
      • Report sentence embeddings fetched ONCE (not repeated per item)
      • Items below similarity threshold skip phi4-mini → TF-IDF heuristic
      • All LLM calls run in parallel (ThreadPoolExecutor, 4 workers)
      • Prompt shortened + token cap (num_predict=80)
    """
    segments = segment_report(report_text)
    if not segments:
        return {it['item_id']: {'status': 'NC', 'evidence': 'No report text', 'faithful': False}
                for it in checklist}, {}

    sent_texts = [s['text'] for s in segments]

    # ── 1. Embed ALL report sentences once ──────────────────────────────────
    sent_embs = [get_ollama_embedding(t) for t in sent_texts]

    # Build TF-IDF backup for retrieval when embeddings fail
    try:
        tfidf_vec  = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_mat  = tfidf_vec.fit_transform(sent_texts)
        _has_tfidf = True
    except Exception:
        _has_tfidf = False

    # ── 2. Retrieve top-k per item, decide LLM vs fast fallback ─────────────
    items_for_llm = []   # (item_dict, top_chunks) → need LLM
    verdicts      = {}

    for it in checklist:
        it_id  = it['item_id']
        q_text = it['question']
        is_ast = it.get('asterisk_critical', False)

        q_emb = get_ollama_embedding(q_text)
        if q_emb and any(e is not None for e in sent_embs):
            sims    = [cosine_similarity(q_emb, e) if e else 0.0 for e in sent_embs]
            max_sim = max(sims)
        else:
            sims    = [0.0] * len(sent_texts)
            max_sim = 0.0

        top_indices = np.argsort(sims)[::-1][:top_k]
        top_chunks  = [sent_texts[i] for i in top_indices if sims[i] > 0.10]

        # TF-IDF retrieval fallback when embedding gives nothing
        if not top_chunks and _has_tfidf:
            try:
                q_tfidf = tfidf_vec.transform([q_text])
                t_sims  = sk_cosine(q_tfidf, tfidf_mat)[0]
                t_top   = np.argsort(t_sims)[::-1][:top_k]
                top_chunks = [sent_texts[i] for i in t_top if t_sims[i] > 0.05]
            except Exception:
                pass

        if not top_chunks:
            verdicts[it_id] = {
                'status': 'NC', 'evidence': 'N/A – not mentioned',
                'faithful': True, 'retrieved_chunks': []
            }
            continue

        # Low similarity → skip LLM entirely
        if max_sim < _EMBED_SIM_THRESH:
            status = _tfidf_classify(q_text, top_chunks, is_ast)
            verdicts[it_id] = {
                'status': status,
                'evidence': f'[fast] {top_chunks[0][:200]}',
                'faithful': True,
                'retrieved_chunks': top_chunks
            }
        else:
            items_for_llm.append((it, top_chunks))

    # ── 3. Run LLM calls in parallel ─────────────────────────────────────────
    hallucination_count = 0
    total_evaluated     = 0

    with ThreadPoolExecutor(max_workers=_MAX_PARALLEL) as pool:
        futures = {
            pool.submit(_classify_item_llm, it, chunks): (it, chunks)
            for it, chunks in items_for_llm
        }
        for future in as_completed(futures):
            it, top_chunks = futures[future]
            it_id  = it['item_id']
            is_ast = it.get('asterisk_critical', False)
            try:
                _, status, quote = future.result()
            except Exception:
                status = _tfidf_classify(it['question'], top_chunks, is_ast)
                quote  = top_chunks[0] if top_chunks else "N/A"

            faithful = (
                any(quote.strip().lower() in c.lower() or c.lower() in quote.strip().lower()
                    for c in top_chunks)
                if quote and quote not in ("N/A", "N/A – not mentioned")
                else True
            )
            if not faithful:
                hallucination_count += 1
            total_evaluated += 1

            verdicts[it_id] = {
                'status': status,
                'evidence': f'Cited: "{quote}"',
                'faithful': faithful,
                'retrieved_chunks': top_chunks
            }

    diagnostics = {
        'total_evaluated': total_evaluated,
        'faithfulness_rate': round(1.0 - (hallucination_count / max(total_evaluated, 1)), 3),
        'hallucinations_detected': hallucination_count,
        'top_k': top_k
    }
    return verdicts, diagnostics


# ── RAG Bot ───────────────────────────────────────────────────────────────────
def rag_bot_answer(question: str, report_text: str) -> str:
    """Conversational RAG bot for auditor follow-up questions."""
    segments = segment_report(report_text)
    if not segments:
        return "No audit report content loaded."

    sent_texts = [s['text'] for s in segments]
    sent_embs  = [get_ollama_embedding(t) for t in sent_texts]
    q_emb      = get_ollama_embedding(question)

    if q_emb and any(e is not None for e in sent_embs):
        sims         = [cosine_similarity(q_emb, e) if e else 0.0 for e in sent_embs]
        relevant_ctx = "\n".join(sent_texts[i] for i in np.argsort(sims)[::-1][:4])
    else:
        relevant_ctx = "\n".join(sent_texts[:4])

    prompt = (
        f'Question: "{question}"\n'
        f'Context:\n{relevant_ctx[:1200]}\n\n'
        'Answer concisely with cited evidence.'
    )
    return query_phi4_mini(prompt)
