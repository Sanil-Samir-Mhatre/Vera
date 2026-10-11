import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from src.report_ingestion import segment_report

# Negative indicators that unambiguously signify non-compliance
_NEGATIVE_PHRASES = [
    'failed to', 'critical non-compliance', 'lacked active', 'marked nc',
    'deficiency noted', 'deficiency found', 'violation found',
    'pest infestation', 'flaking paint', 'unhygienic',
    'not maintained', 'not available', 'not provided', 'not verified',
    'was absent', 'were absent', 'no records', 'no documentation',
    'no license', 'no calibration', 'expired license'
]

_PARTIAL_PHRASES = [
    'partial implementation', 'procedural gaps', 'minor deficiency',
    'minor gap', 'partially documented', 'pending review', 'in progress'
]

def _classify_sentence(sent: str) -> str:
    """
    Returns 'positive', 'partial', or 'negative' for a sentence.
    Uses precise phrase matching to avoid false positives.
    """
    lower = sent.lower()
    for neg_phrase in _NEGATIVE_PHRASES:
        if neg_phrase in lower:
            return 'negative'
    for partial in _PARTIAL_PHRASES:
        if partial in lower:
            return 'partial'
    return 'positive'

def run_classical_nlp_engine(
    checklist: List[Dict[str, Any]], 
    report_text: str, 
    threshold: float = 0.40
) -> Dict[str, Dict[str, Any]]:
    """
    Architecture 1: Classical NLP Engine.
    - Matches checklist items with report sentences using TF-IDF n-gram cosine similarity.
    - Binary C/NC verdict (Classical NLP baseline does not classify PC).
    - Returns verdicts map: {item_id: {'status': 'C'|'NC', 'evidence': str, 'similarity': float}}
    """
    segments = segment_report(report_text)
    # Filter out audit header lines if present
    sent_texts = [
        s['text'] for s in segments 
        if not s['text'].startswith('On 2026') and not s['text'].startswith('situated in') and 'conducted an on-site' not in s['text']
    ]
    if not sent_texts:
        sent_texts = [s['text'] for s in segments]
    if not sent_texts:
        return {it['item_id']: {'status': 'NC', 'evidence': 'No report text available', 'similarity': 0.0} for it in checklist}

    questions = [it['question'] for it in checklist]
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_mat = vec.fit_transform(sent_texts + questions)
    n_sents = len(sent_texts)
    sent_vecs = tfidf_mat[:n_sents]
    q_vecs = tfidf_mat[n_sents:]
    sims_matrix = sk_cosine(q_vecs, sent_vecs)

    verdicts = {}
    for i, it in enumerate(checklist):
        it_id = it['item_id']
        sims = sims_matrix[i]
        best_idx = int(np.argmax(sims)) if len(sims) > 0 else -1
        best_sim = float(sims[best_idx]) if best_idx >= 0 else 0.0
        best_sent = sent_texts[best_idx] if best_idx >= 0 else ""

        sent_class = _classify_sentence(best_sent)

        if best_sim >= threshold and sent_class != 'negative':
            status = 'C'
            evidence = f"Matched evidence (sim={best_sim:.2f}): \"{best_sent}\""
        elif best_sim >= threshold and sent_class == 'negative':
            status = 'NC'
            evidence = f"Deficiency identified (sim={best_sim:.2f}): \"{best_sent}\""
        else:
            status = 'NC'
            evidence = f"No corroborating evidence met cutoff {threshold:.2f} (highest match {best_sim:.2f})"

        verdicts[it_id] = {
            'status': status,
            'evidence': evidence,
            'similarity': best_sim,
            'matched_sentence': best_sent
        }

    return verdicts
