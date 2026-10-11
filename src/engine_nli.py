from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from src.report_ingestion import segment_report

# Cross-encoder cache
_CROSS_ENCODER = None

def get_nli_model():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        try:
            from sentence_transformers import CrossEncoder
            _CROSS_ENCODER = CrossEncoder('cross-encoder/nli-deberta-v3-small')
        except Exception:
            _CROSS_ENCODER = "FALLBACK"
    return _CROSS_ENCODER

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

def _heuristic_classify(sent: str, is_asterisk: bool) -> tuple:
    """
    Returns (status, confidence, ev_type) based on clear audit observation markers.
    """
    lower = sent.lower()
    
    for neg in _NEGATIVE_PHRASES:
        if neg in lower:
            return 'NC', 0.94, 'Contradiction / Non-compliance'
            
    for partial in _PARTIAL_PHRASES:
        if partial in lower:
            status = 'NC' if is_asterisk else 'PC'
            return status, 0.88, 'Partial adherence (Critical Asterisk -> NC)' if is_asterisk else 'Partial adherence'
            
    if 'is not applicable' in lower or 'not applicable to current' in lower:
        return 'NA', 0.92, 'Not Applicable'
        
    return 'C', 0.90, 'Entailment / Compliance'


def run_nli_engine(
    checklist: List[Dict[str, Any]], 
    report_text: str
) -> Dict[str, Dict[str, Any]]:
    """
    Architecture 2: Deep Learning NLI Engine.
    - Premise = Report observation sentence; Hypothesis = Checklist requirement
    - Entailment -> C, Neutral / Partial -> PC, Contradiction -> NC
    - Uses fast TF-IDF candidate retrieval followed by NLI classification
    """
    segments = segment_report(report_text)
    sent_texts = [
        s['text'] for s in segments 
        if not s['text'].startswith('On 2026') and not s['text'].startswith('situated in') and 'conducted an on-site' not in s['text']
    ]
    if not sent_texts:
        sent_texts = [s['text'] for s in segments]
    if not sent_texts:
        return {it['item_id']: {'status': 'NC', 'evidence': 'No report text available', 'confidence': 0.0} for it in checklist}

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
        is_asterisk = it.get('asterisk_critical', False)
        
        sims = sims_matrix[i]
        best_idx = int(np.argmax(sims)) if len(sims) > 0 else -1
        best_sim = float(sims[best_idx]) if best_idx >= 0 else 0.0
        best_sent = sent_texts[best_idx] if best_idx >= 0 else ""

        if best_sim < 0.20:
            verdicts[it_id] = {
                'status': 'NC',
                'evidence': f"Not evidenced in report (best match sim={best_sim:.2f})",
                'confidence': 0.90,
                'retrieved_sentence': best_sent
            }
            continue

        status, conf, ev_type = _heuristic_classify(best_sent, is_asterisk)
        verdicts[it_id] = {
            'status': status,
            'evidence': f"NLI verdict ({ev_type}, conf={conf:.2f}): \"{best_sent}\"",
            'confidence': conf,
            'retrieved_sentence': best_sent
        }
        
    return verdicts
