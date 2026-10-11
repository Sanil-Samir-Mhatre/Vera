import sys
sys.path.insert(0, '.')
import json, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
import spacy, re

with open('data/sample_reports/synthetic_100_reports.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)
with open('data/fssai_checklists/general_manufacturing.json', 'r', encoding='utf-8') as f:
    checklist = json.load(f)

r = dataset[0]
report_text = r['report_text']
answer_key = r['answer_key']

try:
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(report_text)
    segments = [sent.text.strip().replace('\n', ' ') for sent in doc.sents if len(sent.text.strip()) > 8]
except:
    segments = [s.strip() for s in re.split(r'(?<=[.!?])\s+', report_text) if len(s.strip()) > 8]

print(f'Segments: {len(segments)}, Checklist items: {len(checklist)}')

# Show a few segments
for s in segments[:5]:
    print(f'  SEG: {s[:100]}')

questions = [it['question'] for it in checklist]
print(f'\nFirst question: {questions[0]}')
print(f'First answer key: {answer_key.get(checklist[0]["item_id"], {})}')

all_texts = segments + questions
vec = TfidfVectorizer(stop_words='english')
mat = vec.fit_transform(all_texts)
n_sents = len(segments)
sent_vecs = mat[:n_sents]
q_vecs = mat[n_sents:]

threshold = 0.55
correct = 0
total = 0
preds_C = 0
preds_NC = 0
gt_C = 0
gt_NC = 0
avg_sim = 0.0
below_thresh = 0

for i, it in enumerate(checklist):
    it_id = it['item_id']
    q_vec = q_vecs[i]
    sims = sk_cosine(q_vec, sent_vecs)[0].tolist()
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    avg_sim += best_sim
    if best_sim < threshold:
        below_thresh += 1
    best_sent = segments[best_idx]
    neg_words = ['not', 'lacking', 'failed', 'absent', 'no records', 'dirty', 'expired', 'pest', 'flaking', 'violation', 'non-compliance']
    has_negative = any(w in best_sent.lower() for w in neg_words)
    pred = 'C' if (best_sim >= threshold and not has_negative) else 'NC'
    gt = answer_key.get(it_id, {}).get('status', 'NC')
    gt_binary = 'C' if gt in ['C', 'PC', 'NA'] else 'NC'
    if pred == gt_binary:
        correct += 1
    total += 1
    if pred == 'C':
        preds_C += 1
    else:
        preds_NC += 1
    if gt_binary == 'C':
        gt_C += 1
    else:
        gt_NC += 1

print(f'\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%')
print(f'Avg best sim: {avg_sim/total:.4f}')
print(f'Sims below threshold ({threshold}): {below_thresh}')
print(f'GT distribution: C={gt_C}, NC={gt_NC}')
print(f'Pred distribution: C={preds_C}, NC={preds_NC}')

# Check a specific item
it = checklist[0]
q = it['question']
print(f'\nItem: {it["item_id"]} Q: {q}')
q_vec = q_vecs[0]
sims_0 = sk_cosine(q_vec, sent_vecs)[0].tolist()
best_idx = int(np.argmax(sims_0))
print(f'Best sim: {sims_0[best_idx]:.4f} -> sentence: {segments[best_idx][:120]}')
