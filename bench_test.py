import sys
sys.path.insert(0, '.')
import json
from sklearn.metrics import accuracy_score
from src.engine_nlp import run_classical_nlp_engine

with open('data/sample_reports/synthetic_100_reports.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

CATEGORIES = {
    'General Manufacturing': 'general_manufacturing.json',
    'Milk Processing': 'milk_processing.json',
    'Meat Processing': 'meat_processing.json',
    'Slaughterhouse': 'slaughterhouse.json',
    'Catering': 'catering.json',
    'Retail': 'retail.json',
    'Transport': 'transport.json',
    'Storage & Warehouse': 'storage_and_warehouse.json'
}

def cat_to_label(cat_str):
    norm = cat_str.lower().replace(' & ', '_and_').replace('&', 'and').replace(' ', '_')
    for k, v in CATEGORIES.items():
        if v.replace('.json', '') == norm:
            return k
    return cat_str

def load_checklist(cat_name):
    fname = CATEGORIES.get(cat_name, 'general_manufacturing.json')
    with open(f'data/fssai_checklists/{fname}', 'r', encoding='utf-8') as f:
        return json.load(f)

# Pick 2 per category
per_cat = {}
for rep in dataset:
    cat = rep['category']
    if cat not in per_cat:
        per_cat[cat] = []
    if len(per_cat[cat]) < 2:
        per_cat[cat].append(rep)

test_reports = [r for v in per_cat.values() for r in v]
print(f'Test reports: {len(test_reports)} across {len(per_cat)} categories')

y_true, y_pred = [], []
for rep in test_reports:
    cat_name = cat_to_label(rep['category'])
    chk = load_checklist(cat_name)
    verdicts = run_classical_nlp_engine(chk, rep['report_text'], threshold=0.55)
    for it in chk:
        it_id = it['item_id']
        gt = rep['answer_key'].get(it_id, {}).get('status', 'NC')
        pred = verdicts.get(it_id, {}).get('status', 'NC')
        gt_b = 'C' if gt in ['C', 'PC', 'NA'] else 'NC'
        y_true.append(gt_b)
        y_pred.append(pred)

acc = accuracy_score(y_true, y_pred)
print(f'NLP Engine Accuracy (improved): {acc*100:.1f}%')
gt_C = sum(1 for x in y_true if x == 'C')
gt_NC = sum(1 for x in y_true if x == 'NC')
pred_C = sum(1 for x in y_pred if x == 'C')
pred_NC = sum(1 for x in y_pred if x == 'NC')
print(f'GT distribution: C={gt_C}, NC={gt_NC}')
print(f'Pred distribution: C={pred_C}, NC={pred_NC}')
