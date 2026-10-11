import json, os, re

CHECKLIST_DIR = 'data/fssai_checklists'
OKF_DIR = 'data/okf_bundle'
os.makedirs(OKF_DIR, exist_ok=True)

files = [f for f in os.listdir(CHECKLIST_DIR) if f.endswith('.json') and f != 'checklists_index.json']

for fname in files:
    cat_slug = fname.replace('.json', '')
    cat_bundle_dir = os.path.join(OKF_DIR, cat_slug)
    os.makedirs(cat_bundle_dir, exist_ok=True)
    
    with open(os.path.join(CHECKLIST_DIR, fname), 'r', encoding='utf-8') as f:
        items = json.load(f)
    
    cat_name = items[0]['category'] if items else cat_slug
    
    # Group items by section
    sections = {}
    for it in items:
        sec = it['section']
        if sec not in sections:
            sections[sec] = []
        sections[sec].append(it)
    
    # 1. Create index.md
    index_md = f"""---
type: checklist_index
category: "{cat_name}"
slug: "{cat_slug}"
total_items: {len(items)}
total_points: {sum(it['weight'] for it in items)}
---

# {cat_name} Audit Checklist — Traversal Root

Welcome to the knowledge traversal root for **{cat_name}**.
Auditors and automated verifiers can traverse through sections to inspect specific compliance requirements.

## Sections
"""
    for sec_name, sec_items in sections.items():
        sec_slug = re.sub(r'[^a-zA-Z0-9]+', '_', sec_name).strip('_').lower()
        index_md += f"- [{sec_name}](./{sec_slug}.md) ({len(sec_items)} items)\n"
        
        # 2. Create Section Markdown file
        sec_md = f"""---
type: checklist_section
category: "{cat_name}"
section: "{sec_name}"
items_count: {len(sec_items)}
---

# Section: {sec_name}
Category: [{cat_name}](./index.md)

Items in this section:
"""
        for it in sec_items:
            it_id = it['item_id']
            crit_mark = " ★ (Critical)" if it['asterisk_critical'] else ""
            sec_md += f"- [{it_id}: {it['question'][:60]}...](./{it_id}.md){crit_mark} — {it['weight']} pts\n"
        
        with open(os.path.join(cat_bundle_dir, f"{sec_slug}.md"), 'w', encoding='utf-8') as sf:
            sf.write(sec_md)
            
    with open(os.path.join(cat_bundle_dir, 'index.md'), 'w', encoding='utf-8') as inf:
        inf.write(index_md)
        
    # 3. Create Item Markdown files with cross-links
    for it in items:
        it_id = it['item_id']
        sec_name = it['section']
        sec_slug = re.sub(r'[^a-zA-Z0-9]+', '_', sec_name).strip('_').lower()
        
        # Siblings in the same section
        siblings = [x['item_id'] for x in sections[sec_name] if x['item_id'] != it_id]
        sibling_links = ", ".join([f"[{s}](./{s}.md)" for s in siblings[:5]])
        
        item_md = f"""---
type: checklist_item
item_id: "{it_id}"
category: "{cat_name}"
section: "{sec_name}"
weight: {it['weight']}
asterisk_critical: {str(it['asterisk_critical']).lower()}
---

# Item {it_id}: {it['question']}

- **Category**: [{cat_name}](./index.md)
- **Section**: [{sec_name}](./{sec_slug}.md)
- **Weight**: {it['weight']} points
- **Critical (Asterisk)**: {'Yes — Non-Compliance cannot be partial' if it['asterisk_critical'] else 'No'}

## Requirement Description
{it['question']}

## Audit Verification Guidance
- **Compliance (C)**: Requirement fully implemented and evidenced in audit documentation.
- **Partial Compliance (PC)**: {'NOT APPLICABLE FOR ASTERISK CRITICAL ITEMS (Automatically NC).' if it['asterisk_critical'] else 'Partially implemented or slight procedural lapse.'}
- **Non-Compliance (NC)**: Requirement not documented or not implemented.
- **Not Applicable (NA)**: Legitimate non-applicability to this facility's scope.

## Related Items in this Section
{sibling_links if sibling_links else 'None'}
"""
        with open(os.path.join(cat_bundle_dir, f"{it_id}.md"), 'w', encoding='utf-8') as itf:
            itf.write(item_md)

print("OKF Bundle created successfully with index, section, and item cross-links!")
