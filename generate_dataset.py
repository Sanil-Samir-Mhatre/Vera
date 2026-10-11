import os, json, random
from src.scoring import compute_fssai_scores

# Load all 8 checklists
CHECKLIST_DIR = 'data/fssai_checklists'
cats = [
    ('General Manufacturing', 'general_manufacturing.json'),
    ('Milk Processing', 'milk_processing.json'),
    ('Meat Processing', 'meat_processing.json'),
    ('Slaughterhouse', 'slaughterhouse.json'),
    ('Catering', 'catering.json'),
    ('Retail', 'retail.json'),
    ('Transport', 'transport.json'),
    ('Storage & Warehouse', 'storage_and_warehouse.json')
]

facilities_prefixes = [
    'Annapurna', 'Kaveri', 'Himalayan', 'Deccan', 'Ganga Valley', 'Satpura', 'Vindhya', 'Sahyadri',
    'Konkan', 'Brahmaputra', 'Malabar', 'Coromandel', 'Thar', 'Godavari', 'Narmada', 'Shivalik'
]
facilities_suffixes = [
    'Foods Ltd.', 'Agro Industries', 'Dairy & Foods', 'Cold Chain Logistics', 'Retail Mart',
    'Catering Solutions', 'Food Processing Hub', 'Fresh Meats', 'Packers & Logistics', 'Harvest Hub'
]

locations = [
    ('Indore', 'Madhya Pradesh', 'Tier-2'),
    ('Surat', 'Gujarat', 'Metro'),
    ('Coimbatore', 'Tamil Nadu', 'Tier-2'),
    ('Nashik', 'Maharashtra', 'Tier-2'),
    ('Ludhiana', 'Punjab', 'Tier-2'),
    ('Varanasi', 'Uttar Pradesh', 'Tier-2'),
    ('Dharwad', 'Karnataka', 'Tier-3'),
    ('Solan', 'Himachal Pradesh', 'Tier-3'),
    ('Kurnool', 'Andhra Pradesh', 'Tier-3'),
    ('Muzaffarpur', 'Bihar', 'Tier-3'),
    ('Bengaluru', 'Karnataka', 'Metro'),
    ('Hyderabad', 'Telangana', 'Metro'),
    ('Kochi', 'Kerala', 'Tier-2'),
    ('Guwahati', 'Assam', 'Tier-2'),
    ('Ranchi', 'Jharkhand', 'Tier-2'),
    ('Cuttack', 'Odisha', 'Tier-2')
]

auditors = [
    'Dr. Rajeshwar Sharma', 'Sunita Nair', 'Vikramaditya Rao', 'Meenakshi Iyer',
    'Arvind Kulshreshtha', 'Preeti Deshmukh', 'Farhan Qureshi', 'Harpreet Kaur',
    'Ananya Sen', 'Brijesh Bhattacharya', 'Kavita Menon', 'Siddharth Varma'
]

# 100 reports total: 8 categories * 12 or 13 = 100
counts_per_cat = [13, 13, 13, 12, 13, 12, 12, 12] # Sum = 100

profiles_dist = ['Clean', 'Clean', 'Clean', 'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Weak', 'Weak', 'Weak', 'Clustered-failure', 'Clustered-failure']

reports = []
report_idx = 1

random.seed(42)

for (cat_name, fname), num_reports in zip(cats, counts_per_cat):
    with open(os.path.join(CHECKLIST_DIR, fname), 'r', encoding='utf-8') as f:
        checklist = json.load(f)
        
    sections = list(set(it['section'] for it in checklist))
    
    for r_in_cat in range(num_reports):
        # Pick profile
        if r_in_cat < 3:
            profile = 'Clean'
        elif r_in_cat < 7:
            profile = 'Moderate'
        elif r_in_cat < 10:
            profile = 'Weak'
        else:
            profile = 'Clustered-failure'
            
        fac_name = f"{random.choice(facilities_prefixes)} {random.choice(facilities_suffixes)}"
        aud_name = random.choice(auditors)
        city, state, city_tier = random.choice(locations)
        date = f"2026-0{random.randint(1,9)}-{random.randint(10,28)}"
        
        fail_section = random.choice(sections) if profile == 'Clustered-failure' else None
        
        # Build answer key & report sentences
        answer_key = {}
        report_sentences = []
        
        # Meta intro prose
        intro_sentence = f"On {date}, food safety auditor {aud_name} conducted an on-site audit at {fac_name} situated in {city}, {state} ({city_tier})."
        report_sentences.append(intro_sentence)
        
        for it in checklist:
            it_id = it['item_id']
            weight = it['weight']
            is_crit = it['asterisk_critical']
            sec = it['section']
            q = it['question']
            
            # Determine status based on profile
            if profile == 'Clean':
                # 90-100% C, very rare PC (non-crit) or rare NA
                roll = random.random()
                if roll < 0.88:
                    status = 'C'
                elif roll < 0.94 and not is_crit:
                    status = 'PC'
                elif roll < 0.98:
                    status = 'NA'
                else:
                    status = 'NC'
            elif profile == 'Moderate':
                # 65-85% compliance
                roll = random.random()
                if roll < 0.65:
                    status = 'C'
                elif roll < 0.82 and not is_crit:
                    status = 'PC'
                elif roll < 0.88:
                    status = 'NA'
                else:
                    status = 'NC'
            elif profile == 'Weak':
                # 40-64% compliance
                roll = random.random()
                if roll < 0.45:
                    status = 'C'
                elif roll < 0.60 and not is_crit:
                    status = 'PC'
                elif roll < 0.65:
                    status = 'NA'
                else:
                    status = 'NC'
            else: # Clustered-failure
                if sec == fail_section:
                    status = 'NC' # Fail entire section
                else:
                    roll = random.random()
                    if roll < 0.85:
                        status = 'C'
                    elif roll < 0.95 and not is_crit:
                        status = 'PC'
                    else:
                        status = 'NC'
                        
            # Asterisk items cannot be PC
            if is_crit and status == 'PC':
                status = 'NC'
                
            # Synthesize realistic natural language observation prose
            short_topic = q.split('.')[0] if '.' in q else q
            if status == 'C':
                sent = f"Regarding {short_topic.lower()}, the facility demonstrated full compliance and verified all related operational records."
            elif status == 'PC':
                sent = f"Observation for {short_topic.lower()}: partial implementation was noted, though formal documentation or log frequency showed procedural gaps."
            elif status == 'NC':
                sent = f"Critical non-compliance observed in {short_topic.lower()}: the facility failed to implement required controls and lacked active verification records."
            else: # NA
                sent = f"Requirement for {short_topic.lower()} is not applicable to current facility operations and scope."
                
            report_sentences.append(sent)
            answer_key[it_id] = {
                'status': status,
                'weight': weight,
                'justification': sent
            }
            
        # Shuffle prose to ensure evidence is mentioned in random order, not checklist order
        header_sent = report_sentences[0]
        body_sents = report_sentences[1:]
        random.shuffle(body_sents)
        full_report_text = header_sent + " " + " ".join(body_sents)
        
        # Calculate ground truth score and grade
        verdicts_format = {k: {'status': v['status'], 'evidence': v['justification']} for k, v in answer_key.items()}
        computed = compute_fssai_scores(checklist, verdicts_format)
        
        rep_id_str = f"Report_{report_idx:03d}"
        reports.append({
            'report_id': rep_id_str,
            'category': cat_name,
            'facility_name': fac_name,
            'auditor_name': aud_name,
            'date': date,
            'location': f"{city}, {state} ({city_tier})",
            'compliance_profile': profile,
            'total_score': computed['total_awarded'],
            'total_max': computed['total_max'],
            'percentage': computed['percentage'],
            'grade': computed['grade'],
            'status_label': computed['status_label'],
            'report_text': full_report_text,
            'answer_key': answer_key
        })
        
        report_idx += 1

# Save dataset
os.makedirs('data/sample_reports', exist_ok=True)
with open('data/sample_reports/synthetic_100_reports.json', 'w', encoding='utf-8') as f:
    json.dump(reports, f, indent=2, ensure_ascii=False)

# Also save plain text benchmark files for quick uploader testing
for i in [1, 14, 27, 40, 53, 66, 78, 89]:
    sample = reports[i-1]
    with open(f"data/sample_reports/{sample['report_id']}_{sample['category'].replace(' ', '_')}.txt", 'w', encoding='utf-8') as f:
        f.write(f"AUDIT REPORT: {sample['report_id']}\n")
        f.write(f"Category: {sample['category']}\n")
        f.write(f"Facility: {sample['facility_name']}\n")
        f.write(f"Auditor: {sample['auditor_name']} | Date: {sample['date']}\n")
        f.write(f"Profile: {sample['compliance_profile']}\n")
        f.write("-" * 50 + "\n\n")
        f.write(sample['report_text'])

print(f"Generated {len(reports)} reports across 8 categories with ground-truth answer keys and FSSAI scores!")
