from typing import Dict, List, Any, Tuple

def compute_fssai_scores(
    checklist: List[Dict[str, Any]], 
    verdicts: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Official FSSAI scoring calculator according to Auditor Manual rules.
    
    Verdicts format:
    {
       item_id: {
           'status': 'C' | 'PC' | 'NC' | 'NA',
           'evidence': str,
           ...
       }
    }
    
    Rules:
    - Compliance (C): full weight (2 or 4)
    - Partial Compliance (PC): 1 point for normal items. Asterisk items CANNOT be PC -> collapse to NC (0 points)
    - Non-Compliance (NC): 0 points
    - Not Applicable (NA): full weight awarded (excluded from denominator impact per FSSAI manual)
    - Total points awarded vs max applicable points
    - Grade bands:
      A+: >= 90%
      A: 80% - 89%
      B: 50% - 79%
      No grade: < 50%
    """
    total_awarded = 0.0
    total_max = 0.0
    asterisk_failures = []
    
    section_breakdown = {}
    scored_items = []
    
    for item in checklist:
        it_id = item['item_id']
        weight = float(item['weight'])
        is_asterisk = bool(item.get('asterisk_critical', False))
        sec = item.get('section', 'General')
        
        if sec not in section_breakdown:
            section_breakdown[sec] = {
                'awarded': 0.0,
                'max': 0.0,
                'count': 0,
                'c_count': 0,
                'pc_count': 0,
                'nc_count': 0,
                'na_count': 0
            }
            
        v_info = verdicts.get(it_id, {'status': 'NC', 'evidence': 'Not evaluated / omitted'})
        status = v_info.get('status', 'NC').upper().strip()
        
        # Enforce asterisk critical rule: Asterisk cannot be PC -> collapse to NC
        if is_asterisk and status == 'PC':
            status = 'NC'
            v_info['status'] = 'NC'
            v_info['note'] = 'Asterisk-critical item cannot be Partial Compliance; collapsed to NC.'
            
        pts = 0.0
        if status == 'C':
            pts = weight
            section_breakdown[sec]['c_count'] += 1
        elif status == 'PC':
            pts = 1.0  # Only normal items reach here
            section_breakdown[sec]['pc_count'] += 1
        elif status == 'NC':
            pts = 0.0
            section_breakdown[sec]['nc_count'] += 1
            if is_asterisk:
                asterisk_failures.append(it_id)
        elif status == 'NA':
            pts = weight
            section_breakdown[sec]['na_count'] += 1
        else:
            # Default fallback to NC
            status = 'NC'
            pts = 0.0
            section_breakdown[sec]['nc_count'] += 1
            if is_asterisk:
                asterisk_failures.append(it_id)
                
        total_awarded += pts
        total_max += weight
        
        section_breakdown[sec]['awarded'] += pts
        section_breakdown[sec]['max'] += weight
        section_breakdown[sec]['count'] += 1
        
        scored_items.append({
            'item_id': it_id,
            'question': item['question'],
            'section': sec,
            'weight': int(weight),
            'asterisk_critical': is_asterisk,
            'status': status,
            'points_awarded': pts,
            'evidence': v_info.get('evidence', '')
        })
        
    percentage = (total_awarded / total_max * 100.0) if total_max > 0 else 0.0
    
    # Official Grade
    if percentage >= 90.0:
        grade = 'A+'
        status_label = 'Compliance – Exemplar'
    elif percentage >= 80.0:
        grade = 'A'
        status_label = 'Compliance – Satisfactory'
    elif percentage >= 50.0:
        grade = 'B'
        status_label = 'Needs Improvement'
    else:
        grade = 'No grade'
        status_label = 'Non Compliance'
        
    return {
        'total_awarded': round(total_awarded, 1),
        'total_max': round(total_max, 1),
        'percentage': round(percentage, 2),
        'grade': grade,
        'status_label': status_label,
        'asterisk_failures': asterisk_failures,
        'section_breakdown': section_breakdown,
        'scored_items': scored_items
    }
