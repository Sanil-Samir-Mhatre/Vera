import pandas as pd
import io
import os

def export_audit_to_excel(
    checklist_df: pd.DataFrame,
    report_df: pd.DataFrame,
    results_df: pd.DataFrame,
    summary_dict: dict,
    output_path: str = None
) -> bytes:
    """
    Exports BOTH the input (checklist + report) and the output (scored results + summary)
    as separate sheets in one Excel workbook.
    """
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Executive Summary
        summary_rows = [
            {'Metric': 'Overall Score Awarded', 'Value': summary_dict.get('total_awarded')},
            {'Metric': 'Max Possible Points', 'Value': summary_dict.get('total_max')},
            {'Metric': 'Compliance Percentage', 'Value': f"{summary_dict.get('percentage')}%"},
            {'Metric': 'Official FSSAI Grade', 'Value': summary_dict.get('grade')},
            {'Metric': 'Status', 'Value': summary_dict.get('status_label')},
            {'Metric': 'Critical Asterisk Failures', 'Value': ", ".join(summary_dict.get('asterisk_failures', [])) or 'None'}
        ]
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
        
        # Sheet 2: Verification Results
        results_df.to_excel(writer, sheet_name='Scored Results', index=False)
        
        # Sheet 3: Input Checklist
        checklist_df.to_excel(writer, sheet_name='Input Checklist', index=False)
        
        # Sheet 4: Input Report
        report_df.to_excel(writer, sheet_name='Input Report Data', index=False)

    excel_bytes = output.getvalue()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(excel_bytes)
            
    return excel_bytes
