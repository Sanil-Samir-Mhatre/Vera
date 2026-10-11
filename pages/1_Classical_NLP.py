import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.file_converter import normalize_to_excel
from src.engine_nlp import run_classical_nlp_engine
from src.scoring import compute_fssai_scores
from src.excel_export import export_audit_to_excel

st.set_page_config(page_title="Architecture 1 | Classical NLP", page_icon="⚙️", layout="wide")

st.title("⚙️ Architecture 1 — Classical NLP (Embedding Cosine Similarity)")
st.caption("Dense semantic similarity (`nomic-embed-text`) with threshold-based verification and phrase-level negative detection.")

CATEGORIES = {
    "General Manufacturing": "general_manufacturing.json",
    "Milk Processing": "milk_processing.json",
    "Meat Processing": "meat_processing.json",
    "Slaughterhouse": "slaughterhouse.json",
    "Catering": "catering.json",
    "Retail": "retail.json",
    "Transport": "transport.json",
    "Storage & Warehouse": "storage_and_warehouse.json"
}

@st.cache_data
def load_checklist(cat_name):
    fname = CATEGORIES[cat_name]
    with open(os.path.join("data", "fssai_checklists", fname), "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_dataset():
    path = "data/sample_reports/synthetic_100_reports.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def cat_to_label(cat_str):
    """Map dataset category string back to CATEGORIES key."""
    norm = cat_str.lower().replace(' & ', '_and_').replace('&', 'and').replace(' ', '_')
    for k, v in CATEGORIES.items():
        if v.replace('.json', '') == norm:
            return k
    for k in CATEGORIES.keys():
        if k.lower().replace(' & ', ' and ') == cat_str.lower().replace(' & ', ' and '):
            return k
    return None

@st.cache_data(show_spinner=False)
def run_perf_benchmark():
    """
    Evaluate Architecture 1 across 16 reports (2 per category) from the ground truth dataset.
    Returns (y_true, y_pred, sweep_scores, per_category_acc).
    """
    dataset = load_dataset()
    if not dataset:
        return [], [], [], {}

    # Pick 2 reports per category for a balanced 16-report benchmark
    per_cat = {}
    for rep in dataset:
        cat = rep['category']
        if cat not in per_cat:
            per_cat[cat] = []
        if len(per_cat[cat]) < 2:
            per_cat[cat].append(rep)

    test_reports = []
    for v in per_cat.values():
        test_reports.extend(v)

    thresholds = [0.45, 0.55, 0.65, 0.72, 0.80]
    sweep_scores = []
    y_true_main = []
    y_pred_main = []
    per_cat_acc = {}

    for th in thresholds:
        all_t, all_p = [], []
        for rep in test_reports:
            cat = rep['category']
            cat_name = cat_to_label(cat)
            if not cat_name:
                continue
            try:
                chk = load_checklist(cat_name)
            except Exception:
                continue
            verdicts = run_classical_nlp_engine(chk, rep['report_text'], threshold=th)
            for it in chk:
                it_id = it['item_id']
                gt = rep['answer_key'].get(it_id, {}).get('status', 'NC')
                pred = verdicts.get(it_id, {}).get('status', 'NC')
                # Collapse: NA and PC count as Compliant in ground truth
                gt_b = 'C' if gt in ['C', 'PC', 'NA'] else 'NC'
                all_t.append(gt_b)
                all_p.append(pred)

                if th == 0.72:
                    y_true_main.append(gt_b)
                    y_pred_main.append(pred)
                    if cat not in per_cat_acc:
                        per_cat_acc[cat] = {'t': [], 'p': []}
                    per_cat_acc[cat]['t'].append(gt_b)
                    per_cat_acc[cat]['p'].append(pred)

        acc = accuracy_score(all_t, all_p) if all_t else 0.0
        sweep_scores.append({'Threshold': th, 'Accuracy': round(acc * 100, 1)})

    # Per-category accuracy at calibrated threshold 0.72
    cat_acc_rows = []
    for cat, data in per_cat_acc.items():
        cat_acc = accuracy_score(data['t'], data['p']) if data['t'] else 0.0
        cat_acc_rows.append({'Category': cat.replace('_', ' ').title(), 'Accuracy (%)': round(cat_acc * 100, 1)})

    return y_true_main, y_pred_main, sweep_scores, cat_acc_rows

tab_app, tab_perf, tab_example = st.tabs(["🚀 Application", "📈 Performance", "📖 Example"])

# ================= TAB 1: APPLICATION =================
with tab_app:
    st.sidebar.header("Audit Setup")
    selected_cat = st.sidebar.radio("Select Checklist Category", list(CATEGORIES.keys()))
    checklist_data = load_checklist(selected_cat)
    
    threshold_slider = st.sidebar.slider("Cosine Similarity Threshold", min_value=0.30, max_value=0.85, value=0.72, step=0.02)
    st.sidebar.warning("⚠️ **Documented Architecture Limitation**: Classical NLP supports **Compliance (C)** and **Non-Compliance (NC)** only. It cannot reliably distinguish Partial Compliance (PC).")
    
    uploaded_file = st.sidebar.file_uploader("Upload Audit Report", type=["pdf", "docx", "xlsx", "txt"])
    run_btn = st.sidebar.button("Run Verification", type="primary")
    
    if run_btn:
        if not uploaded_file:
            st.error("Please upload an audit report document to begin verification.")
        else:
            with st.spinner("1. Converting & Normalizing uploaded document to Excel..."):
                norm_df, norm_path = normalize_to_excel(uploaded_file)
            
            st.markdown("#### 1. Normalized Uploaded Report (Excel Preview)")
            st.dataframe(norm_df.head(10), use_container_width=True)
            
            report_text = " ".join(norm_df.astype(str).values.flatten())
            
            st.markdown(f"#### 2. Selected FSSAI Checklist: `{selected_cat}`")
            chk_df = pd.DataFrame(checklist_data)
            st.dataframe(chk_df[['item_id', 'question', 'section', 'weight', 'asterisk_critical']], use_container_width=True)
            
            with st.spinner("3. Running Classical NLP Embedding Engine..."):
                verdicts = run_classical_nlp_engine(checklist_data, report_text, threshold=threshold_slider)
                scoring_results = compute_fssai_scores(checklist_data, verdicts)
                
            st.markdown("---")
            b_col1, b_col2, b_col3, b_col4 = st.columns(4)
            with b_col1:
                st.metric("Total Score", f"{scoring_results['total_awarded']} / {scoring_results['total_max']}")
            with b_col2:
                st.metric("Percentage", f"{scoring_results['percentage']}%")
            with b_col3:
                badge_color = "🟢" if "A" in scoring_results['grade'] else "🟠" if scoring_results['grade'] == "B" else "🔴"
                st.metric("FSSAI Official Grade", f"{badge_color} {scoring_results['grade']}")
            with b_col4:
                st.metric("Status", scoring_results['status_label'])
                
            sec_dict = scoring_results['section_breakdown']
            sec_df = pd.DataFrame([
                {
                    'Section': sec,
                    'Awarded Points': data['awarded'],
                    'Max Points': data['max'],
                    'Compliance %': round((data['awarded'] / data['max'] * 100), 1) if data['max'] > 0 else 0
                } for sec, data in sec_dict.items()
            ])
            fig = px.bar(sec_df, x='Section', y=['Awarded Points', 'Max Points'], barmode='group',
                         title="Section-by-Section Score Breakdown", color_discrete_sequence=['#2ecc71', '#bdc3c7'])
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 4. Verification Results Table")
            res_df = pd.DataFrame(scoring_results['scored_items'])
            
            def color_rows(val):
                if val == 'C':
                    return 'background-color: #d4edda; color: #155724;'
                elif val == 'PC':
                    return 'background-color: #fff3cd; color: #856404;'
                elif val == 'NC':
                    return 'background-color: #f8d7da; color: #721c24;'
                else:
                    return 'background-color: #e2e3e5; color: #383d41;'
                    
            styled_df = res_df.style.applymap(color_rows, subset=['status'])
            st.dataframe(styled_df, use_container_width=True)
            
            excel_bytes = export_audit_to_excel(chk_df, norm_df, res_df, scoring_results)
            st.download_button(
                label="📥 Download Results (Excel Workbook with Input & Output Sheets)",
                data=excel_bytes,
                file_name=f"VERA_NLP_{selected_cat.replace(' ', '_')}_Results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# ================= TAB 2: PERFORMANCE =================
with tab_perf:
    st.subheader("Classical NLP Benchmark Performance Metrics")
    st.markdown("""
    Evaluated on **16 ground-truth reports** (2 per category) from the synthetic FSSAI benchmark dataset.
    Ground truth labels: `C` / `PC` / `NA` → collapsed to **Compliant**, `NC` → **Non-Compliant** for binary evaluation.
    Architecture limitation: **no PC prediction** (Binary C/NC only).
    """)
    
    with st.spinner("Computing benchmark metrics across 16 ground-truth reports..."):
        y_true, y_pred, sweep_data, cat_acc_rows = run_perf_benchmark()
    
    if y_true:
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        overall_acc = accuracy_score(y_true, y_pred)
        
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            st.metric("Accuracy (Threshold 0.72)", f"{overall_acc * 100:.1f}%")
        with p_col2:
            st.metric("Weighted Precision", f"{p * 100:.1f}%")
        with p_col3:
            st.metric("Weighted Recall", f"{r * 100:.1f}%")
        with p_col4:
            st.metric("Weighted F1 Score", f"{f1 * 100:.1f}%")
        
        st.markdown("#### Similarity-Threshold Sweep Curve")
        st.caption("Shows how binary accuracy changes as the cosine similarity cutoff varies across thresholds.")
        sweep_df = pd.DataFrame(sweep_data)
        fig_sweep = px.line(
            sweep_df, x='Threshold', y='Accuracy', markers=True,
            title="Cutoff Sweep: Accuracy vs. Cosine Similarity Threshold",
            labels={'Accuracy': 'Accuracy (%)'},
            color_discrete_sequence=['#2ecc71']
        )
        fig_sweep.add_vline(x=0.72, line_dash="dash", line_color="red", annotation_text="Chosen Cutoff (0.72)")
        fig_sweep.update_layout(yaxis_range=[50, 100])
        st.plotly_chart(fig_sweep, use_container_width=True)
        
        if cat_acc_rows:
            st.markdown("#### Per-Category Accuracy at Threshold 0.72")
            cat_df = pd.DataFrame(cat_acc_rows).sort_values('Accuracy (%)', ascending=False)
            fig_cat = px.bar(
                cat_df, x='Category', y='Accuracy (%)',
                text='Accuracy (%)',
                color='Accuracy (%)',
                color_continuous_scale='RdYlGn',
                title="Per-Category Verification Accuracy"
            )
            fig_cat.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_cat.update_layout(yaxis_range=[0, 105], coloraxis_showscale=False)
            st.plotly_chart(fig_cat, use_container_width=True)

        st.info("⚠️ **Architecture Limitation**: Classical NLP cannot distinguish Partial Compliance from full Compliance. "
                "All predictions are binary (C or NC). Items that should be PC may be misclassified as C or NC, "
                "contributing to accuracy variance. See Architecture 2 (NLI) for 3-class support.")
    else:
        st.warning("Dataset not found. Please verify `data/sample_reports/synthetic_100_reports.json` exists.")

# ================= TAB 3: EXAMPLE =================
with tab_example:
    st.subheader("Pre-Loaded Example Walkthrough")
    st.markdown("""
    **Report: Report_001 | General Manufacturing | Clean Profile**
    
    This is a pre-computed verification of the first benchmark report using Architecture 1 (Classical NLP).
    The report describes a General Manufacturing facility audited by Siddharth Varma in Kurnool, Andhra Pradesh.
    """)
    
    dataset = load_dataset()
    if dataset:
        ex_rep = dataset[0]
        ex_chk = load_checklist("General Manufacturing")
        
        with st.spinner("Running Classical NLP on example report..."):
            ex_verdicts = run_classical_nlp_engine(ex_chk, ex_rep['report_text'], threshold=0.55)
            ex_scores = compute_fssai_scores(ex_chk, ex_verdicts)
        
        e_col1, e_col2, e_col3, e_col4 = st.columns(4)
        with e_col1:
            st.metric("Predicted Score", f"{ex_scores['total_awarded']} / {ex_scores['total_max']}")
        with e_col2:
            st.metric("Predicted %", f"{ex_scores['percentage']}%")
        with e_col3:
            st.metric("Predicted Grade", ex_scores['grade'])
        with e_col4:
            gt_pct = ex_rep['percentage']
            st.metric("Ground Truth %", f"{gt_pct}%", delta=f"{round(ex_scores['percentage']-gt_pct, 1)}% vs GT")
        
        st.markdown("#### Report Excerpt (first 600 characters)")
        st.info(ex_rep['report_text'][:600] + "...")
        
        st.markdown("#### Item-Level Verdict Table")
        res_df = pd.DataFrame(ex_scores['scored_items'])
        gt_ak = ex_rep['answer_key']
        res_df['GT Status'] = res_df['item_id'].apply(lambda x: gt_ak.get(x, {}).get('status', '?'))
        res_df['Match'] = res_df.apply(lambda row: '✅' if row['status'] == row['GT Status'] or 
                                        (row['GT Status'] in ['C','PC','NA'] and row['status'] == 'C') else '❌', axis=1)
        
        def color_rows(val):
            if val == 'C':
                return 'background-color: #d4edda; color: #155724;'
            elif val == 'PC':
                return 'background-color: #fff3cd; color: #856404;'
            elif val == 'NC':
                return 'background-color: #f8d7da; color: #721c24;'
            else:
                return 'background-color: #e2e3e5; color: #383d41;'
        
        display_df = res_df[['item_id', 'question', 'section', 'status', 'GT Status', 'Match', 'points_awarded', 'evidence']]
        styled = display_df.style.applymap(color_rows, subset=['status', 'GT Status'])
        st.dataframe(styled, use_container_width=True)
        
        # Summary chart
        match_pct = res_df['Match'].value_counts()
        st.markdown("#### Prediction Match Summary")
        fig_match = px.pie(
            values=match_pct.values,
            names=match_pct.index,
            color=match_pct.index,
            color_discrete_map={'✅': '#2ecc71', '❌': '#e74c3c'},
            title="Item-Level Prediction vs. Ground Truth"
        )
        st.plotly_chart(fig_match, use_container_width=True)
