import streamlit as st
import json
import os
import urllib.request
import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title="VERA-Compare | FSSAI Audit Verification Benchmark",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.15rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Check Ollama Connectivity
@st.cache_data(ttl=60)
def check_ollama():
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as resp:
            data = json.loads(resp.read().decode())
            models = [m['name'] for m in data.get('models', [])]
            return True, models
    except Exception as e:
        return False, str(e)

ollama_ok, ollama_info = check_ollama()

st.markdown('<div class="main-title">VERA-Compare</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">A Unified Multi-Architecture Benchmark for Automated FSSAI Checklist Verification</div>', unsafe_allow_html=True)

# Header Alert for Services
col_status1, col_status2 = st.columns([2, 1])
with col_status1:
    if ollama_ok:
        st.success(f"🟢 **Ollama Local Engine Connected** | Active Models: `{', '.join(ollama_info)}`")
    else:
        st.error(f"🔴 **Ollama Unreachable**: {ollama_info}. Please run `ollama serve`.")
with col_status2:
    st.info("📋 **Annex 5 Standard**: 8 Checklists | Official C/PC/NC/NA Scoring")

st.markdown("---")

st.markdown("""
### Benchmark Architectures Overview
VERA-Compare benchmarks **four independent AI architectures** on identical official FSSAI inspection checklists:

1. **[1_Classical_NLP](./1_Classical_NLP)**: Dense Embedding Cosine Matching (`nomic-embed-text`) with threshold sweeps. (Binary C/NC limitation documented).
2. **[2_Deep_Learning_NLI](./2_Deep_Learning_NLI)**: Cross-Encoder Natural Language Inference (`nli-deberta-v3-small`) mapping premises and hypotheses into C, PC, and NC.
3. **[3_RAG](./3_RAG)**: Retrieval-Augmented Generation featuring `phi4-mini`, verbatim evidence citations, hallucination detection, and an interactive **RAG Auditor Bot**.
4. **[4_OKF](./4_OKF)**: Open Knowledge Formulation knowledge-graph traversal (`index.md → [Section].md → [Item].md`) with verifiable breadcrumbs and structural reasoning.

---
""")

# Load dataset summary
DATASET_PATH = "data/sample_reports/synthetic_100_reports.json"
if os.path.exists(DATASET_PATH):
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    df_reports = pd.DataFrame([
        {
            'Report ID': r['report_id'],
            'Category': r['category'],
            'Facility Name': r['facility_name'],
            'Auditor': r['auditor_name'],
            'Location': r['location'],
            'Profile': r['compliance_profile'],
            'Ground Truth Score': f"{r['total_score']} / {r['total_max']}",
            'Percentage': f"{r['percentage']}%",
            'Grade': r['grade']
        } for r in dataset
    ])
    
    st.subheader(f"📊 Synthetic Evaluation Dataset ({len(dataset)} Ground-Truth Reports)")
    st.markdown("""
    This dataset enforces strict bias controls across 8 official FSSAI sectors:
    - Balanced distribution: 12–13 reports per category
    - Profiles: Clean (~25%), Moderate (~35%), Weak (~25%), and Clustered-Failure (~15%)
    - Independent randomized failure points, facility locations, and writing styles.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Ground Truth Reports", len(dataset))
    with col2:
        st.metric("FSSAI Checklist Categories", len(df_reports['Category'].unique()))
    with col3:
        st.metric("Avg Score Across Set", f"{round(float(pd.Series([r['percentage'] for r in dataset]).mean()), 1)}%")
    with col4:
        st.metric("Clean Profiles", f"{len([r for r in dataset if r['compliance_profile'] == 'Clean'])} Reports")
        
    st.dataframe(df_reports.head(15), use_container_width=True)
else:
    st.warning("Synthetic dataset not yet generated. Please execute `python generate_dataset.py`.")

st.markdown("""
---
*Select a verification architecture from the left sidebar to begin verification, audit exploration, and benchmark comparisons.*
""")
