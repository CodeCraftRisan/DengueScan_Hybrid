import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import plotly.graph_objects as go

# ------------------------------------------------------------------
# 1. PAGE CONFIG — MUST BE FIRST STREAMLIT CALL
# ------------------------------------------------------------------
st.set_page_config(page_title="DengueScan Hybrid  ", page_icon="🦟", layout="wide")

# ------------------------------------------------------------------
# 2. CSS / STYLING
# ------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

.stApp {
    background-color: #07090f;
    background-image:
        radial-gradient(ellipse 100% 60% at 0% 0%,   rgba(185,28,28,0.22)  0%, transparent 55%),
        radial-gradient(ellipse 70%  50% at 100% 0%,  rgba(124,58,237,0.10) 0%, transparent 50%),
        radial-gradient(ellipse 80%  55% at 50% 100%, rgba(220,38,38,0.14)  0%, transparent 60%),
        radial-gradient(ellipse 50%  40% at 100% 60%, rgba(239,68,68,0.07)  0%, transparent 50%),
        linear-gradient(160deg, #0d1020 0%, #080b16 50%, #0d0a14 100%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1020 0%, #0a0b14 100%) !important;
    border-right: 1px solid rgba(220,38,38,0.18) !important;
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

.stApp, .stMarkdown, p, label, div { color: #e2e8f0; }

div[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
}
.stSlider label, .stNumberInput label {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.stCheckbox label { color: #94a3b8 !important; font-size: 0.88rem !important; }

.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #dc2626, #991b1b) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    height: 52px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    box-shadow: 0 4px 24px rgba(220,38,38,0.3) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(220,38,38,0.45) !important;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(220,38,38,0.10);
    border: 1px solid rgba(220,38,38,0.28);
    border-radius: 100px;
    padding: 5px 16px;
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #f87171 !important;
    margin-bottom: 14px;
    font-family: 'Outfit', sans-serif;
}
.hero-title {
    font-family: 'DM Serif Display', Georgia, serif !important;
    font-size: clamp(2.4rem, 5vw, 3.8rem);
    font-weight: 400;
    line-height: 1.05;
    color: #f8fafc !important;
    letter-spacing: -0.02em;
    margin: 0 0 10px 0;
    text-shadow: 0 0 80px rgba(220,38,38,0.2);
}
.hero-title span {
    background: linear-gradient(125deg, #ff6b6b 0%, #dc2626 40%, #ff4444 70%, #fca5a5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-style: italic;
}
.hero-sub {
    font-size: 0.92rem;
    color: #4a5568 !important;
    font-weight: 400;
    font-family: 'Outfit', sans-serif;
    line-height: 1.6;
}

.stats-bar {
    display: flex;
    gap: 24px;
    padding: 18px 28px;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    margin: 18px 0 22px 0;
    position: relative;
    overflow: hidden;
}
.stats-bar::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(220,38,38,0.5), rgba(239,68,68,0.3), transparent);
}
.stat-item  { display: flex; flex-direction: column; gap: 3px; }
.stat-value {
    font-size: 1.5rem; font-weight: 700;
    background: linear-gradient(135deg, #ff6b6b, #dc2626);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    font-family: 'JetBrains Mono', monospace; line-height: 1;
}
.stat-label {
    font-size: 0.68rem; color: #4b5563 !important;
    text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;
}
.stat-divider {
    width: 1px;
    background: linear-gradient(180deg, transparent, rgba(255,255,255,0.1), transparent);
    align-self: stretch;
}

.card-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 20px; padding-bottom: 14px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.card-icon {
    width: 36px; height: 36px;
    background: rgba(220,38,38,0.15); border-radius: 8px;
    display: flex; align-items: center; justify-content: center; font-size: 18px;
}
.card-title    { font-size: 0.95rem; font-weight: 600; color: #f1f5f9 !important; }
.card-subtitle { font-size: 0.7rem; color: #475569 !important; text-transform: uppercase; letter-spacing: 0.08em; }

div[data-testid="stAlert"] { border-radius: 12px !important; border-left-width: 3px !important; }
.stRadio > label { font-size: 0.75rem !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.1em; color: #374151 !important; }

#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 3. LOAD MODELS
# ------------------------------------------------------------------
@st.cache_resource
def load_models():
    lab_model    = joblib.load('dengue_model_final.pkl')
    lab_features = joblib.load('model_features.pkl')
    try:
        clinical_model = joblib.load('dengue_model_clinical.pkl')
    except Exception:
        clinical_model = None
    return lab_model, lab_features, clinical_model

try:
    lab_model, lab_feat_names, clinical_model = load_models()
except Exception as e:
    st.error(f"⚠️ Model files not found: {e}\n\nPlease put `dengue_model_final.pkl`, `model_features.pkl`, and `dengue_model_clinical.pkl` in the same folder as this script.")
    st.stop()

# ------------------------------------------------------------------
# 4. SIDEBAR
# ------------------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align:center; padding:8px 0 16px 0;">
    <div style="font-size:2.5rem; margin-bottom:4px;">🦟</div>
    <div style="font-size:0.65rem; text-transform:uppercase; letter-spacing:0.15em; color:#475569; font-weight:700;">DengueScan Hybrid</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="background:rgba(220,38,38,0.08); border:1px solid rgba(220,38,38,0.2); border-radius:10px; padding:14px; margin-bottom:16px;">
    <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; color:#f87171; font-weight:700; margin-bottom:8px;">System Architecture</div>
    <div style="font-size:0.82rem; color:#94a3b8; line-height:1.7;">
        <div>① Symptom Screening &nbsp;<span style="color:#f87171; font-weight:700;">96% Recall</span></div>
        <div>② Lab Confirmation &nbsp;<span style="color:#f87171; font-weight:700;">83% Accuracy</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

mode = st.sidebar.radio(
    "Input Mode",
    ["Hybrid (Symptoms + Lab)", "Symptoms Only (Screening)", "Lab Report Only (Hospital)"]
)

# ------------------------------------------------------------------
# 5. HERO HEADER
# ------------------------------------------------------------------
st.markdown("""
<div class="hero-badge">🔬 Clinical Decision Support System</div>
<h1 class="hero-title">Dengue<span>Scan</span> Hybrid</h1>
<p class="hero-sub">Tiered multi-stage diagnostic intelligence — optimized for Bangladesh clinical context</p>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(124,58,237,0.08) 0%, rgba(220,38,38,0.10) 50%, rgba(124,58,237,0.06) 100%);
    border: 1px solid rgba(124,58,237,0.22);
    border-left: 3px solid #7c3aed;
    border-radius: 12px;
    padding: 14px 20px;
    margin: 14px 0 6px 0;
    display: flex;
    align-items: flex-start;
    gap: 14px;
">
    <div style="background:rgba(124,58,237,0.15); border-radius:8px; padding:8px 10px; font-size:1.1rem; flex-shrink:0; line-height:1;">📄</div>
    <div>
        <div style="font-size:0.7rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#a78bfa; margin-bottom:4px;">Research Prototype — Under Academic Review</div>
        <div style="font-size:0.82rem; color:#64748b; line-height:1.55;">
            Developed as part of a supervised research project on
            <span style="color:#94a3b8; font-weight:500;">Hybrid Dengue diagnosis in low-resource settings</span>.
            All outputs are probabilistic estimates — not clinical verdicts.
            <span style="color:#a78bfa; font-weight:500;">Do not use for treatment decisions without physician oversight.</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stats-bar">
    <div class="stat-item"><div class="stat-value">96%</div><div class="stat-label">Symptom Recall</div></div>
    <div class="stat-divider"></div>
    <div class="stat-item"><div class="stat-value">83%</div><div class="stat-label">Lab Accuracy</div></div>
    <div class="stat-divider"></div>
    <div class="stat-item"><div class="stat-value">2</div><div class="stat-label"> Stages</div></div>
    <div class="stat-divider"></div>
    <div class="stat-item"><div class="stat-value">BD</div><div class="stat-label">Context</div></div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 6. INPUT SECTIONS — LOGIC UNCHANGED
# ------------------------------------------------------------------
col1, col2 = st.columns([1, 1], gap="large")

symptoms_data = {}
if mode in ["Hybrid (Symptoms + Lab)", "Symptoms Only (Screening)"]:
    with col1:
        st.markdown("""
        <div class="card-header">
            <div class="card-icon">🌡️</div>
            <div><div class="card-title">Clinical Symptoms</div><div class="card-subtitle">Patient-reported findings</div></div>
        </div>
        """, unsafe_allow_html=True)
        fever    = st.checkbox("Fever — High Temperature")
        headache = st.checkbox("Severe Headache")
        vomiting = st.checkbox("Vomiting / Nausea")
        pain     = st.checkbox("Muscle / Joint Pain")
        rash     = st.checkbox("Skin Rash")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        days = st.slider("Duration of Symptoms (Days)", 1, 10, 3)
        symptoms_data = {
            'Fever': int(fever), 'Duration_of_Fever': days,
            'Headache': int(headache), 'Muscle_Pain': int(pain),
            'Rash': int(rash), 'Vomiting': int(vomiting)
        }

lab_data = {}
if mode in ["Hybrid (Symptoms + Lab)", "Lab Report Only (Hospital)"]:
    with col2:
        st.markdown("""
        <div class="card-header">
            <div class="card-icon">🩸</div>
            <div><div class="card-title">Hematological Report</div><div class="card-subtitle">Laboratory CBC values</div></div>
        </div>
        """, unsafe_allow_html=True)
        age       = st.number_input("Age (Years)", 1, 120, 25)
        platelets = st.number_input("Platelet Count (per μL)", 1000, 1000000, 150000, step=5000)
        wbc       = st.number_input("WBC Count (per μL)", 1000, 50000, 5000, step=100)
        lab_data  = pd.DataFrame([[age, platelets, wbc]], columns=lab_feat_names)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 7. PREDICTION LOGIC — UNCHANGED
# ------------------------------------------------------------------
if st.button("⚡ Analyze Risk Now"):

    score_clinical = None
    score_lab      = None
    final_score    = 0.0

    if mode != "Lab Report Only (Hospital)" and clinical_model:
        df_sym = pd.DataFrame([symptoms_data])
        score_clinical = clinical_model.predict_proba(df_sym)[0][1]

    if mode != "Symptoms Only (Screening)":
        score_lab = lab_model.predict_proba(lab_data)[0][1]

    if mode == "Hybrid (Symptoms + Lab)":
        final_score = (score_clinical * 0.3) + (score_lab * 0.7)
        threshold   = 0.35
    elif mode == "Symptoms Only (Screening)":
        final_score = score_clinical
        threshold   = 0.25
    else:
        final_score = score_lab
        threshold   = 0.40

    is_positive = final_score >= threshold
    bar_color   = "#f87171" if is_positive else "#4ade80"

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = final_score * 100,
        domain= {'x': [0,1], 'y': [0,1]},
        title = {'text': "Dengue Risk Score", 'font': {'color':'#94a3b8','size':14,'family':'Outfit'}},
        number= {'suffix':'%', 'font': {'color':'#f1f5f9','size':40,'family':'JetBrains Mono'}},
        gauge = {
            'axis'       : {'range':[0,100], 'tickcolor':'#334155', 'tickfont':{'color':'#475569','size':11}},
            'bar'        : {'color': bar_color, 'thickness': 0.25},
            'bgcolor'    : 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range':[0, threshold*100],  'color':'rgba(22,163,74,0.12)'},
                {'range':[threshold*100, 100],'color':'rgba(220,38,38,0.12)'}
            ],
            'threshold': {'line':{'color':'#475569','width':2},'thickness':0.75,'value':threshold*100}
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_family='Outfit', height=280,
        margin=dict(t=40, b=10, l=20, r=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    if is_positive:
        st.error(f"### ⚠️ POSITIVE — High Risk Detected")
        st.write(f"Confidence: **{final_score:.1%}** | Threshold: `{threshold}`")
        if mode == "Symptoms Only (Screening)":
            st.warning("**Recommended Action:** Immediate CBC / NS1 Test required.")
        else:
            st.warning("**Recommended Action:** Clinical management protocol initiated.")
    else:
        st.success(f"### ✅ NEGATIVE — Low Risk")
        st.write(f"Confidence: **{final_score:.1%}** | Threshold: `{threshold}`")
        st.info("**Recommended Action:** Monitor symptoms. Re-evaluate if condition changes.")

# ------------------------------------------------------------------
# 8. FOOTER
# ------------------------------------------------------------------
st.markdown("""
<div style="
    margin-top: 40px; padding: 20px 24px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-top: 1px solid rgba(124,58,237,0.18);
    border-radius: 12px;
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;
">
    <div>
        <div style="font-family:'DM Serif Display',serif; font-size:1rem; color:#e2e8f0; font-style:italic; margin-bottom:3px;">
            DengueScan Hybrid
            <span style="font-style:normal; font-family:'Outfit',sans-serif; font-size:0.7rem; color:#334155; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; margin-left:10px;">v1.0 Research Build</span>
        </div>
        <div style="font-size:0.72rem; color:#374151; font-family:'Outfit',sans-serif; letter-spacing:0.02em;">
            Optimized for Bangladesh clinical context · XGBoost ensemble model
        </div>
    </div>
    <div style="display:flex; gap:14px; align-items:center; flex-wrap:wrap;">
        <span style="font-size:0.68rem; font-family:'Outfit',sans-serif; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:#a78bfa; background:rgba(124,58,237,0.10); border:1px solid rgba(124,58,237,0.22); padding:4px 12px; border-radius:100px;">
            ⚠️ Not for Clinical Use
        </span>
        <span style="font-size:0.7rem; color:#1f2937; font-family:'JetBrains Mono',monospace;">© 2026 Research Prototype</span>
    </div>
</div>
""", unsafe_allow_html=True)
