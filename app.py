import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Personality Lens",
    page_icon="🧠",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

:root {
    --bg:       #f7f6f3;
    --surface:  #ffffff;
    --border:   #e8e6e0;
    --text:     #5E7AC4;
    --muted:    #8a8780;
    --accent:   #2d2d2a;
    --intro:    #4a6fa5;
    --extro:    #c2622d;
    --intro-bg: #eef2f8;
    --extro-bg: #fdf1eb;
    --radius:   14px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.main .block-container {
    padding: 3rem 2rem 5rem;
    max-width: 680px;
}

#MainMenu, footer, header, .stDeployButton { visibility: hidden; }

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 2.5rem 0 2rem;
}
.app-eyebrow {
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.75rem;
}
.app-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    font-weight: 400;
    line-height: 1.15;
    color: var(--text);
    margin: 0 0 0.75rem;
}
.app-subtitle {
    font-size: 0.9rem;
    color: var(--muted);
    line-height: 1.6;
    max-width: 400px;
    margin: 0 auto;
}

/* ── Section label ── */
.section-label {
    font-size: 0.65rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--muted);
    margin: 1.6rem 0 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

/* ── Sliders ── */
div[data-testid="stSlider"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: var(--text) !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}

/* ── Selectboxes ── */
div[data-testid="stSelectbox"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: var(--text) !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
}

/* ── Button ── */
div[data-testid="stButton"] > button {
    width: 100%;
    background: var(--text) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #f7f6f3 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    padding: 0.85rem !important;
    margin-top: 1.2rem;
    transition: opacity 0.2s;
}
div[data-testid="stButton"] > button:hover { opacity: 0.82; }

/* ── Result ── */
.result-card {
    border-radius: var(--radius);
    padding: 1.8rem 2rem;
    margin: 1.5rem 0 1rem;
    border: 1px solid;
}
.result-card.introvert { background: var(--intro-bg); border-color: #c2cfe0; }
.result-card.extrovert { background: var(--extro-bg); border-color: #e8c4ae; }

.result-label {
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.result-name {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    font-weight: 400;
    margin: 0 0 0.2rem;
    line-height: 1.1;
}
.result-name.introvert { color: var(--intro); }
.result-name.extrovert { color: var(--extro); }

.result-conf {
    font-size: 0.8rem;
    color: var(--muted);
    margin-bottom: 1rem;
}

.conf-track {
    width: 100%;
    height: 3px;
    background: rgba(0,0,0,0.08);
    border-radius: 2px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 2px;
}
.conf-fill.introvert { background: var(--intro); }
.conf-fill.extrovert { background: var(--extro); }

/* ── Traits ── */
.traits {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 1rem;
}
.trait {
    font-size: 0.75rem;
    font-weight: 500;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    border: 1px solid;
}
.trait.introvert { border-color: #b0bfd4; color: var(--intro); background: #dde5f0; }
.trait.extrovert { border-color: #ddb89a; color: #a04d1f; background: #f6ddd0; }

/* ── Probability rows ── */
.prob-section { margin: 1rem 0; }
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.6rem;
    font-size: 0.82rem;
}
.prob-label { width: 72px; color: var(--muted); }
.prob-track { flex: 1; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
.prob-fill-intro { height: 100%; background: var(--intro); border-radius: 3px; }
.prob-fill-extro  { height: 100%; background: var(--extro);  border-radius: 3px; }
.prob-pct { width: 38px; text-align: right; font-weight: 500; color: var(--text); font-size: 0.8rem; }

/* ── Signals ── */
.signals { margin-top: 0.5rem; }
.signal {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.83rem;
    color: #5a5855;
    line-height: 1.5;
}
.signal:last-child { border-bottom: none; }
.signal-dot {
    margin-top: 6px;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    flex-shrink: 0;
}
.signal-dot.introvert { background: var(--intro); }
.signal-dot.extrovert { background: var(--extro); }

/* ── Divider ── */
.divider {
    height: 1px;
    background: var(--border);
    margin: 1.4rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = pickle.load(open("personality_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    le     = pickle.load(open("label_encoder.pkl", "rb"))
    try:
        feat_cols = pickle.load(open("feature_columns.pkl", "rb"))
    except FileNotFoundError:
        feat_cols = ['Time_spent_Alone','Stage_fear','Social_event_attendance',
                     'Going_outside','Drained_after_socializing','Friends_circle_size','Post_frequency']
    return model, scaler, le, feat_cols

try:
    model, scaler, le, feat_cols = load_artifacts()
    artifacts_ok = True
except Exception as e:
    artifacts_ok = False
    artifact_err = str(e)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <p class="app-eyebrow">Personality Lens</p>
    <h1 class="app-title">Who are you,<br>really?</h1>
    <p class="app-subtitle">Answer a few questions and we'll predict your personality.</p>
</div>
""", unsafe_allow_html=True)

if not artifacts_ok:
    st.error(f"Could not load model files: {artifact_err}. Run MODEL.py first.")
    st.stop()

# ── Inputs ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Daily habits</p>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    time_alone    = st.slider("Hours alone per day", 0, 11, 5)
    social_events = st.slider("Social events per week", 0, 10, 3)
with c2:
    going_out     = st.slider("Days out per week", 0, 10, 4)
    friends       = st.slider("Size of friend circle", 0, 20, 8)

st.markdown('<p class="section-label">Social energy</p>', unsafe_allow_html=True)

c3, c4, c5 = st.columns(3)
with c3:
    posts      = st.slider("Posts per week", 0, 10, 3)
with c4:
    stage_fear = st.selectbox("Stage fright?", ["No", "Yes"])
with c5:
    drained    = st.selectbox("Drained after socialising?", ["No", "Yes"])

predict_btn = st.button("Analyse my personality")

# ── Prediction ─────────────────────────────────────────────────────────────────
if predict_btn:
    sf = 1 if stage_fear == "Yes" else 0
    dr = 1 if drained    == "Yes" else 0

    social_score     = social_events + going_out
    alone_ratio      = time_alone / (social_score + 1)
    network_activity = friends * posts

    base = {
        'Time_spent_Alone'          : time_alone,
        'Stage_fear'                : sf,
        'Social_event_attendance'   : social_events,
        'Going_outside'             : going_out,
        'Drained_after_socializing' : dr,
        'Friends_circle_size'       : friends,
        'Post_frequency'            : posts,
        'Social_score'              : social_score,
        'Alone_ratio'               : alone_ratio,
        'Network_activity'          : network_activity,
    }

    input_arr  = np.array([[base[c] for c in feat_cols]])
    input_sc   = scaler.transform(input_arr)
    prediction = model.predict(input_sc)
    proba      = model.predict_proba(input_sc)[0]
    result     = le.inverse_transform(prediction)[0]
    confidence = max(proba) * 100

    is_intro = result == "Introvert"
    cls      = "introvert" if is_intro else "extrovert"

    classes = le.classes_
    extro_p = proba[list(classes).index("Extrovert")] * 100
    intro_p = proba[list(classes).index("Introvert")] * 100

    traits = (
        ["Deep thinker", "Empathetic", "Self-aware", "Reflective", "Independent"]
        if is_intro else
        ["Energetic", "Sociable", "Expressive", "Action-oriented", "Spontaneous"]
    )
    trait_chips = "".join(f'<span class="trait {cls}">{t}</span>' for t in traits)

    st.markdown(f"""
    <div class="result-card {cls}">
        <p class="result-label">Your personality type</p>
        <p class="result-name {cls}">{result}</p>
        <p class="result-conf">{confidence:.0f}% model confidence</p>
        <div class="conf-track">
            <div class="conf-fill {cls}" style="width:{confidence:.1f}%"></div>
        </div>
        <div class="traits">{trait_chips}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-label">Probability breakdown</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="prob-section">
            <div class="prob-row">
                <span class="prob-label">Introvert</span>
                <div class="prob-track"><div class="prob-fill-intro" style="width:{intro_p:.1f}%"></div></div>
                <span class="prob-pct">{intro_p:.0f}%</span>
            </div>
            <div class="prob-row">
                <span class="prob-label">Extrovert</span>
                <div class="prob-track"><div class="prob-fill-extro" style="width:{extro_p:.1f}%"></div></div>
                <span class="prob-pct">{extro_p:.0f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown('<p class="section-label">Key signals</p>', unsafe_allow_html=True)
        signals = []
        if is_intro:
            if time_alone >= 6:   signals.append("High time alone points to solitude preference.")
            if sf == 1:           signals.append("Stage fright is a common introvert trait.")
            if dr == 1:           signals.append("Feeling drained after socialising is a classic introvert sign.")
            if social_events <= 3: signals.append("Low social event attendance supports introversion.")
        else:
            if friends >= 10:     signals.append("A large friend circle is a strong extrovert signal.")
            if social_events >= 6: signals.append("Frequent social events fit extrovert energy.")
            if going_out >= 6:    signals.append("Going out often reflects an outward-facing personality.")
            if posts >= 6:        signals.append("High post frequency suggests social self-expression.")
        if not signals:
            signals.append("Your profile shows balanced tendencies — the model picked the stronger pattern.")

        chips = "".join(
            f'<div class="signal"><div class="signal-dot {cls}"></div><span>{s}</span></div>'
            for s in signals[:4]
        )
        st.markdown(f'<div class="signals">{chips}</div>', unsafe_allow_html=True)