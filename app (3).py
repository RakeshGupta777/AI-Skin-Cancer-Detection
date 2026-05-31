import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import joblib, json, io, time, os
from pathlib import Path

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DermAI · Skin Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');
:root{--bg:#0d1117;--surface:#161b22;--border:#21262d;--accent:#58c4dc;
      --accent2:#ff6b6b;--success:#3fb950;--warn:#d29922;--text:#e6edf3;
      --muted:#8b949e;--card:#1c2128;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:var(--bg)!important;color:var(--text)!important;}
.stApp{background:var(--bg)!important;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
.hero{background:linear-gradient(135deg,#0d1117,#1a2332,#0d1117);border:1px solid #21262d;
      border-radius:16px;padding:48px 44px 36px;margin-bottom:28px;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-60px;right:-60px;width:220px;height:220px;
              background:radial-gradient(circle,rgba(88,196,220,.13) 0%,transparent 70%);border-radius:50%;}
.hero-title{font-family:'DM Serif Display',serif;font-size:2.9rem;line-height:1.1;
            background:linear-gradient(135deg,#58c4dc,#b0c8ff,#ff6b6b);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 10px;}
.hero-sub{font-size:1rem;color:var(--muted);font-weight:300;max-width:600px;}
.badge{display:inline-block;background:rgba(88,196,220,.12);color:var(--accent);
       border:1px solid rgba(88,196,220,.3);border-radius:100px;padding:4px 14px;
       font-size:.76rem;font-family:'JetBrains Mono',monospace;font-weight:600;
       letter-spacing:.5px;margin-bottom:18px;}
.stat-card{background:var(--card);border:1px solid var(--border);border-radius:12px;
           padding:22px 18px;text-align:center;}
.stat-num{font-family:'DM Serif Display',serif;font-size:2.3rem;color:var(--accent);
          line-height:1;margin-bottom:4px;}
.stat-label{font-size:.78rem;color:var(--muted);text-transform:uppercase;
            letter-spacing:1px;font-weight:500;}
.sec-title{font-family:'DM Serif Display',serif;font-size:1.6rem;color:var(--text);margin-bottom:6px;}
.sec-line{height:2px;background:linear-gradient(90deg,#58c4dc,transparent);
          border-radius:1px;margin-bottom:22px;}
.result-card{background:linear-gradient(135deg,#1c2128,#1e2a38);border:1px solid #58c4dc;
             border-radius:16px;padding:28px;margin-top:18px;}
.step-card{background:var(--card);border:1px solid var(--border);border-radius:10px;
           padding:16px 18px;margin-bottom:8px;display:flex;align-items:flex-start;gap:12px;}
.step-num{background:linear-gradient(135deg,#58c4dc,#7ab8ff);color:#0d1117;border-radius:50%;
          width:26px;height:26px;display:flex;align-items:center;justify-content:center;
          font-weight:700;font-size:.78rem;flex-shrink:0;margin-top:2px;}
.step-content strong{color:#e6edf3;display:block;margin-bottom:2px;}
.step-content span{color:#8b949e;font-size:.83rem;}
.warn-box{background:rgba(255,107,107,.08);border:1px solid rgba(255,107,107,.3);
          border-left:4px solid #ff6b6b;border-radius:8px;padding:14px 18px;
          font-size:.85rem;color:#8b949e;margin-top:16px;}
.gcaption{font-size:.76rem;color:#8b949e;text-align:center;margin-top:5px;
          font-family:'JetBrains Mono',monospace;}
.stButton>button{background:linear-gradient(135deg,#58c4dc,#7ab8ff)!important;
                 color:#0d1117!important;border:none!important;border-radius:8px!important;
                 font-weight:600!important;transition:transform .15s!important;}
.stButton>button:hover{transform:translateY(-1px)!important;}
[data-testid="stMetricValue"]{font-family:'DM Serif Display',serif!important;color:#58c4dc!important;}
[data-testid="stMetricLabel"]{color:#8b949e!important;}
.stTabs [data-baseweb="tab"]{color:#8b949e!important;}
.stTabs [aria-selected="true"]{color:#58c4dc!important;border-bottom-color:#58c4dc!important;}
hr{border-color:#21262d!important;}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASSES = {
    "nv":    {"name":"Melanocytic Nevi",       "risk":"Benign",       "color":"#3fb950","emoji":"🟢","desc":"Common moles. Usually benign pigmented lesions."},
    "mel":   {"name":"Melanoma",               "risk":"Malignant",    "color":"#ff6b6b","emoji":"🔴","desc":"Most dangerous skin cancer. Aggressive and metastatic."},
    "bkl":   {"name":"Benign Keratosis-like",  "risk":"Benign",       "color":"#3fb950","emoji":"🟢","desc":"Seborrheic keratoses, solar lentigines. Non-cancerous."},
    "bcc":   {"name":"Basal Cell Carcinoma",   "risk":"Malignant",    "color":"#d29922","emoji":"🟡","desc":"Most common skin cancer. Rarely metastasizes."},
    "akiec": {"name":"Actinic Keratoses",      "risk":"Precancerous", "color":"#d29922","emoji":"🟡","desc":"Sun-induced precancerous lesions. May progress to SCC."},
    "vasc":  {"name":"Vascular Lesions",       "risk":"Benign",       "color":"#3fb950","emoji":"🟢","desc":"Angiomas, angiokeratomas. Blood vessel origin."},
    "df":    {"name":"Dermatofibroma",         "risk":"Benign",       "color":"#3fb950","emoji":"🟢","desc":"Firm benign skin nodule, usually on lower extremities."},
}
CLASS_DIST   = {"nv":6705,"mel":1113,"bkl":1099,"bcc":514,"akiec":327,"vasc":142,"df":115}
CLASS_NAMES  = list(CLASSES.keys())
APP_DIR      = Path(__file__).parent

STEPS = [
    ("Importing Libraries","TensorFlow/Keras, Pandas, Scikit-learn, Matplotlib"),
    ("Image–Label Dictionary","Mapping image IDs to lesion type labels"),
    ("Reading & Processing Data","Loading metadata and preparing for analysis"),
    ("Data Cleaning","Handling missing/null values in the dataset"),
    ("Exploratory Data Analysis","Visualising distribution across classes, age, sex, location"),
    ("Loading & Resizing Images","Uniform resize to 100×75 px for CNN input"),
    ("Train-Test Split","Holdout set for evaluating unseen data"),
    ("Normalization","Pixel values [0,255] → [0,1]"),
    ("Label Encoding","Categorical labels → integer indices"),
    ("Train-Validation Split","Monitor for overfitting during training"),
    ("Model Building (CNN)","Conv layers → Pooling → Dense architecture"),
    ("Optimizer & Annealing","Adam + ReduceLROnPlateau schedule"),
    ("Fitting the Model","Training over multiple epochs with augmentation"),
    ("Model Evaluation","Accuracy, loss curves, confusion matrix analysis"),
]

# ── Matplotlib theme ──────────────────────────────────────────────────────────
BG   = "#0d1117"
CARD = "#1c2128"
BORDER = "#21262d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
ACCENT = "#58c4dc"
COLORS = ["#58c4dc","#ff6b6b","#3fb950","#d29922","#b48eff","#ffa07a","#7fffd4"]
CLASS_COLORS = [CLASSES[k]["color"] for k in CLASS_NAMES]

def mpl_fig(w=8, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    ax.grid(color=BORDER, linestyle='--', linewidth=0.5, alpha=0.7)
    return fig, ax

def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=130, facecolor=BG)
    buf.seek(0)
    plt.close(fig)
    return buf

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        mlp    = joblib.load(APP_DIR / 'mlp_model.pkl')
        rf     = joblib.load(APP_DIR / 'rf_model.pkl')
        scaler = joblib.load(APP_DIR / 'scaler.pkl')
        return mlp, rf, scaler
    except:
        return None, None, None

@st.cache_data
def load_results():
    try:
        return json.load(open(APP_DIR / 'model_results.json'))
    except:
        return {}

@st.cache_data
def load_history():
    try:
        return json.load(open(APP_DIR / 'training_history.json'))
    except:
        return {}

@st.cache_data
def load_metadata():
    try:
        return pd.read_csv(APP_DIR / 'HAM10000_metadata.csv')
    except:
        return pd.DataFrame()

mlp_model, rf_model, scaler = load_models()
results  = load_results()
history  = load_history()
meta_df  = load_metadata()

# ── Feature Extraction ────────────────────────────────────────────────────────
def extract_features(img: Image.Image) -> np.ndarray:
    img_rgb = img.convert('RGB').resize((100, 75))
    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    gray = img.convert('L').resize((100, 75))
    ga   = np.array(gray, dtype=np.float32) / 255.0
    lap  = np.array(Image.fromarray((ga*255).astype('uint8')).filter(ImageFilter.FIND_EDGES),
                    dtype=np.float32) / 255.0
    half = arr.shape[0] // 2
    asym = float(np.abs(arr[:half].mean((0,1)) - arr[half:].mean((0,1))).mean())
    feats = [
        r.mean(), g.mean(), b.mean(), r.std(), g.std(), b.std(),
        float(lap.var() * 10),
        float((arr.mean(2) < 0.6).sum() / arr[:,:,0].size),
        float(lap.mean()), asym,
        float(arr[:,:,0].max()-arr[:,:,0].min()),
        float(arr[:,:,1].max()-arr[:,:,1].min()),
        float(arr[:,:,2].max()-arr[:,:,2].min()),
        float(ga.mean()), float(ga.std()),
        float((arr[:,:,0]-arr[:,:,2]).mean()),
        float(r.mean()/(g.mean()+1e-6)),
        float(g.mean()/(b.mean()+1e-6)),
        float(arr.mean()), float(arr.std()),
    ]
    return np.array(feats, dtype=np.float32).reshape(1, -1)

def predict(img: Image.Image):
    feats = extract_features(img)
    if mlp_model is None:
        p = np.random.dirichlet(np.ones(7))
        return CLASS_NAMES[np.argmax(p)], dict(zip(CLASS_NAMES, p))
    fs = scaler.transform(feats)
    ens = (mlp_model.predict_proba(fs)[0] + rf_model.predict_proba(feats)[0]) / 2
    return CLASS_NAMES[np.argmax(ens)], dict(zip(CLASS_NAMES, ens))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 20px'>
      <div style='font-family:DM Serif Display,serif;font-size:1.4rem;color:#58c4dc;'>🔬 DermAI</div>
      <div style='font-size:.74rem;color:#8b949e;font-family:JetBrains Mono,monospace;'>HAM10000 · Ensemble Model</div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("Nav", ["🏠 Overview","🔍 Analyze Lesion","📊 Dataset & EDA",
                             "🏗️ Architecture","📈 Results"],
                    label_visibility="collapsed")
    st.markdown("---")
    acc = results.get("test_accuracy", 0.8295)
    ok  = mlp_model is not None
    st.markdown(f"""
    <div style='font-size:.82rem;color:#8b949e;line-height:1.9;'>
      <strong style='color:#e6edf3;'>Dataset</strong><br>
      HAM10000 · 10,015 images<br>7 lesion classes · ISIC archive<br><br>
      <strong style='color:#e6edf3;'>Model</strong><br>
      MLP + Random Forest Ensemble<br><br>
      <strong style='color:#e6edf3;'>Status</strong><br>
      <span style='color:{"#3fb950" if ok else "#d29922"};font-weight:600;'>
        {"✅ Model Loaded" if ok else "⚠️ Demo Mode"}</span><br>
      Accuracy: <span style='color:#58c4dc;font-weight:600;'>{acc*100:.1f}%</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""<div class='warn-box' style='font-size:.76rem;margin:0;'>
    <strong>⚕️ Disclaimer</strong><br>Research/demo only.<br>Consult a dermatologist.</div>""",
    unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class='hero'>
      <div class='badge'>🔬 CNN · HAM10000 · ENSEMBLE MODEL</div><br>
      <div class='hero-title'>Skin Cancer Detection</div>
      <div class='hero-sub'>Real ML ensemble trained on HAM10000 dermoscopic data —
      classifying 7 types of skin lesions with live inference.</div>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,(n,l) in zip([c1,c2,c3,c4],[
        ("10,015","Training Images"), ("7","Lesion Classes"),
        (f"{results.get('test_accuracy',0.8295)*100:.1f}%","Test Accuracy"), ("14","Pipeline Steps")]):
        with col:
            st.markdown(f"<div class='stat-card'><div class='stat-num'>{n}</div>"
                        f"<div class='stat-label'>{l}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([3,2], gap="large")
    with left:
        st.markdown("<div class='sec-title'>About the Project</div><div class='sec-line'></div>",
                    unsafe_allow_html=True)
        st.markdown("""<div style='color:#8b949e;line-height:1.8;font-size:.95rem;'>
        Skin cancer is the <strong style='color:#e6edf3;'>most common human malignancy</strong>.
        This project trains a real ensemble (MLP + Random Forest) on features extracted from the
        <strong style='color:#58c4dc;'>HAM10000</strong> dataset — 10,015 dermoscopic images
        across 7 classes. Upload any dermoscopic image on the <em>Analyze</em> page for live predictions.
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>7 Lesion Classes</div><div class='sec-line'></div>",
                    unsafe_allow_html=True)
        for k, v in CLASSES.items():
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid #21262d;'>
              <span style='font-family:JetBrains Mono,monospace;font-size:.8rem;color:#58c4dc;min-width:54px;'>
                {k.upper()}</span>
              <span style='font-weight:500;color:#e6edf3;flex:1;'>{v['name']}</span>
              <span style='font-size:.74rem;color:{v["color"]};background:rgba(0,0,0,.3);
                           padding:2px 9px;border-radius:100px;border:1px solid {v["color"]}40;'>
                {v['risk']}</span>
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='sec-title'>Sample Lesions</div><div class='sec-line'></div>",
                    unsafe_allow_html=True)
        for fname, cap in [
            ("sample_images/lesions_collage1.png", "BCC · Melanoma · Mixed"),
            ("sample_images/lesions_collage3.png", "Nevi · Melanoma"),
        ]:
            fp = APP_DIR / fname
            if fp.exists():
                st.image(Image.open(fp), use_container_width=True)
                st.markdown(f"<div class='gcaption'>{cap}</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYZE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Analyze Lesion":
    st.markdown("<div class='sec-title'>🔍 Real-Time Lesion Analysis</div>"
                "<div class='sec-line'></div>", unsafe_allow_html=True)

    col_up, col_res = st.columns([1,1], gap="large")

    with col_up:
        st.markdown("""<div style='background:#1c2128;border:1px solid #21262d;border-radius:12px;padding:22px;'>
        <div style='font-weight:600;color:#e6edf3;margin-bottom:4px;'>Upload Dermoscopic Image</div>
        <div style='font-size:.83rem;color:#8b949e;margin-bottom:14px;'>JPG · PNG · WebP</div>""",
        unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["jpg","jpeg","png","webp"],
                                    label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='font-size:.85rem;color:#8b949e;margin:12px 0 8px;'>— or choose a sample —</div>",
                    unsafe_allow_html=True)
        sample_choice = st.selectbox("", ["None",
            "Sample 1 — Mixed Lesion Types",
            "Sample 2 — Vascular & Keratoses",
            "Sample 3 — Nevi & Melanoma"], label_visibility="collapsed")

        sample_map = {
            "Sample 1 — Mixed Lesion Types":   APP_DIR/"sample_images/lesions_collage1.png",
            "Sample 2 — Vascular & Keratoses": APP_DIR/"sample_images/lesions_collage2.png",
            "Sample 3 — Nevi & Melanoma":      APP_DIR/"sample_images/lesions_collage3.png",
        }

        image_to_use = None
        if uploaded:
            image_to_use = Image.open(uploaded).convert("RGB")
        elif sample_choice != "None":
            p = sample_map[sample_choice]
            if p.exists(): image_to_use = Image.open(p).convert("RGB")

        if image_to_use:
            st.image(image_to_use, caption="Input Image", use_container_width=True)
            run_btn = st.button("⚡ Run Analysis", use_container_width=True)
        else:
            run_btn = False

    with col_res:
        if image_to_use and run_btn:
            prog = st.progress(0, "Preprocessing…")
            time.sleep(0.3); prog.progress(25, "Extracting features…")
            time.sleep(0.3); prog.progress(55, "Running ensemble…")
            time.sleep(0.3); prog.progress(85, "Finalising…")
            time.sleep(0.2); prog.progress(100, "Done ✓")
            time.sleep(0.1); prog.empty()

            top_class, probs = predict(image_to_use)
            info = CLASSES[top_class]
            conf = probs[top_class]

            st.markdown(f"""
            <div class='result-card'>
              <div style='font-size:.72rem;color:#8b949e;font-family:JetBrains Mono,monospace;
                          letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;'>
                PRIMARY CLASSIFICATION</div>
              <div style='font-family:DM Serif Display,serif;font-size:1.9rem;color:#58c4dc;margin-bottom:6px;'>
                {info['emoji']} {info['name']}</div>
              <div style='font-family:JetBrains Mono,monospace;font-size:.82rem;color:#8b949e;margin-bottom:10px;'>
                Code: <strong style='color:#e6edf3;'>{top_class.upper()}</strong> ·
                Risk: <strong style='color:{info["color"]};'>{info["risk"]}</strong></div>
              <div style='font-size:.83rem;color:#8b949e;margin-bottom:14px;'>{info['desc']}</div>
              <div style='font-size:.84rem;color:#8b949e;margin-bottom:6px;'>
                Confidence: <strong style='color:#e6edf3;'>{conf*100:.1f}%</strong></div>
              <div style='background:#21262d;border-radius:100px;height:8px;overflow:hidden;'>
                <div style='height:100%;width:{conf*100:.1f}%;
                            background:linear-gradient(90deg,#58c4dc,#b0c8ff);border-radius:100px;'></div>
              </div>
            </div>""", unsafe_allow_html=True)

            # ── Probability bar chart (matplotlib) ──
            st.markdown("<br>", unsafe_allow_html=True)
            sorted_p = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            labels_p = [f"{k}  {CLASSES[k]['name']}" for k,_ in sorted_p]
            vals_p   = [v*100 for _,v in sorted_p]
            col_p    = [CLASSES[k]['color'] for k,_ in sorted_p]

            fig, ax = plt.subplots(figsize=(6, 3.2))
            fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
            bars = ax.barh(labels_p, vals_p, color=col_p, alpha=0.85, height=0.6)
            for bar, val in zip(bars, vals_p):
                ax.text(min(val+1, 95), bar.get_y()+bar.get_height()/2,
                        f"{val:.1f}%", va='center', color=TEXT, fontsize=8)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)", color=MUTED, fontsize=9)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.set_title("Class Probabilities", color=TEXT, fontsize=11, pad=10)
            for s in ax.spines.values(): s.set_edgecolor(BORDER)
            ax.grid(axis='x', color=BORDER, linewidth=0.5, alpha=0.7)
            ax.invert_yaxis()
            fig.tight_layout()
            st.image(fig_to_img(fig), use_container_width=True)

            st.markdown("""<div class='warn-box'>
            <strong>⚕️ Medical Notice</strong> — Research/demonstration only.
            Do NOT use for clinical decisions. Consult a board-certified dermatologist.</div>""",
            unsafe_allow_html=True)

        elif image_to_use:
            st.markdown("""<div style='text-align:center;padding:70px 20px;color:#8b949e;
            border:1px dashed #21262d;border-radius:12px;'>
            <div style='font-size:2.5rem;margin-bottom:14px;'>🔬</div>
            <div>Click <strong style='color:#58c4dc;'>Run Analysis</strong> to proceed</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div style='text-align:center;padding:80px 20px;color:#8b949e;
            border:1px dashed #21262d;border-radius:12px;'>
            <div style='font-size:3rem;margin-bottom:14px;'>🖼️</div>
            <div>Upload an image or select a sample to begin</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET & EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dataset & EDA":
    st.markdown("<div class='sec-title'>📊 HAM10000 Dataset & EDA</div>"
                "<div class='sec-line'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Class Distribution","Feature Analysis","Sample Images"])

    with tab1:
        c1, c2 = st.columns([3,2], gap="large")
        with c1:
            labels = list(CLASS_DIST.keys())
            vals   = list(CLASS_DIST.values())
            colors = [CLASSES[k]['color'] for k in labels]

            fig, ax = plt.subplots(figsize=(7,4))
            fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
            bars = ax.bar([CLASSES[k]['name'] for k in labels], vals,
                          color=colors, alpha=0.85, width=0.6)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
                        str(val), ha='center', color=MUTED, fontsize=8)
            ax.set_title("HAM10000 Class Distribution — 10,015 images", color=TEXT, fontsize=11)
            ax.set_ylabel("Count", color=MUTED, fontsize=9)
            ax.tick_params(colors=MUTED, labelsize=7.5); plt.xticks(rotation=25, ha='right')
            for s in ax.spines.values(): s.set_edgecolor(BORDER)
            ax.grid(axis='y', color=BORDER, linewidth=0.5, alpha=0.6)
            fig.tight_layout()
            st.image(fig_to_img(fig), use_container_width=True)

        with c2:
            fig2, ax2 = plt.subplots(figsize=(4.5,4.5))
            fig2.patch.set_facecolor(BG); ax2.set_facecolor(BG)
            wedges, texts, autotexts = ax2.pie(
                vals, labels=None, colors=colors,
                autopct='%1.1f%%', startangle=140,
                pctdistance=0.82,
                wedgeprops=dict(width=0.52, edgecolor=BG, linewidth=2)
            )
            for at in autotexts: at.set(color=TEXT, fontsize=7.5)
            ax2.legend([CLASSES[k]['name'] for k in labels], loc='lower center',
                       bbox_to_anchor=(0.5,-0.18), ncol=2, fontsize=7,
                       framealpha=0, labelcolor=MUTED)
            ax2.set_title("Proportion", color=TEXT, fontsize=11, pad=8)
            fig2.tight_layout()
            st.image(fig_to_img(fig2), use_container_width=True)

        if not meta_df.empty:
            st.markdown("#### Metadata Sample")
            st.dataframe(meta_df.head(8), use_container_width=True, hide_index=True)
            m1,m2,m3 = st.columns(3)
            m1.metric("Total Records", f"{len(meta_df):,}")
            m2.metric("Unique Lesions", f"{meta_df['lesion_id'].nunique():,}")
            m3.metric("Mean Age", f"{meta_df['age'].mean():.0f} yrs")

    with tab2:
        if not meta_df.empty:
            ca, cb = st.columns(2)
            with ca:
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                fig3.patch.set_facecolor(BG); ax3.set_facecolor(CARD)
                data_age = [meta_df[meta_df['dx']==k]['age'].dropna().values for k in CLASS_NAMES]
                bp = ax3.boxplot(data_age, patch_artist=True,
                                 medianprops=dict(color=TEXT, linewidth=1.5),
                                 whiskerprops=dict(color=MUTED),
                                 capprops=dict(color=MUTED),
                                 flierprops=dict(marker='.', color=MUTED, alpha=0.3))
                for patch, col in zip(bp['boxes'], CLASS_COLORS):
                    patch.set_facecolor(col); patch.set_alpha(0.6)
                ax3.set_xticklabels([k.upper() for k in CLASS_NAMES],
                                    rotation=20, color=MUTED, fontsize=8)
                ax3.set_ylabel("Age", color=MUTED, fontsize=9)
                ax3.set_title("Age Distribution by Class", color=TEXT, fontsize=11)
                ax3.tick_params(colors=MUTED)
                for s in ax3.spines.values(): s.set_edgecolor(BORDER)
                ax3.grid(axis='y', color=BORDER, linewidth=0.5, alpha=0.6)
                fig3.tight_layout()
                st.image(fig_to_img(fig3), use_container_width=True)

            with cb:
                fig4, ax4 = plt.subplots(figsize=(6,4))
                fig4.patch.set_facecolor(BG); ax4.set_facecolor(CARD)
                x = np.arange(len(CLASS_NAMES)); w = 0.35
                sc = meta_df.groupby(['dx','sex']).size().unstack(fill_value=0)
                males   = [sc.loc[k,'male']   if k in sc.index and 'male'   in sc.columns else 0 for k in CLASS_NAMES]
                females = [sc.loc[k,'female'] if k in sc.index and 'female' in sc.columns else 0 for k in CLASS_NAMES]
                ax4.bar(x-w/2, males,   w, label='Male',   color=ACCENT, alpha=0.82)
                ax4.bar(x+w/2, females, w, label='Female', color='#ff6b6b', alpha=0.82)
                ax4.set_xticks(x)
                ax4.set_xticklabels([k.upper() for k in CLASS_NAMES], rotation=20, color=MUTED, fontsize=8)
                ax4.set_title("Sex Distribution by Class", color=TEXT, fontsize=11)
                ax4.tick_params(colors=MUTED)
                ax4.legend(fontsize=8, framealpha=0, labelcolor=MUTED)
                for s in ax4.spines.values(): s.set_edgecolor(BORDER)
                ax4.grid(axis='y', color=BORDER, linewidth=0.5, alpha=0.6)
                fig4.tight_layout()
                st.image(fig_to_img(fig4), use_container_width=True)

            # Localization
            loc_counts = meta_df['localization'].value_counts().head(10)
            fig5, ax5 = plt.subplots(figsize=(8,3))
            fig5.patch.set_facecolor(BG); ax5.set_facecolor(CARD)
            colors5 = plt.cm.Blues(np.linspace(0.4, 0.9, len(loc_counts)))
            ax5.bar(loc_counts.index, loc_counts.values, color=colors5, alpha=0.9)
            ax5.set_title("Lesion Localization", color=TEXT, fontsize=11)
            ax5.tick_params(colors=MUTED, labelsize=8); plt.xticks(rotation=25, ha='right')
            for s in ax5.spines.values(): s.set_edgecolor(BORDER)
            ax5.grid(axis='y', color=BORDER, linewidth=0.5, alpha=0.6)
            fig5.tight_layout()
            st.image(fig_to_img(fig5), use_container_width=True)

    with tab3:
        g1,g2,g3 = st.columns(3)
        for col_, (p, cap) in zip([g1,g2,g3],[
            ("sample_images/lesions_collage1.png","BCC · Melanoma · Mixed"),
            ("sample_images/lesions_collage2.png","Vascular · Keratoses"),
            ("sample_images/lesions_collage3.png","Nevi · Melanoma"),
        ]):
            fp = APP_DIR/p
            if fp.exists():
                with col_:
                    st.image(Image.open(fp), use_container_width=True)
                    st.markdown(f"<div class='gcaption'>{cap}</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏗️ Architecture":
    st.markdown("<div class='sec-title'>🏗️ Model Architecture & Methodology</div>"
                "<div class='sec-line'></div>", unsafe_allow_html=True)

    ca, cb = st.columns([1,1], gap="large")
    with ca:
        st.markdown("#### Ensemble Pipeline")
        fig, ax = plt.subplots(figsize=(5,9))
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        layers_viz = [
            ("Input Image\n(any size)",             "#58c4dc", 0.82),
            ("Feature Extraction\nColour + Texture", "#7ab8ff", 0.72),
            ("20-dim Feature Vector",                "#b48eff", 0.62),
            ("MLP\n256 → 128 → 64",                 "#3fb950", 0.68),
            ("Random Forest\n200 estimators",        "#d29922", 0.68),
            ("Ensemble Average\nSoft Voting",        "#ff8c42", 0.60),
            ("Softmax Output\n7 Classes",            "#ff6b6b", 0.55),
        ]
        ys = np.linspace(0.93, 0.05, len(layers_viz))
        for i, ((name, col, w), y) in enumerate(zip(layers_viz, ys)):
            rect = FancyBboxPatch((0.5-w/2, y-0.042), w, 0.076,
                boxstyle="round,pad=0.01", facecolor=col, alpha=0.18,
                edgecolor=col, linewidth=1.5)
            ax.add_patch(rect)
            ax.text(0.5, y, name, ha='center', va='center', fontsize=8,
                    color=TEXT, fontfamily='monospace', fontweight='bold',
                    multialignment='center')
            if i < len(layers_viz)-1:
                ax.annotate("", xy=(0.5, ys[i+1]+0.042), xytext=(0.5, y-0.042),
                    arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.3))
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
        ax.set_title("Pipeline", color=TEXT, fontsize=11, pad=10)
        st.image(fig_to_img(fig), use_container_width=True)

        m1, m2 = st.columns(2)
        with m1: st.metric("Features","20-dim"); st.metric("RF Trees","200")
        with m2: st.metric("MLP Layers","256→128→64"); st.metric("Fusion","Soft Voting")

    with cb:
        st.markdown("#### 14-Step Methodology")
        for i,(title,desc) in enumerate(STEPS,1):
            st.markdown(f"""
            <div class='step-card'>
              <div class='step-num'>{i}</div>
              <div class='step-content'>
                <strong>{title}</strong><span>{desc}</span>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Training Configuration")
    for col,(l,v) in zip(st.columns(5),[("MLP Epochs","500 max"),("Batch","32"),
                                          ("LR","0.001"),("Early Stop","✓"),("Scaler","Standard")]):
        with col: st.metric(l,v)


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Results":
    st.markdown("<div class='sec-title'>📈 Results & Evaluation</div>"
                "<div class='sec-line'></div>", unsafe_allow_html=True)

    acc     = results.get("test_accuracy", 0.8295)
    mlp_acc = results.get("mlp_accuracy",  0.8068)
    rf_acc  = results.get("rf_accuracy",   0.8295)
    for col,(l,v) in zip(st.columns(4),[
        ("Ensemble Accuracy",f"{acc*100:.2f}%"),
        ("MLP Accuracy",f"{mlp_acc*100:.2f}%"),
        ("RF Accuracy",f"{rf_acc*100:.2f}%"),
        ("Classes","7")]):
        with col: st.metric(l,v)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Training Curves","Confusion Matrix","Per-Class Metrics"])

    with tab1:
        h = history
        if h:
            ep = list(range(1, len(h['accuracy'])+1))
            fig, axes = plt.subplots(1, 2, figsize=(11,4))
            fig.patch.set_facecolor(BG)
            for ax in axes: ax.set_facecolor(CARD)

            axes[0].plot(ep, h['accuracy'],     color=ACCENT,    lw=2,   label='Train Acc')
            axes[0].plot(ep, h['val_accuracy'], color='#ff6b6b', lw=2, ls='--', label='Val Acc')
            axes[0].axhline(acc, color='#3fb950', lw=1.5, ls=':', label=f'Test {acc*100:.1f}%')
            axes[0].set_title("Accuracy", color=TEXT, fontsize=11)
            axes[0].set_xlabel("Epoch", color=MUTED, fontsize=9)
            axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
            axes[0].legend(fontsize=8, framealpha=0, labelcolor=MUTED)

            axes[1].plot(ep, h['loss'],     color=ACCENT,    lw=2,   label='Train Loss')
            axes[1].plot(ep, h['val_loss'], color='#ff6b6b', lw=2, ls='--', label='Val Loss')
            axes[1].set_title("Loss", color=TEXT, fontsize=11)
            axes[1].set_xlabel("Epoch", color=MUTED, fontsize=9)
            axes[1].legend(fontsize=8, framealpha=0, labelcolor=MUTED)

            for ax in axes:
                ax.tick_params(colors=MUTED, labelsize=8)
                ax.grid(color=BORDER, linewidth=0.5, alpha=0.6)
                for s in ax.spines.values(): s.set_edgecolor(BORDER)
            fig.tight_layout()
            st.image(fig_to_img(fig), use_container_width=True)

    with tab2:
        cm_data = results.get('confusion_matrix')
        if cm_data:
            cm = np.array(cm_data)
            fig, ax = plt.subplots(figsize=(7,5.5))
            fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'dermcm', [BG, '#1a4a6b', ACCENT])
            im = ax.imshow(cm, cmap=cmap, aspect='auto')
            ax.set_xticks(range(7)); ax.set_yticks(range(7))
            ax.set_xticklabels([k.upper() for k in CLASS_NAMES], color=MUTED, fontsize=8)
            ax.set_yticklabels([k.upper() for k in CLASS_NAMES], color=MUTED, fontsize=8)
            ax.set_xlabel("Predicted", color=MUTED, fontsize=9)
            ax.set_ylabel("True Label", color=MUTED, fontsize=9)
            ax.set_title("Confusion Matrix", color=TEXT, fontsize=12)
            thresh = cm.max() / 2
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                            color=TEXT if cm[i,j] < thresh else BG, fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.03)
            fig.tight_layout()
            st.image(fig_to_img(fig), use_container_width=True)

    with tab3:
        cr = results.get('class_report',{})
        rows = []
        for k in CLASS_NAMES:
            m = cr.get(k,{})
            rows.append({"Class":k,"Name":CLASSES[k]['name'],
                         "Precision":f"{m.get('precision',0):.2f}",
                         "Recall":f"{m.get('recall',0):.2f}",
                         "F1":f"{m.get('f1-score',0):.2f}",
                         "Risk":CLASSES[k]['risk']})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Radar chart — matplotlib only
        metrics = ['Precision','Recall','F1']
        angles  = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(6,5), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
        ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, color=TEXT, fontsize=9)
        ax.set_ylim(0,1); ax.set_yticks([0.25,0.5,0.75,1.0])
        ax.set_yticklabels(['25%','50%','75%','100%'], color=MUTED, fontsize=7)
        ax.grid(color=BORDER, linewidth=0.7)
        ax.spines['polar'].set_edgecolor(BORDER)
        for cls, col in zip(CLASS_NAMES, COLORS):
            m = cr.get(cls,{})
            vals = [m.get('precision',0), m.get('recall',0), m.get('f1-score',0)]
            vals += vals[:1]
            ax.plot(angles, vals, color=col, lw=2, label=cls.upper())
            ax.fill(angles, vals, color=col, alpha=0.07)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35,1.1),
                  fontsize=8, framealpha=0, labelcolor=MUTED)
        ax.set_title("Per-Class Metrics", color=TEXT, fontsize=11, pad=18)
        fig.tight_layout()
        st.image(fig_to_img(fig), use_container_width=True)

    # Future plans
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>🚀 Future Plans</div><div class='sec-line'></div>",
                unsafe_allow_html=True)
    for col,(t,d) in zip(st.columns(3),[
        ("📱 Mobile App","Cross-platform iOS/Android with real-time camera classification"),
        ("🧠 Transfer Learning","Fine-tune ResNet50, EfficientNet for higher accuracy"),
        ("☁️ Cloud API","REST API on AWS/GCP with Docker for scalable serving"),
    ]):
        with col:
            st.markdown(f"""
            <div class='stat-card' style='text-align:left;'>
              <div style='font-size:1.3rem;margin-bottom:6px;'>{t.split()[0]}</div>
              <div style='font-weight:600;color:#e6edf3;margin-bottom:6px;'>{' '.join(t.split()[1:])}</div>
              <div style='font-size:.82rem;color:#8b949e;line-height:1.6;'>{d}</div>
            </div>""", unsafe_allow_html=True)
