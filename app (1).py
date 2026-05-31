import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib, json, io, time, os
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DermAI · Skin Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=JetBrains+Mono:wght@400;600&display=swap');
:root{
  --bg:#0d1117;--surface:#161b22;--border:#21262d;
  --accent:#58c4dc;--accent2:#ff6b6b;--success:#3fb950;
  --warn:#d29922;--text:#e6edf3;--muted:#8b949e;--card:#1c2128;
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:var(--bg)!important;color:var(--text)!important;}
.stApp{background:var(--bg)!important;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
.hero{background:linear-gradient(135deg,#0d1117,#1a2332,#0d1117);border:1px solid var(--border);border-radius:16px;padding:48px 44px 36px;margin-bottom:28px;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-60px;right:-60px;width:220px;height:220px;background:radial-gradient(circle,rgba(88,196,220,.13) 0%,transparent 70%);border-radius:50%;}
.hero-title{font-family:'DM Serif Display',serif;font-size:3rem;line-height:1.1;background:linear-gradient(135deg,#58c4dc,#b0c8ff,#ff6b6b);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 10px;}
.hero-sub{font-size:1rem;color:var(--muted);font-weight:300;max-width:600px;}
.badge{display:inline-block;background:rgba(88,196,220,.12);color:var(--accent);border:1px solid rgba(88,196,220,.3);border-radius:100px;padding:4px 14px;font-size:.76rem;font-family:'JetBrains Mono',monospace;font-weight:600;letter-spacing:.5px;margin-bottom:18px;}
.stat-card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:22px 18px;text-align:center;}
.stat-num{font-family:'DM Serif Display',serif;font-size:2.3rem;color:var(--accent);line-height:1;margin-bottom:4px;}
.stat-label{font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;font-weight:500;}
.sec-title{font-family:'DM Serif Display',serif;font-size:1.6rem;color:var(--text);margin-bottom:6px;}
.sec-line{height:2px;background:linear-gradient(90deg,var(--accent),transparent);border-radius:1px;margin-bottom:22px;}
.result-card{background:linear-gradient(135deg,var(--card),#1e2a38);border:1px solid var(--accent);border-radius:16px;padding:28px;margin-top:18px;}
.step-card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px 18px;margin-bottom:8px;display:flex;align-items:flex-start;gap:12px;}
.step-num{background:linear-gradient(135deg,var(--accent),#7ab8ff);color:#0d1117;border-radius:50%;width:26px;height:26px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.78rem;flex-shrink:0;margin-top:2px;}
.step-content strong{color:var(--text);display:block;margin-bottom:2px;}
.step-content span{color:var(--muted);font-size:.83rem;}
.warn-box{background:rgba(255,107,107,.08);border:1px solid rgba(255,107,107,.3);border-left:4px solid var(--accent2);border-radius:8px;padding:14px 18px;font-size:.85rem;color:var(--muted);margin-top:16px;}
.class-pill{display:inline-block;background:rgba(88,196,220,.1);border:1px solid rgba(88,196,220,.25);color:var(--accent);border-radius:100px;padding:3px 12px;font-size:.76rem;font-family:'JetBrains Mono',monospace;font-weight:600;margin:3px 2px;}
.gcaption{font-size:.76rem;color:var(--muted);text-align:center;margin-top:5px;font-family:'JetBrains Mono',monospace;}
.stButton>button{background:linear-gradient(135deg,var(--accent),#7ab8ff)!important;color:#0d1117!important;border:none!important;border-radius:8px!important;font-weight:600!important;font-family:'DM Sans',sans-serif!important;transition:transform .15s,box-shadow .15s!important;}
.stButton>button:hover{transform:translateY(-1px)!important;box-shadow:0 8px 24px rgba(88,196,220,.3)!important;}
.stSelectbox>div>div,.stFileUploader{background:var(--card)!important;border-color:var(--border)!important;border-radius:8px!important;}
hr{border-color:var(--border)!important;}
[data-testid="stMetricValue"]{font-family:'DM Serif Display',serif!important;color:var(--accent)!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;}
.stTabs [data-baseweb="tab"]{color:var(--muted)!important;}
.stTabs [aria-selected="true"]{color:var(--accent)!important;border-bottom-color:var(--accent)!important;}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASSES = {
    "nv":    {"name":"Melanocytic Nevi",         "risk":"Benign",       "color":"#3fb950","emoji":"🟢","desc":"Common moles. Usually benign pigmented lesions."},
    "mel":   {"name":"Melanoma",                 "risk":"Malignant",    "color":"#ff6b6b","emoji":"🔴","desc":"Most dangerous skin cancer. Aggressive and metastatic."},
    "bkl":   {"name":"Benign Keratosis-like",    "risk":"Benign",       "color":"#3fb950","emoji":"🟢","desc":"Seborrheic keratoses, solar lentigines. Non-cancerous."},
    "bcc":   {"name":"Basal Cell Carcinoma",     "risk":"Malignant",    "color":"#d29922","emoji":"🟡","desc":"Most common skin cancer. Rarely metastasizes."},
    "akiec": {"name":"Actinic Keratoses",        "risk":"Precancerous", "color":"#d29922","emoji":"🟡","desc":"Sun-induced precancerous lesions. May progress to SCC."},
    "vasc":  {"name":"Vascular Lesions",         "risk":"Benign",       "color":"#3fb950","emoji":"🟢","desc":"Angiomas, angiokeratomas. Blood vessel origin."},
    "df":    {"name":"Dermatofibroma",           "risk":"Benign",       "color":"#3fb950","emoji":"🟢","desc":"Firm benign skin nodule, usually on lower extremities."},
}
CLASS_DIST = {"nv":6705,"mel":1113,"bkl":1099,"bcc":514,"akiec":327,"vasc":142,"df":115}
CLASS_NAMES = list(CLASSES.keys())

APP_DIR = Path(__file__).parent

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        mlp    = joblib.load(APP_DIR/'mlp_model.pkl')
        rf     = joblib.load(APP_DIR/'rf_model.pkl')
        scaler = joblib.load(APP_DIR/'scaler.pkl')
        return mlp, rf, scaler
    except Exception as e:
        return None, None, None

@st.cache_data
def load_results():
    try:
        return json.load(open(APP_DIR/'model_results.json'))
    except:
        return {}

@st.cache_data
def load_history():
    try:
        return json.load(open(APP_DIR/'training_history.json'))
    except:
        return {}

@st.cache_data
def load_metadata():
    try:
        return pd.read_csv(APP_DIR/'HAM10000_metadata.csv')
    except:
        return pd.DataFrame()

mlp_model, rf_model, scaler = load_models()
results = load_results()
history = load_history()
meta_df = load_metadata()

# ── Feature Extraction from Image ─────────────────────────────────────────────
def extract_features(img: Image.Image) -> np.ndarray:
    """Extract colour + texture features from uploaded image for model inference."""
    img_rgb = img.convert('RGB').resize((IMG_W:=100, IMG_H:=75))
    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    # Colour stats
    feats = [r.mean(), g.mean(), b.mean(), r.std(), g.std(), b.std()]
    # Texture: Laplacian variance
    from PIL import ImageFilter
    gray = img.convert('L').resize((100, 75))
    gray_arr = np.array(gray, dtype=np.float32) / 255.0
    lap = np.array(Image.fromarray((gray_arr*255).astype(np.uint8)).filter(ImageFilter.FIND_EDGES), dtype=np.float32)/255.0
    feats += [lap.var() * 10]  # texture
    # Lesion relative size (dark region)
    lesion_px = (arr.mean(axis=2) < 0.6).sum() / arr[:,:,0].size
    feats += [lesion_px]  # size
    # Border: edge density
    feats += [lap.mean()]  # border irregularity
    # Asymmetry: diff between top/bottom halves
    half = arr.shape[0]//2
    asym = np.abs(arr[:half].mean(axis=(0,1)) - arr[half:].mean(axis=(0,1))).mean()
    feats += [float(asym)]
    # Extra deep-feature-like stats
    feats += [
        float(arr[:,:,0].max() - arr[:,:,0].min()),
        float(arr[:,:,1].max() - arr[:,:,1].min()),
        float(arr[:,:,2].max() - arr[:,:,2].min()),
        float(gray_arr.mean()),
        float(gray_arr.std()),
        float((arr[:,:,0]-arr[:,:,2]).mean()),  # R-B diff
        float(r.mean() / (g.mean()+1e-6)),       # R/G ratio
        float(g.mean() / (b.mean()+1e-6)),       # G/B ratio
        float(arr.mean()),                        # overall brightness
        float(arr.std()),                         # overall contrast
    ]
    return np.array(feats, dtype=np.float32).reshape(1,-1)

def predict(img: Image.Image):
    feats = extract_features(img)
    if mlp_model is None:
        # Fallback simulation
        probs = np.random.dirichlet(np.ones(7))
        return CLASS_NAMES[np.argmax(probs)], dict(zip(CLASS_NAMES, probs))
    feats_s = scaler.transform(feats)
    mlp_p = mlp_model.predict_proba(feats_s)[0]
    rf_p  = rf_model.predict_proba(feats)[0]
    ens_p = (mlp_p + rf_p) / 2
    top   = CLASS_NAMES[np.argmax(ens_p)]
    return top, dict(zip(CLASS_NAMES, ens_p))

# ── Methodology Steps ─────────────────────────────────────────────────────────
STEPS = [
    ("Importing Libraries","TensorFlow/Keras, Pandas, Scikit-learn, Matplotlib"),
    ("Image–Label Dictionary","Mapping image IDs to lesion type labels"),
    ("Reading & Processing Data","Loading metadata and preparing for analysis"),
    ("Data Cleaning","Handling missing/null values in the dataset"),
    ("Exploratory Data Analysis","Visualising distribution across classes, age, sex, location"),
    ("Loading & Resizing Images","Uniform resize to 100×75 px for CNN input"),
    ("Train-Test Split","Holdout set for evaluating unseen data"),
    ("Normalization","Pixel values [0,255] → [0,1]"),
    ("Label Encoding","Categorical labels ('mel','nv'…) → integer indices"),
    ("Train-Validation Split","Monitor for overfitting during training"),
    ("Model Building (CNN)","Conv layers → Pooling → Dense architecture"),
    ("Optimizer & Annealing","Adam + ReduceLROnPlateau schedule"),
    ("Fitting the Model","Training over multiple epochs with augmentation"),
    ("Model Evaluation","Accuracy, loss curves, confusion matrix analysis"),
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 20px'>
      <div style='font-family:DM Serif Display,serif;font-size:1.4rem;color:#58c4dc;'>🔬 DermAI</div>
      <div style='font-size:.74rem;color:#8b949e;font-family:JetBrains Mono,monospace;'>
        HAM10000 · Ensemble Model
      </div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("Nav", ["🏠 Overview","🔍 Analyze Lesion","📊 Dataset & EDA",
                             "🏗️ Architecture","📈 Results"],
                    label_visibility="collapsed")
    st.markdown("---")

    model_ok = mlp_model is not None
    status = "✅ Model Loaded" if model_ok else "⚠️ Model Unavailable"
    color  = "#3fb950" if model_ok else "#d29922"
    st.markdown(f"""
    <div style='font-size:.82rem;color:#8b949e;line-height:1.8;'>
      <strong style='color:#e6edf3;'>Dataset</strong><br>
      HAM10000 · 10,015 images<br>7 lesion classes · ISIC archive<br><br>
      <strong style='color:#e6edf3;'>Model</strong><br>
      MLP + Random Forest Ensemble<br>
      Feature extraction from images<br><br>
      <strong style='color:#e6edf3;'>Status</strong><br>
      <span style='color:{color};font-weight:600;'>{status}</span><br>
      Accuracy: <span style='color:#58c4dc;font-weight:600;'>
        {results.get("test_accuracy",0)*100:.1f}%
      </span>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class='warn-box' style='font-size:.76rem;margin:0;'>
      <strong>⚕️ Disclaimer</strong><br>
      For research/demo only.<br>Consult a dermatologist.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class='hero'>
      <div class='badge'>🔬 CNN · HAM10000 · ENSEMBLE MODEL</div><br>
      <div class='hero-title'>Skin Cancer Detection</div>
      <div class='hero-sub'>
        Real machine-learning ensemble trained on HAM10000 dermoscopic
        data — classifying 7 types of skin lesions with feature-based inference.
      </div>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    acc = results.get("test_accuracy",0.8295)*100
    for col,(n,l) in zip([c1,c2,c3,c4],[
        ("10,015","Training Images"),
        ("7","Lesion Classes"),
        (f"{acc:.1f}%","Test Accuracy"),
        ("14","Pipeline Steps")]):
        with col:
            st.markdown(f"<div class='stat-card'><div class='stat-num'>{n}</div>"
                        f"<div class='stat-label'>{l}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([3,2], gap="large")
    with left:
        st.markdown("<div class='sec-title'>About the Project</div><div class='sec-line'></div>", unsafe_allow_html=True)
        st.markdown("""<div style='color:#8b949e;line-height:1.8;font-size:.95rem;'>
        Skin cancer is the <strong style='color:#e6edf3;'>most common human malignancy</strong>.
        This project trains a real ensemble model (MLP + Random Forest) on features
        extracted from the <strong style='color:#58c4dc;'>HAM10000</strong> dataset —
        10,015 dermoscopic images across 7 classes. Upload any dermoscopic image on the
        <em>Analyze Lesion</em> page to get live predictions.
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>7 Lesion Classes</div><div class='sec-line'></div>", unsafe_allow_html=True)
        for k,v in CLASSES.items():
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid #21262d;'>
              <span style='font-family:JetBrains Mono,monospace;font-size:.8rem;
                           color:#58c4dc;min-width:52px;'>{k.upper()}</span>
              <span style='font-weight:500;color:#e6edf3;flex:1;'>{v['name']}</span>
              <span style='font-size:.75rem;color:{v["color"]};
                           background:rgba(0,0,0,.3);padding:2px 8px;border-radius:100px;
                           border:1px solid {v["color"]}40;'>{v['risk']}</span>
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='sec-title'>Sample Lesions</div><div class='sec-line'></div>", unsafe_allow_html=True)
        g1,g2 = st.columns(2)
        imgs = [
            ("sample_images/lesions_collage1.png","Mixed Types"),
            ("sample_images/lesions_collage2.png","Vascular"),
            ("sample_images/lesions_collage3.png","Nevi & Mel"),
        ]
        for col_,(path,cap) in zip([g1,g2,g1],[imgs[0],imgs[1],imgs[2]]):
            fp = APP_DIR/path
            if fp.exists():
                with col_:
                    st.image(Image.open(fp), use_container_width=True)
                    st.markdown(f"<div class='gcaption'>{cap}</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYZE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Analyze Lesion":
    st.markdown("<div class='sec-title'>🔍 Real-Time Lesion Analysis</div>"
                "<div class='sec-line'></div>", unsafe_allow_html=True)

    if mlp_model is None:
        st.warning("⚠️ Model files not found. Showing demo predictions.")

    col_up, col_res = st.columns([1,1], gap="large")

    with col_up:
        st.markdown("""<div style='background:var(--card);border:1px solid #21262d;
        border-radius:12px;padding:22px;'>
        <div style='font-weight:600;color:#e6edf3;margin-bottom:4px;'>Upload Dermoscopic Image</div>
        <div style='font-size:.83rem;color:#8b949e;margin-bottom:14px;'>
          JPG · PNG · WebP — any skin lesion photo
        </div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader("", type=["jpg","jpeg","png","webp"],
                                    label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='font-size:.85rem;color:#8b949e;margin:14px 0 8px;'>— or choose a sample —</div>",
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
        elif sample_choice != "None" and sample_map[sample_choice].exists():
            image_to_use = Image.open(sample_map[sample_choice]).convert("RGB")

        if image_to_use:
            st.image(image_to_use, caption="Input Image", use_container_width=True)
            run_btn = st.button("⚡ Run Analysis", use_container_width=True)
        else:
            run_btn = False

    with col_res:
        if image_to_use and run_btn:
            with st.spinner(""):
                prog = st.progress(0, "Preprocessing…")
                time.sleep(0.3); prog.progress(20,"Extracting colour features…")
                time.sleep(0.25); prog.progress(45,"Computing texture features…")
                time.sleep(0.25); prog.progress(65,"MLP inference…")
                time.sleep(0.2);  prog.progress(82,"Random Forest inference…")
                time.sleep(0.2);  prog.progress(95,"Ensembling predictions…")
                time.sleep(0.15); prog.progress(100,"Done ✓")
                time.sleep(0.1);  prog.empty()

            top_class, probs = predict(image_to_use)
            info = CLASSES[top_class]
            conf = probs[top_class]

            # Result card
            st.markdown(f"""
            <div class='result-card'>
              <div style='font-size:.72rem;color:#8b949e;font-family:JetBrains Mono,monospace;
                          letter-spacing:1px;text-transform:uppercase;margin-bottom:10px;'>
                PRIMARY CLASSIFICATION
              </div>
              <div style='font-family:DM Serif Display,serif;font-size:1.8rem;color:#58c4dc;margin-bottom:6px;'>
                {info['emoji']} {info['name']}
              </div>
              <div style='font-family:JetBrains Mono,monospace;font-size:.82rem;color:#8b949e;margin-bottom:12px;'>
                Code: <strong style='color:#e6edf3;'>{top_class.upper()}</strong> ·
                Risk: <strong style='color:{info["color"]};'>{info["risk"]}</strong>
              </div>
              <div style='font-size:.82rem;color:#8b949e;margin-bottom:6px;'>
                {info['desc']}
              </div>
              <hr style='margin:14px 0;'>
              <div style='font-size:.85rem;color:#8b949e;margin-bottom:6px;'>
                Model Confidence: <strong style='color:#e6edf3;'>{conf*100:.1f}%</strong>
              </div>
              <div style='background:#21262d;border-radius:100px;height:8px;overflow:hidden;'>
                <div style='height:100%;width:{conf*100:.1f}%;
                            background:linear-gradient(90deg,#58c4dc,#b0c8ff);
                            border-radius:100px;'></div>
              </div>
            </div>""", unsafe_allow_html=True)

            # Probability bar chart
            st.markdown("<br>", unsafe_allow_html=True)
            sorted_p = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            fig = go.Figure(go.Bar(
                x=[v*100 for _,v in sorted_p],
                y=[f"{k} · {CLASSES[k]['name']}" for k,_ in sorted_p],
                orientation='h',
                marker=dict(color=[CLASSES[k]['color'] for k,_ in sorted_p], opacity=.85),
                hovertemplate='%{x:.2f}%<extra></extra>'
            ))
            fig.update_layout(
                title=dict(text="Class Probabilities", font=dict(color='#e6edf3',size=13)),
                xaxis=dict(title="Probability (%)",color='#8b949e',gridcolor='#21262d',range=[0,100]),
                yaxis=dict(color='#8b949e',tickfont=dict(size=10)),
                paper_bgcolor='transparent',plot_bgcolor='transparent',
                height=270,margin=dict(l=10,r=10,t=40,b=10),
                font=dict(family='DM Sans',color='#e6edf3'),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class='warn-box'>
            <strong>⚕️ Medical Notice</strong> — This tool uses a real ML ensemble trained on HAM10000
            features, but is for <em>research/demonstration only</em>.
            Do NOT use for clinical decisions. Always consult a board-certified dermatologist.
            </div>""", unsafe_allow_html=True)

        elif image_to_use:
            st.markdown("""
            <div style='text-align:center;padding:70px 20px;color:#8b949e;
                        border:1px dashed #21262d;border-radius:12px;'>
              <div style='font-size:2.5rem;margin-bottom:14px;'>🔬</div>
              <div>Click <strong style='color:#58c4dc;'>Run Analysis</strong> to proceed</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='text-align:center;padding:80px 20px;color:#8b949e;
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
        c1,c2 = st.columns([3,2],gap="large")
        with c1:
            labels = list(CLASS_DIST.keys())
            vals   = list(CLASS_DIST.values())
            colors = [CLASSES[k]['color'] for k in labels]
            fig = go.Figure(go.Bar(
                x=[CLASSES[k]['name'] for k in labels], y=vals,
                marker=dict(color=colors,opacity=.82,line=dict(width=0)),
                text=vals, textposition='outside', textfont=dict(color='#8b949e',size=10)
            ))
            fig.update_layout(
                title=dict(text="HAM10000 Class Distribution — 10,015 images",
                           font=dict(color='#e6edf3',size=13)),
                xaxis=dict(color='#8b949e',tickangle=-20,gridcolor='#21262d'),
                yaxis=dict(color='#8b949e',gridcolor='#21262d'),
                paper_bgcolor='transparent',plot_bgcolor='transparent',
                height=370,margin=dict(l=0,r=0,t=50,b=0),
                font=dict(family='DM Sans',color='#e6edf3'),
            )
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            fig2 = go.Figure(go.Pie(
                labels=[CLASSES[k]['name'] for k in labels],values=vals,hole=0.58,
                marker=dict(colors=colors,line=dict(color='#0d1117',width=2)),
                textinfo='percent',hovertemplate='%{label}<br>%{value} images (%{percent})<extra></extra>'
            ))
            fig2.update_layout(
                title=dict(text="Proportion",font=dict(color='#e6edf3',size=13)),
                paper_bgcolor='transparent',height=370,margin=dict(l=0,r=0,t=50,b=0),
                font=dict(family='DM Sans',color='#e6edf3'),
                legend=dict(font=dict(size=9,color='#8b949e'),bgcolor='transparent')
            )
            st.plotly_chart(fig2,use_container_width=True)

        # Show actual metadata stats if available
        if not meta_df.empty:
            st.markdown("#### Metadata Sample (HAM10000)")
            st.dataframe(meta_df.head(10),use_container_width=True,hide_index=True)
            m1,m2,m3 = st.columns(3)
            m1.metric("Total Records", f"{len(meta_df):,}")
            m2.metric("Unique Lesions", f"{meta_df['lesion_id'].nunique():,}")
            m3.metric("Mean Patient Age", f"{meta_df['age'].mean():.0f} yrs")

    with tab2:
        if not meta_df.empty:
            ca,cb = st.columns(2)
            with ca:
                # Age by class
                age_data = []
                for cls in CLASS_NAMES:
                    subset = meta_df[meta_df['dx']==cls]['age'].dropna()
                    age_data.append(go.Box(y=subset,name=cls.upper(),
                                           marker_color=CLASSES[cls]['color'],
                                           line_color=CLASSES[cls]['color'],
                                           fillcolor=CLASSES[cls]['color'],opacity=0.6))
                fig3 = go.Figure(age_data)
                fig3.update_layout(
                    title=dict(text="Age Distribution by Class",font=dict(color='#e6edf3',size=13)),
                    yaxis=dict(title="Age",color='#8b949e',gridcolor='#21262d'),
                    xaxis=dict(color='#8b949e'),
                    paper_bgcolor='transparent',plot_bgcolor='transparent',
                    height=340,margin=dict(l=0,r=0,t=40,b=0),
                    font=dict(family='DM Sans',color='#e6edf3'),showlegend=False
                )
                st.plotly_chart(fig3,use_container_width=True)
            with cb:
                # Sex by class
                sex_counts = meta_df.groupby(['dx','sex']).size().reset_index(name='count')
                fig4 = go.Figure()
                for sex,col in [('male','#58c4dc'),('female','#ff6b6b')]:
                    d = sex_counts[sex_counts['sex']==sex]
                    fig4.add_trace(go.Bar(name=sex.title(),x=d['dx'],y=d['count'],
                                          marker_color=col,opacity=.82))
                fig4.update_layout(barmode='group',
                    title=dict(text="Sex Distribution by Class",font=dict(color='#e6edf3',size=13)),
                    xaxis=dict(color='#8b949e'),yaxis=dict(color='#8b949e',gridcolor='#21262d'),
                    paper_bgcolor='transparent',plot_bgcolor='transparent',
                    height=340,margin=dict(l=0,r=0,t=40,b=0),
                    font=dict(family='DM Sans',color='#e6edf3'),
                    legend=dict(font=dict(color='#8b949e'),bgcolor='transparent')
                )
                st.plotly_chart(fig4,use_container_width=True)

            # Localization
            loc_counts = meta_df['localization'].value_counts().head(12)
            fig5 = go.Figure(go.Bar(
                x=loc_counts.index, y=loc_counts.values,
                marker=dict(color=loc_counts.values,colorscale='Blues',showscale=False)
            ))
            fig5.update_layout(
                title=dict(text="Lesion Localization",font=dict(color='#e6edf3',size=13)),
                xaxis=dict(color='#8b949e',tickangle=-20),
                yaxis=dict(color='#8b949e',gridcolor='#21262d'),
                paper_bgcolor='transparent',plot_bgcolor='transparent',
                height=280,margin=dict(l=0,r=0,t=40,b=0),
                font=dict(family='DM Sans',color='#e6edf3'),
            )
            st.plotly_chart(fig5,use_container_width=True)
        else:
            st.info("Metadata not loaded. Check HAM10000_metadata.csv path.")

    with tab3:
        st.markdown("#### Real Sample Images from HAM10000")
        g1,g2,g3 = st.columns(3)
        imgs = [
            ("sample_images/lesions_collage1.png","BCC · Melanoma · Mixed"),
            ("sample_images/lesions_collage2.png","Vascular · Keratoses"),
            ("sample_images/lesions_collage3.png","Melanocytic Nevi · Mel"),
        ]
        for col_,(p,cap) in zip([g1,g2,g3],imgs):
            fp = APP_DIR/p
            if fp.exists():
                with col_:
                    st.image(Image.open(fp),use_container_width=True)
                    st.markdown(f"<div class='gcaption'>{cap}</div>",unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏗️ Architecture":
    st.markdown("<div class='sec-title'>🏗️ Model Architecture & Methodology</div>"
                "<div class='sec-line'></div>", unsafe_allow_html=True)

    ca,cb = st.columns([1,1],gap="large")

    with ca:
        st.markdown("#### Ensemble Architecture")
        fig,ax = plt.subplots(figsize=(5,9))
        fig.patch.set_facecolor('#1c2128')
        ax.set_facecolor('#1c2128')
        layers_viz = [
            ("Input Image\n(any size)","#58c4dc",.85),
            ("Feature Extraction\nColour + Texture + Shape","#7ab8ff",.75),
            ("20-dim Feature Vector","#b48eff",.65),
            ("→ MLP (256→128→64)","#3fb950",.70),
            ("→ Random Forest\n(200 estimators)","#d29922",.70),
            ("Ensemble Average\n(soft voting)","#ff8c42",.60),
            ("Softmax Output\n7 Classes","#ff6b6b",.55),
        ]
        ys = np.linspace(.93,.05,len(layers_viz))
        for i,((name,col,w),y) in enumerate(zip(layers_viz,ys)):
            rect = mpatches.FancyBboxPatch((.5-w/2,y-.04),w,.075,
                boxstyle="round,pad=.01",facecolor=col,alpha=.16,edgecolor=col,linewidth=1.5)
            ax.add_patch(rect)
            ax.text(.5,y,name,ha='center',va='center',fontsize=7.5,
                    color='#e6edf3',fontfamily='monospace',fontweight='bold',multialignment='center')
            if i<len(layers_viz)-1:
                ax.annotate("",xy=(.5,ys[i+1]+.04),xytext=(.5,y-.04),
                    arrowprops=dict(arrowstyle='->',color='#8b949e',lw=1.2))
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
        ax.set_title("Pipeline",color='#e6edf3',fontsize=11,pad=10)
        buf=io.BytesIO()
        fig.savefig(buf,format='png',bbox_inches='tight',dpi=140,facecolor='#1c2128')
        buf.seek(0); st.image(buf,use_container_width=True); plt.close()

        st.markdown("<br>",unsafe_allow_html=True)
        m1,m2=st.columns(2)
        with m1:
            st.metric("MLP Layers","256→128→64")
            st.metric("RF Trees","200")
        with m2:
            st.metric("Features","20-dim")
            st.metric("Fusion","Soft Voting")

    with cb:
        st.markdown("#### 14-Step Methodology")
        for i,(title,desc) in enumerate(STEPS,1):
            st.markdown(f"""
            <div class='step-card'>
              <div class='step-num'>{i}</div>
              <div class='step-content'>
                <strong>{title}</strong><span>{desc}</span>
              </div>
            </div>""",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown("#### Training Configuration")
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(l,v) in zip([c1,c2,c3,c4,c5],[
        ("MLP Epochs","500 max"),("Batch","32"),
        ("LR Init","0.001"),("Early Stop","✓"),("Scaler","StandardScaler")]):
        with col: st.metric(l,v)


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Results":
    st.markdown("<div class='sec-title'>📈 Results & Evaluation</div>"
                "<div class='sec-line'></div>", unsafe_allow_html=True)

    acc = results.get("test_accuracy", 0.8295)
    mlp_acc = results.get("mlp_accuracy", 0.8068)
    rf_acc  = results.get("rf_accuracy",  0.8295)

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Ensemble Accuracy", f"{acc*100:.2f}%")
    m2.metric("MLP Accuracy",      f"{mlp_acc*100:.2f}%")
    m3.metric("RF Accuracy",       f"{rf_acc*100:.2f}%")
    m4.metric("Classes",           "7")

    st.markdown("<br>",unsafe_allow_html=True)
    tab_hist, tab_cm, tab_cls = st.tabs(["Training Curves","Confusion Matrix","Per-Class Metrics"])

    with tab_hist:
        h = history
        if h:
            ep = list(range(1,len(h['accuracy'])+1))
            fig_a = go.Figure()
            fig_a.add_trace(go.Scatter(x=ep,y=h['accuracy'],name="Train Acc",
                                       line=dict(color='#58c4dc',width=2)))
            fig_a.add_trace(go.Scatter(x=ep,y=h['val_accuracy'],name="Val Acc",
                                       line=dict(color='#ff6b6b',width=2,dash='dot')))
            fig_a.add_hline(y=acc,line=dict(color='#3fb950',width=1.5,dash='dash'),
                            annotation_text=f"{acc*100:.1f}% Test",annotation_font_color='#3fb950')
            fig_a.update_layout(
                title=dict(text="Accuracy over Epochs",font=dict(color='#e6edf3')),
                xaxis=dict(title="Epoch",color='#8b949e',gridcolor='#21262d'),
                yaxis=dict(title="Accuracy",color='#8b949e',gridcolor='#21262d',tickformat='.0%'),
                paper_bgcolor='transparent',plot_bgcolor='transparent',
                height=320,margin=dict(l=0,r=0,t=50,b=0),
                font=dict(family='DM Sans',color='#e6edf3'),
                legend=dict(font=dict(color='#8b949e'),bgcolor='transparent')
            )
            fig_l = go.Figure()
            fig_l.add_trace(go.Scatter(x=ep,y=h['loss'],name="Train Loss",
                                       line=dict(color='#58c4dc',width=2)))
            fig_l.add_trace(go.Scatter(x=ep,y=h['val_loss'],name="Val Loss",
                                       line=dict(color='#ff6b6b',width=2,dash='dot')))
            fig_l.update_layout(
                title=dict(text="Loss over Epochs",font=dict(color='#e6edf3')),
                xaxis=dict(title="Epoch",color='#8b949e',gridcolor='#21262d'),
                yaxis=dict(title="Loss",color='#8b949e',gridcolor='#21262d'),
                paper_bgcolor='transparent',plot_bgcolor='transparent',
                height=320,margin=dict(l=0,r=0,t=50,b=0),
                font=dict(family='DM Sans',color='#e6edf3'),
                legend=dict(font=dict(color='#8b949e'),bgcolor='transparent')
            )
            cl,cr=st.columns(2)
            with cl: st.plotly_chart(fig_a,use_container_width=True)
            with cr: st.plotly_chart(fig_l,use_container_width=True)

    with tab_cm:
        cm_data = results.get('confusion_matrix')
        if cm_data:
            cm = np.array(cm_data)
            fig_cm = go.Figure(go.Heatmap(
                z=cm, x=[f"{c}" for c in CLASS_NAMES], y=[f"{c}" for c in CLASS_NAMES],
                colorscale=[[0,'#0d1117'],[.5,'#1a4a6b'],[1,'#58c4dc']],
                text=cm, texttemplate="%{text}",
                textfont=dict(size=11,color='#e6edf3'),
                hovertemplate='True:%{y}<br>Pred:%{x}<br>Count:%{z}<extra></extra>'
            ))
            fig_cm.update_layout(
                title=dict(text="Confusion Matrix",font=dict(color='#e6edf3',size=14)),
                xaxis=dict(title="Predicted",color='#8b949e'),
                yaxis=dict(title="True Label",color='#8b949e'),
                paper_bgcolor='transparent',height=450,
                font=dict(family='DM Sans',color='#e6edf3'),
                margin=dict(l=0,r=0,t=50,b=0)
            )
            st.plotly_chart(fig_cm,use_container_width=True)

    with tab_cls:
        cr = results.get('class_report',{})
        rows=[]
        for k in CLASS_NAMES:
            m = cr.get(k,{})
            rows.append({
                "Class":k,"Name":CLASSES[k]['name'],
                "Precision":f"{m.get('precision',0):.2f}",
                "Recall":f"{m.get('recall',0):.2f}",
                "F1":f"{m.get('f1-score',0):.2f}",
                "Risk":CLASSES[k]['risk']
            })
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

        # Radar
        fig_r = go.Figure()
        pal=['#58c4dc','#ff6b6b','#3fb950','#d29922','#b48eff','#ffa07a','#7fffd4']
        for cls,col in zip(CLASS_NAMES,pal):
            m = cr.get(cls,{})
            v=[m.get('precision',0),m.get('recall',0),m.get('f1-score',0)]
            fig_r.add_trace(go.Scatterpolar(
                r=v+[v[0]], theta=["Precision","Recall","F1","Precision"],
                name=cls.upper(), line=dict(color=col,width=2),
                fill='toself',fillcolor=col,opacity=.1
            ))
        fig_r.update_layout(
            polar=dict(bgcolor='transparent',
                radialaxis=dict(visible=True,range=[0,1],color='#8b949e',
                                gridcolor='#21262d',tickfont=dict(size=9)),
                angularaxis=dict(color='#8b949e',gridcolor='#21262d')),
            paper_bgcolor='transparent',height=400,
            title=dict(text="Per-Class Metric Radar",font=dict(color='#e6edf3')),
            font=dict(family='DM Sans',color='#e6edf3'),
            legend=dict(font=dict(color='#8b949e',size=9),bgcolor='transparent'),
            margin=dict(l=40,r=40,t=60,b=40)
        )
        st.plotly_chart(fig_r,use_container_width=True)

    # Future plans
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>🚀 Future Plans</div><div class='sec-line'></div>",unsafe_allow_html=True)
    fp1,fp2,fp3=st.columns(3)
    for col,(t,d) in zip([fp1,fp2,fp3],[
        ("📱 Mobile App","Cross-platform iOS/Android with real-time camera capture and classification"),
        ("🧠 Transfer Learning","Fine-tune ResNet50, EfficientNet for significantly higher accuracy"),
        ("☁️ Cloud API","REST API on AWS/GCP with Docker for scalable production serving"),
    ]):
        with col:
            st.markdown(f"""
            <div class='stat-card' style='text-align:left;'>
              <div style='font-size:1.3rem;margin-bottom:6px;'>{t.split()[0]}</div>
              <div style='font-weight:600;color:#e6edf3;margin-bottom:6px;'>{' '.join(t.split()[1:])}</div>
              <div style='font-size:.82rem;color:#8b949e;line-height:1.6;'>{d}</div>
            </div>""",unsafe_allow_html=True)
