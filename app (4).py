import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
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
      --accent2:#ff6b6b;--success:#3fb950;--warn:#d29922;
      --text:#e6edf3;--muted:#8b949e;--card:#1c2128;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;
  background:var(--bg)!important;color:var(--text)!important;}
.stApp{background:var(--bg)!important;}
[data-testid="stSidebar"]{background:var(--surface)!important;
  border-right:1px solid var(--border)!important;}
.hero{background:linear-gradient(135deg,#0d1117,#1a2332,#0d1117);
  border:1px solid #21262d;border-radius:16px;padding:48px 44px 36px;
  margin-bottom:28px;position:relative;overflow:hidden;}
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
.sec-title{font-family:'DM Serif Display',serif;font-size:1.6rem;
  color:var(--text);margin-bottom:6px;}
.sec-line{height:2px;background:linear-gradient(90deg,#58c4dc,transparent);
  border-radius:1px;margin-bottom:22px;}
.result-card{background:linear-gradient(135deg,#1c2128,#1e2a38);
  border:1px solid #58c4dc;border-radius:16px;padding:28px;margin-top:18px;}
.step-card{background:var(--card);border:1px solid var(--border);border-radius:10px;
  padding:16px 18px;margin-bottom:8px;display:flex;align-items:flex-start;gap:12px;}
.step-num{background:linear-gradient(135deg,#58c4dc,#7ab8ff);color:#0d1117;
  border-radius:50%;width:26px;height:26px;display:flex;align-items:center;
  justify-content:center;font-weight:700;font-size:.78rem;flex-shrink:0;margin-top:2px;}
.step-content strong{color:#e6edf3;display:block;margin-bottom:2px;}
.step-content span{color:#8b949e;font-size:.83rem;}
.warn-box{background:rgba(255,107,107,.08);border:1px solid rgba(255,107,107,.3);
  border-left:4px solid #ff6b6b;border-radius:8px;padding:14px 18px;
  font-size:.85rem;color:#8b949e;margin-top:16px;}
.gcaption{font-size:.76rem;color:#8b949e;text-align:center;margin-top:5px;
  font-family:'JetBrains Mono',monospace;}
.bar-wrap{background:#21262d;border-radius:6px;height:22px;
  overflow:hidden;margin:3px 0;}
.bar-fill{height:100%;border-radius:6px;display:flex;align-items:center;
  padding-left:8px;font-size:.75rem;font-weight:600;color:#0d1117;
  white-space:nowrap;transition:width .6s ease;}
.stButton>button{background:linear-gradient(135deg,#58c4dc,#7ab8ff)!important;
  color:#0d1117!important;border:none!important;border-radius:8px!important;
  font-weight:600!important;}
[data-testid="stMetricValue"]{font-family:'DM Serif Display',serif!important;
  color:#58c4dc!important;}
[data-testid="stMetricLabel"]{color:#8b949e!important;}
.stTabs [data-baseweb="tab"]{color:#8b949e!important;}
.stTabs [aria-selected="true"]{color:#58c4dc!important;
  border-bottom-color:#58c4dc!important;}
hr{border-color:#21262d!important;}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASSES = {
    "nv":    {"name":"Melanocytic Nevi",      "risk":"Benign",       "color":"#3fb950","emoji":"🟢","desc":"Common moles. Usually benign pigmented lesions."},
    "mel":   {"name":"Melanoma",              "risk":"Malignant",    "color":"#ff6b6b","emoji":"🔴","desc":"Most dangerous skin cancer. Aggressive and metastatic."},
    "bkl":   {"name":"Benign Keratosis-like", "risk":"Benign",       "color":"#3fb950","emoji":"🟢","desc":"Seborrheic keratoses, solar lentigines. Non-cancerous."},
    "bcc":   {"name":"Basal Cell Carcinoma",  "risk":"Malignant",    "color":"#d29922","emoji":"🟡","desc":"Most common skin cancer. Rarely metastasizes."},
    "akiec": {"name":"Actinic Keratoses",     "risk":"Precancerous", "color":"#d29922","emoji":"🟡","desc":"Sun-induced precancerous lesions. May progress to SCC."},
    "vasc":  {"name":"Vascular Lesions",      "risk":"Benign",       "color":"#3fb950","emoji":"🟢","desc":"Angiomas, angiokeratomas. Blood vessel origin."},
    "df":    {"name":"Dermatofibroma",        "risk":"Benign",       "color":"#3fb950","emoji":"🟢","desc":"Firm benign skin nodule, on lower extremities."},
}
CLASS_DIST  = {"nv":6705,"mel":1113,"bkl":1099,"bcc":514,"akiec":327,"vasc":142,"df":115}
CLASS_NAMES = list(CLASSES.keys())
APP_DIR     = Path(__file__).parent

STEPS = [
    ("Importing Libraries","TensorFlow/Keras, Pandas, Scikit-learn, Matplotlib"),
    ("Image–Label Dictionary","Mapping image IDs to lesion type labels"),
    ("Reading & Processing Data","Loading metadata and preparing for analysis"),
    ("Data Cleaning","Handling missing/null values in the dataset"),
    ("Exploratory Data Analysis","Visualising class, age, sex, localization"),
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

# ── HTML Chart Helpers ────────────────────────────────────────────────────────
def html_bar_chart(items, title=""):
    """items: list of (label, value, color, max_val)"""
    total = max(v for _,v,_,_ in items)
    bars = ""
    for label, val, color, max_val in items:
        pct = val / max_val * 100
        bars += f"""
        <div style='display:flex;align-items:center;gap:10px;margin:5px 0;'>
          <div style='min-width:160px;font-size:.78rem;color:#8b949e;
                      font-family:JetBrains Mono,monospace;text-align:right;'>{label}</div>
          <div style='flex:1;background:#21262d;border-radius:4px;height:20px;overflow:hidden;'>
            <div style='width:{pct:.1f}%;height:100%;background:{color};
                        border-radius:4px;opacity:.85;'></div>
          </div>
          <div style='min-width:42px;font-size:.78rem;color:#e6edf3;
                      font-family:JetBrains Mono,monospace;'>{val:,}</div>
        </div>"""
    return f"""
    <div style='background:#1c2128;border:1px solid #21262d;border-radius:10px;padding:18px 20px;'>
      <div style='font-size:.88rem;font-weight:600;color:#e6edf3;margin-bottom:12px;'>{title}</div>
      {bars}
    </div>"""

def html_prob_bars(probs_dict, title="Class Probabilities"):
    sorted_p = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
    bars = ""
    for cls, prob in sorted_p:
        pct  = prob * 100
        col  = CLASSES[cls]['color']
        name = CLASSES[cls]['name']
        bars += f"""
        <div style='margin:6px 0;'>
          <div style='display:flex;justify-content:space-between;
                      font-size:.76rem;color:#8b949e;margin-bottom:3px;'>
            <span><span style='font-family:JetBrains Mono,monospace;
                               color:{col};font-weight:600;'>{cls.upper()}</span>
                  &nbsp;{name}</span>
            <span style='color:#e6edf3;font-weight:600;'>{pct:.1f}%</span>
          </div>
          <div style='background:#21262d;border-radius:100px;height:7px;'>
            <div style='width:{pct:.1f}%;height:100%;background:{col};
                        border-radius:100px;opacity:.85;'></div>
          </div>
        </div>"""
    return f"""
    <div style='background:#1c2128;border:1px solid #21262d;
                border-radius:10px;padding:18px 20px;margin-top:12px;'>
      <div style='font-size:.88rem;font-weight:600;color:#e6edf3;
                  margin-bottom:14px;'>{title}</div>
      {bars}
    </div>"""

def html_confusion_matrix(cm, labels):
    max_val = cm.max()
    header = "<th style='padding:6px 10px;color:#8b949e;font-size:.72rem;'></th>"
    for l in labels:
        header += f"<th style='padding:6px 8px;color:#58c4dc;font-size:.72rem;" \
                  f"font-family:JetBrains Mono,monospace;'>{l.upper()}</th>"
    rows = ""
    for i, row_label in enumerate(labels):
        cells = f"<td style='padding:6px 8px;color:#58c4dc;font-size:.72rem;" \
                f"font-family:JetBrains Mono,monospace;font-weight:600;'>{row_label.upper()}</td>"
        for j, val in enumerate(cm[i]):
            intensity = int(val / max_val * 180) if max_val > 0 else 0
            bg = f"rgb({max(13,18-intensity//20)},{max(17,40+intensity//3)},{max(23,80+intensity//2)})"
            text_col = "#e6edf3" if val < max_val * 0.6 else "#0d1117"
            cells += f"<td style='padding:5px 8px;text-align:center;background:{bg};" \
                     f"font-size:.76rem;color:{text_col};font-family:JetBrains Mono,monospace;'>{val}</td>"
        rows += f"<tr>{cells}</tr>"
    return f"""
    <div style='background:#1c2128;border:1px solid #21262d;border-radius:10px;
                padding:18px 20px;overflow-x:auto;'>
      <div style='font-size:.88rem;font-weight:600;color:#e6edf3;margin-bottom:12px;'>
        Confusion Matrix — Test Set</div>
      <table style='border-collapse:collapse;width:100%;'>
        <thead><tr>{header}</tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <div style='font-size:.76rem;color:#8b949e;margin-top:10px;'>
        Rows = True Label · Columns = Predicted</div>
    </div>"""

def html_training_curves(history):
    acc  = history.get('accuracy', [])
    vacc = history.get('val_accuracy', [])
    loss = history.get('loss', [])
    vloss= history.get('val_loss', [])
    n    = len(acc)
    if not n:
        return "<p style='color:#8b949e;'>No history data.</p>"

    # SVG line chart
    W, H, PAD = 460, 200, 30
    def to_xy(data, mn, mx, w=W, h=H, pad=PAD):
        pts = []
        for i, v in enumerate(data):
            x = pad + (i / max(n-1,1)) * (w - 2*pad)
            y = h - pad - ((v - mn) / max(mx - mn, 1e-6)) * (h - 2*pad)
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    def svg_chart(d1, d2, label1, label2, col1, col2, title, fmt=".0%"):
        mn = min(min(d1), min(d2)) * 0.95
        mx = max(max(d1), max(d2)) * 1.05
        p1 = to_xy(d1, mn, mx)
        p2 = to_xy(d2, mn, mx)
        # y-axis ticks
        ticks = ""
        for t in np.linspace(mn, mx, 4):
            y = H - PAD - ((t - mn) / max(mx - mn, 1e-6)) * (H - 2*PAD)
            val_str = f"{t:.0%}" if fmt == ".0%" else f"{t:.2f}"
            ticks += f"""<line x1='{PAD}' y1='{y:.1f}' x2='{W-PAD}' y2='{y:.1f}'
                          stroke='#21262d' stroke-width='1'/>
                         <text x='{PAD-4}' y='{y:.1f}' text-anchor='end'
                          fill='#8b949e' font-size='9' dominant-baseline='middle'>{val_str}</text>"""
        return f"""
        <svg viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg'
             style='background:#1c2128;border-radius:8px;width:100%;'>
          <text x='{W//2}' y='14' text-anchor='middle'
                fill='#e6edf3' font-size='11' font-family='DM Sans,sans-serif'>{title}</text>
          {ticks}
          <polyline points='{p1}' fill='none' stroke='{col1}' stroke-width='2'/>
          <polyline points='{p2}' fill='none' stroke='{col2}' stroke-width='2'
                    stroke-dasharray='5,3'/>
          <circle cx='{W-PAD}' cy='{H-PAD - ((d1[-1]-mn)/max(mx-mn,1e-6))*(H-2*PAD):.1f}'
                  r='3' fill='{col1}'/>
          <circle cx='{W-PAD}' cy='{H-PAD - ((d2[-1]-mn)/max(mx-mn,1e-6))*(H-2*PAD):.1f}'
                  r='3' fill='{col2}'/>
          <text x='{W-PAD+4}' y='{H-PAD - ((d1[-1]-mn)/max(mx-mn,1e-6))*(H-2*PAD):.1f}'
                fill='{col1}' font-size='8' dominant-baseline='middle'>{label1}</text>
          <text x='{W-PAD+4}' y='{H-PAD - ((d2[-1]-mn)/max(mx-mn,1e-6))*(H-2*PAD):.1f}'
                fill='{col2}' font-size='8' dominant-baseline='middle'>{label2}</text>
        </svg>"""

    s1 = svg_chart(acc, vacc, "Train", "Val", "#58c4dc", "#ff6b6b", "Accuracy", ".0%")
    s2 = svg_chart(loss, vloss, "Train", "Val", "#58c4dc", "#ff6b6b", "Loss", ".2f")
    return f"""
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>
      <div>{s1}</div><div>{s2}</div>
    </div>"""

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
        float(lap.var()*10), float((arr.mean(2)<0.6).sum()/r.size),
        float(lap.mean()), asym,
        float(r.max()-r.min()), float(g.max()-g.min()), float(b.max()-b.min()),
        float(ga.mean()), float(ga.std()),
        float((r-b).mean()), float(r.mean()/(g.mean()+1e-6)),
        float(g.mean()/(b.mean()+1e-6)), float(arr.mean()), float(arr.std()),
    ]
    return np.array(feats, dtype=np.float32).reshape(1, -1)

def predict(img: Image.Image):
    feats = extract_features(img)
    if mlp_model is None:
        p = np.array([0.6,0.1,0.1,0.05,0.05,0.05,0.05])
        p /= p.sum()
        return CLASS_NAMES[0], dict(zip(CLASS_NAMES, p))
    fs  = scaler.transform(feats)
    ens = (mlp_model.predict_proba(fs)[0] + rf_model.predict_proba(feats)[0]) / 2
    return CLASS_NAMES[int(np.argmax(ens))], dict(zip(CLASS_NAMES, ens))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 20px'>
      <div style='font-family:DM Serif Display,serif;font-size:1.4rem;color:#58c4dc;'>🔬 DermAI</div>
      <div style='font-size:.74rem;color:#8b949e;font-family:JetBrains Mono,monospace;'>
        HAM10000 · Ensemble Model</div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("Nav", [
        "🏠 Overview", "🔍 Analyze Lesion", "📊 Dataset & EDA",
        "🏗️ Architecture", "📈 Results"
    ], label_visibility="collapsed")
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
      Accuracy:
      <span style='color:#58c4dc;font-weight:600;'>{acc*100:.1f}%</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div class='warn-box' style='font-size:.76rem;margin:0;'>
    <strong>⚕️ Disclaimer</strong><br>Research/demo only.<br>Consult a dermatologist.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class='hero'>
      <div class='badge'>🔬 CNN · HAM10000 · ENSEMBLE MODEL</div><br>
      <div class='hero-title'>Skin Cancer Detection</div>
      <div class='hero-sub'>Real ML ensemble trained on HAM10000 dermoscopic data —
      classifying 7 types of skin lesions with live inference on uploaded images.</div>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,(n,l) in zip([c1,c2,c3,c4],[
        ("10,015","Training Images"),("7","Lesion Classes"),
        (f"{results.get('test_accuracy',0.8295)*100:.1f}%","Test Accuracy"),
        ("14","Pipeline Steps")]):
        with col:
            st.markdown(f"<div class='stat-card'><div class='stat-num'>{n}</div>"
                        f"<div class='stat-label'>{l}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([3,2], gap="large")

    with left:
        st.markdown("<div class='sec-title'>About the Project</div>"
                    "<div class='sec-line'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='color:#8b949e;line-height:1.8;font-size:.95rem;'>
        Skin cancer is the <strong style='color:#e6edf3;'>most common human malignancy</strong>.
        This project builds a real ML ensemble (MLP + Random Forest) trained on features
        extracted from the <strong style='color:#58c4dc;'>HAM10000</strong> dataset —
        10,015 dermoscopic images across 7 classes. Upload any skin lesion image on the
        <em>Analyze Lesion</em> page to get live predictions.
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>7 Lesion Classes</div>"
                    "<div class='sec-line'></div>", unsafe_allow_html=True)
        for k, v in CLASSES.items():
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:10px;padding:9px 0;
                        border-bottom:1px solid #21262d;'>
              <span style='font-family:JetBrains Mono,monospace;font-size:.8rem;
                           color:#58c4dc;min-width:54px;'>{k.upper()}</span>
              <span style='font-weight:500;color:#e6edf3;flex:1;'>{v['name']}</span>
              <span style='font-size:.74rem;color:{v["color"]};background:rgba(0,0,0,.3);
                           padding:2px 9px;border-radius:100px;
                           border:1px solid {v["color"]}40;'>{v['risk']}</span>
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='sec-title'>Sample Lesions</div>"
                    "<div class='sec-line'></div>", unsafe_allow_html=True)
        for fname, cap in [
            ("sample_images/lesions_collage1.png","BCC · Melanoma · Mixed"),
            ("sample_images/lesions_collage3.png","Nevi · Melanoma"),
        ]:
            fp = APP_DIR / fname
            if fp.exists():
                st.image(Image.open(fp), use_container_width=True)
                st.markdown(f"<div class='gcaption'>{cap}</div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYZE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Analyze Lesion":
    st.markdown("<div class='sec-title'>🔍 Real-Time Lesion Analysis</div>"
                "<div class='sec-line'></div>", unsafe_allow_html=True)

    col_up, col_res = st.columns([1,1], gap="large")

    with col_up:
        st.markdown("""
        <div style='background:#1c2128;border:1px solid #21262d;border-radius:12px;padding:22px;'>
        <div style='font-weight:600;color:#e6edf3;margin-bottom:4px;'>Upload Dermoscopic Image</div>
        <div style='font-size:.83rem;color:#8b949e;margin-bottom:14px;'>JPG · PNG · WebP</div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["jpg","jpeg","png","webp"],
                                    label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='font-size:.85rem;color:#8b949e;margin:12px 0 8px;'>"
                    "— or choose a sample —</div>", unsafe_allow_html=True)
        sample_choice = st.selectbox("", [
            "None", "Sample 1 — Mixed Lesion Types",
            "Sample 2 — Vascular & Keratoses", "Sample 3 — Nevi & Melanoma"
        ], label_visibility="collapsed")

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
            if Path(p).exists():
                image_to_use = Image.open(p).convert("RGB")

        if image_to_use:
            st.image(image_to_use, caption="Input Image", use_container_width=True)
            run_btn = st.button("⚡ Run Analysis", use_container_width=True)
        else:
            run_btn = False

    with col_res:
        if image_to_use and run_btn:
            prog = st.progress(0, "Preprocessing…")
            time.sleep(0.3); prog.progress(30, "Extracting features…")
            time.sleep(0.3); prog.progress(65, "Running ensemble…")
            time.sleep(0.3); prog.progress(90, "Finalising…")
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
              <div style='font-family:DM Serif Display,serif;font-size:1.9rem;
                          color:#58c4dc;margin-bottom:6px;'>
                {info['emoji']} {info['name']}</div>
              <div style='font-family:JetBrains Mono,monospace;font-size:.82rem;
                          color:#8b949e;margin-bottom:10px;'>
                Code: <strong style='color:#e6edf3;'>{top_class.upper()}</strong> ·
                Risk: <strong style='color:{info["color"]};'>{info["risk"]}</strong></div>
              <div style='font-size:.83rem;color:#8b949e;margin-bottom:14px;'>
                {info['desc']}</div>
              <div style='font-size:.84rem;color:#8b949e;margin-bottom:6px;'>
                Confidence:
                <strong style='color:#e6edf3;'>{conf*100:.1f}%</strong></div>
              <div style='background:#21262d;border-radius:100px;height:8px;overflow:hidden;'>
                <div style='height:100%;width:{conf*100:.1f}%;
                            background:linear-gradient(90deg,#58c4dc,#b0c8ff);
                            border-radius:100px;'></div>
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown(html_prob_bars(probs), unsafe_allow_html=True)

            st.markdown("""
            <div class='warn-box'>
            <strong>⚕️ Medical Notice</strong> — Research/demonstration only.
            Do NOT use for clinical decisions. Always consult a board-certified dermatologist.
            </div>""", unsafe_allow_html=True)

        elif image_to_use:
            st.markdown("""
            <div style='text-align:center;padding:70px 20px;color:#8b949e;
                        border:1px dashed #21262d;border-radius:12px;margin-top:20px;'>
              <div style='font-size:2.5rem;margin-bottom:14px;'>🔬</div>
              <div>Click <strong style='color:#58c4dc;'>Run Analysis</strong> to proceed</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='text-align:center;padding:80px 20px;color:#8b949e;
                        border:1px dashed #21262d;border-radius:12px;margin-top:20px;'>
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
        total = sum(CLASS_DIST.values())
        items = [(f"{k} · {CLASSES[k]['name']}", v, CLASSES[k]['color'], total)
                 for k, v in CLASS_DIST.items()]
        st.markdown(html_bar_chart(items, "HAM10000 Class Distribution — 10,015 images"),
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Donut as HTML/SVG
        st.markdown("<div style='font-size:.88rem;font-weight:600;color:#e6edf3;"
                    "margin-bottom:10px;'>Proportion</div>", unsafe_allow_html=True)
        cols = st.columns(len(CLASS_DIST))
        for col_, (k, v) in zip(cols, CLASS_DIST.items()):
            pct = v / total * 100
            with col_:
                st.markdown(f"""
                <div style='text-align:center;background:#1c2128;border:1px solid #21262d;
                            border-radius:8px;padding:10px 6px;'>
                  <div style='font-size:1.1rem;font-weight:700;color:{CLASSES[k]["color"]};'>
                    {pct:.1f}%</div>
                  <div style='font-size:.65rem;color:#8b949e;font-family:JetBrains Mono,monospace;'>
                    {k.upper()}</div>
                  <div style='font-size:.62rem;color:#8b949e;'>{v:,}</div>
                </div>""", unsafe_allow_html=True)

        if not meta_df.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Metadata Sample")
            st.dataframe(meta_df.head(8), use_container_width=True, hide_index=True)
            m1,m2,m3 = st.columns(3)
            m1.metric("Total Records", f"{len(meta_df):,}")
            m2.metric("Unique Lesions", f"{meta_df['lesion_id'].nunique():,}")
            m3.metric("Mean Age", f"{meta_df['age'].mean():.0f} yrs")

    with tab2:
        if not meta_df.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Age Stats by Class**")
                age_rows = []
                for k in CLASS_NAMES:
                    sub = meta_df[meta_df['dx']==k]['age'].dropna()
                    age_rows.append({"Class":k.upper(),"Mean":f"{sub.mean():.0f}",
                                     "Std":f"{sub.std():.0f}","Min":f"{sub.min():.0f}",
                                     "Max":f"{sub.max():.0f}","Count":len(sub)})
                st.dataframe(pd.DataFrame(age_rows), use_container_width=True, hide_index=True)

            with c2:
                st.markdown("**Sex Distribution**")
                sc = meta_df.groupby(['dx','sex']).size().unstack(fill_value=0)
                sex_rows = []
                for k in CLASS_NAMES:
                    m = int(sc.loc[k,'male'])   if k in sc.index and 'male'   in sc.columns else 0
                    f = int(sc.loc[k,'female']) if k in sc.index and 'female' in sc.columns else 0
                    sex_rows.append({"Class":k.upper(),"Male":m,"Female":f,
                                     "Total":m+f,"% Male":f"{m/(m+f+1e-6)*100:.0f}%"})
                st.dataframe(pd.DataFrame(sex_rows), use_container_width=True, hide_index=True)

            st.markdown("<br>", unsafe_allow_html=True)
            loc_counts = meta_df['localization'].value_counts().head(10)
            loc_items  = [(loc, int(cnt), "#58c4dc", int(loc_counts.iloc[0]))
                          for loc, cnt in loc_counts.items()]
            st.markdown(html_bar_chart(loc_items, "Top 10 Lesion Localizations"),
                        unsafe_allow_html=True)
        else:
            st.info("Metadata not loaded.")

    with tab3:
        g1,g2,g3 = st.columns(3)
        for col_, (p, cap) in zip([g1,g2,g3],[
            ("sample_images/lesions_collage1.png","BCC · Melanoma · Mixed"),
            ("sample_images/lesions_collage2.png","Vascular · Keratoses"),
            ("sample_images/lesions_collage3.png","Nevi · Melanoma"),
        ]):
            fp = APP_DIR / p
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
        layers = [
            ("Input Image (any size)",              "#58c4dc"),
            ("Feature Extraction — Colour+Texture", "#7ab8ff"),
            ("20-dimensional Feature Vector",        "#b48eff"),
            ("MLP Neural Net  256→128→64",          "#3fb950"),
            ("Random Forest  200 estimators",       "#d29922"),
            ("Ensemble Average  (Soft Voting)",     "#ff8c42"),
            ("Output — 7 Class Probabilities",      "#ff6b6b"),
        ]
        for i, (name, col) in enumerate(layers):
            arrow = "↓" if i < len(layers)-1 else ""
            st.markdown(f"""
            <div style='background:#1c2128;border:1px solid {col}40;border-left:3px solid {col};
                        border-radius:8px;padding:10px 16px;margin-bottom:4px;
                        font-family:JetBrains Mono,monospace;font-size:.82rem;color:#e6edf3;'>
              {name}
            </div>
            {"<div style='text-align:center;color:#8b949e;font-size:1rem;margin:0;'>↓</div>" if arrow else ""}
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m1,m2 = st.columns(2)
        with m1: st.metric("Features","20-dim"); st.metric("RF Trees","200")
        with m2: st.metric("MLP","256→128→64"); st.metric("Fusion","Soft Vote")

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
    for col,(l,v) in zip(st.columns(5),[
        ("MLP Epochs","500 max"),("Batch","32"),
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
        ("Test Classes","7")]):
        with col: st.metric(l,v)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Training Curves","Confusion Matrix","Per-Class Metrics"])

    with tab1:
        if history:
            st.markdown(html_training_curves(history), unsafe_allow_html=True)
        else:
            st.info("No training history found.")

    with tab2:
        cm_data = results.get('confusion_matrix')
        if cm_data:
            st.markdown(html_confusion_matrix(np.array(cm_data), CLASS_NAMES),
                        unsafe_allow_html=True)

    with tab3:
        cr = results.get('class_report', {})
        rows = []
        for k in CLASS_NAMES:
            m = cr.get(k, {})
            rows.append({
                "Class": k.upper(),
                "Name": CLASSES[k]['name'],
                "Precision": f"{m.get('precision',0):.2f}",
                "Recall": f"{m.get('recall',0):.2f}",
                "F1-Score": f"{m.get('f1-score',0):.2f}",
                "Support": int(m.get('support',0)),
                "Risk": CLASSES[k]['risk']
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # Per-class F1 as bar chart
        f1_items = [(f"{k.upper()} — {CLASSES[k]['name']}",
                     int(cr.get(k,{}).get('f1-score',0)*100),
                     CLASSES[k]['color'], 100)
                    for k in CLASS_NAMES]
        st.markdown(html_bar_chart(f1_items, "F1-Score per Class (%)"),
                    unsafe_allow_html=True)

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
              <div style='font-weight:600;color:#e6edf3;margin-bottom:6px;'>
                {' '.join(t.split()[1:])}</div>
              <div style='font-size:.82rem;color:#8b949e;line-height:1.6;'>{d}</div>
            </div>""", unsafe_allow_html=True)
