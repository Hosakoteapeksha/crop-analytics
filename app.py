import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_KERAS"] = "1"

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Crop Analytics – PMFBY",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
:root {
    --green-dark:   #0a3d1f;
    --green-mid:    #145a32;
    --green-accent: #27ae60;
    --green-light:  #a9dfbf;
    --gold:         #f0c040;
    --cream:        #fdf6e3;
    --card-bg:      rgba(255,255,255,0.06);
    --card-border:  rgba(39,174,96,0.3);
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--green-dark);
    color: var(--cream);
}
.hero-header {
    background: linear-gradient(135deg, #0a3d1f 0%, #145a32 60%, #0d5c2e 100%);
    border-bottom: 2px solid var(--green-accent);
    border-radius: 0 0 24px 24px;
    padding: 2.5rem 2rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
}
.hero-badge {
    display: inline-block;
    background: var(--gold);
    color: var(--green-dark);
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.6rem, 3.5vw, 2.8rem);
    font-weight: 800;
    color: #ffffff;
    line-height: 1.2;
    margin: 0 0 0.5rem;
}
.hero-title span { color: var(--green-accent); }
.hero-sub { font-size: 0.95rem; color: var(--green-light); opacity: 0.85; margin: 0; }
.upload-card {
    background: var(--card-bg);
    border: 1.5px dashed var(--card-border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.model-card {
    background: linear-gradient(145deg, rgba(20,90,50,0.5), rgba(10,61,31,0.7));
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 1.4rem 1.2rem;
    height: 100%;
}
.model-icon { font-size: 2.2rem; margin-bottom: 0.5rem; }
.model-title { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1rem; margin-bottom: 0.3rem; }
.model-desc { font-size: 0.8rem; color: var(--green-light); opacity: 0.8; margin-bottom: 0.8rem; }
.model-tag {
    display: inline-block; background: rgba(39,174,96,0.2); border: 1px solid rgba(39,174,96,0.4);
    color: var(--green-accent); font-size: 0.68rem; font-weight: 600; padding: 2px 10px;
    border-radius: 20px; margin-bottom: 1rem;
}
.result-box {
    background: rgba(20,90,50,0.4); border: 1px solid var(--green-accent);
    border-radius: 12px; padding: 1.2rem 1.4rem; margin-top: 1.2rem;
}
.result-box h4 { font-family: 'Syne', sans-serif; font-size: 0.85rem; color: var(--green-accent); margin: 0 0 0.6rem; }
.result-item { display: flex; justify-content: space-between; padding: 0.3rem 0; border-bottom: 1px solid rgba(39,174,96,0.15); font-size: 0.88rem; }
.result-item:last-child { border-bottom: none; }
.result-label { color: var(--green-light); }
.result-value { font-weight: 600; color: #fff; }
.result-value.danger { color: #e74c3c; }
.result-value.ok { color: var(--green-accent); }
.result-value.warn { color: var(--gold); }
.conf-bar-bg { background: rgba(255,255,255,0.1); border-radius: 6px; height: 6px; margin-top: 4px; }
.conf-bar-fill { height: 6px; border-radius: 6px; background: linear-gradient(90deg, var(--green-accent), var(--gold)); }
.stButton > button {
    background: linear-gradient(135deg, var(--green-mid), var(--green-accent)) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    width: 100% !important; transition: opacity 0.2s !important;
}
.section-divider { border: none; border-top: 1px solid rgba(39,174,96,0.2); margin: 1.8rem 0; }
.img-label { font-size: 0.75rem; color: var(--green-light); text-align: center; margin-top: 0.4rem; opacity: 0.7; }
.warning-box {
    background: rgba(231,76,60,0.15); border: 1px solid rgba(231,76,60,0.4);
    border-radius: 10px; padding: 0.8rem 1.2rem; font-size: 0.82rem;
    color: #e74c3c; margin-top: 0.5rem;
}
.info-box {
    background: rgba(240,192,64,0.1); border: 1px solid rgba(240,192,64,0.3);
    border-radius: 10px; padding: 0.8rem 1.2rem; font-size: 0.82rem;
    color: var(--gold); margin-top: 1rem;
}
.analyze-btn-wrap { margin-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ─── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">🇮🇳 Government of India &nbsp;|&nbsp; PMFBY Initiative</div>
    <h1 class="hero-title">AI-Based Real-Time<br><span>Crop Image Analytics</span></h1>
    <p class="hero-sub">Pradhan Mantri Fasal Bima Yojana &nbsp;·&nbsp; Powered by Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────
for key in ["uploaded_image", "analysis_result"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ════════════════════════════════════════════════════════════════════════════
# CLASS LABELS  (PlantVillage 38-class standard used by crop_model.h5)
# ════════════════════════════════════════════════════════════════════════════

PLANT_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

CROP_SCIENTIFIC = {
    "Apple":      "Malus domestica",
    "Blueberry":  "Vaccinium sect. Cyanococcus",
    "Cherry":     "Prunus avium",
    "Corn":       "Zea mays",
    "Grape":      "Vitis vinifera",
    "Orange":     "Citrus sinensis",
    "Peach":      "Prunus persica",
    "Pepper":     "Capsicum annuum",
    "Potato":     "Solanum tuberosum",
    "Raspberry":  "Rubus idaeus",
    "Soybean":    "Glycine max",
    "Squash":     "Cucurbita pepo",
    "Strawberry": "Fragaria ananassa",
    "Tomato":     "Solanum lycopersicum",
}

CROP_SEASONS = {
    "Corn":       "Kharif (June – October)",
    "Potato":     "Rabi (October – March)",
    "Tomato":     "Rabi / Year-round",
    "Soybean":    "Kharif (June – September)",
    "Apple":      "Rabi / Zaid",
    "Grape":      "Annual",
    "Strawberry": "Rabi (October – March)",
    "Pepper":     "Kharif (July – November)",
    "Peach":      "Rabi (March – June)",
    "Orange":     "Annual",
    "Blueberry":  "Zaid (March – June)",
    "Cherry":     "Zaid (April – June)",
    "Raspberry":  "Zaid / Annual",
    "Squash":     "Kharif / Rabi",
}

# Growth stage heuristics mapped per crop type
GROWTH_STAGES = [
    "Germination / Seedling",
    "Vegetative Growth",
    "Tillering / Branching",
    "Stem Extension",
    "Heading / Flowering",
    "Grain Filling / Fruiting",
    "Maturity / Harvest Ready",
]

GROWTH_ADVICE = {
    "Germination / Seedling":   "Ensure adequate moisture. Avoid waterlogging. Thin seedlings if overcrowded.",
    "Vegetative Growth":        "Apply nitrogen-rich fertiliser. Monitor for early pest infestation.",
    "Tillering / Branching":    "Irrigation is critical at this stage. Watch for stem borer and aphids.",
    "Stem Extension":           "Apply potassium fertiliser. Provide support for tall varieties.",
    "Heading / Flowering":      "Avoid pesticide spray during flowering. Critical water requirement period.",
    "Grain Filling / Fruiting": "Ensure consistent moisture. Monitor for fungal diseases and fruit borer.",
    "Maturity / Harvest Ready": "Reduce irrigation gradually. Prepare harvesting equipment. Monitor for storage pests.",
}

LOW_CONF_THRESHOLD = 35


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ════════════════════════════════════════════════════════════════════════════

def _rebuild_model_from_h5(src_path):
    """Rebuild MobileNetV2 model and transplant weights from h5. No name= arg (TF 2.21 compat)."""
    import h5py

    inp  = tf.keras.Input(shape=(224, 224, 3))
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights=None
    )
    base.trainable = False
    x   = base(inp)
    x   = tf.keras.layers.GlobalAveragePooling2D()(x)
    x   = tf.keras.layers.Dense(128, activation="relu")(x)
    out = tf.keras.layers.Dense(38,  activation="softmax")(x)
    model = tf.keras.Model(inputs=inp, outputs=out)

    with h5py.File(src_path, "r") as f:
        mw = f["model_weights"]

        # Load MobileNetV2 backbone weights
        mobilenet = model.layers[1]
        mn_grp = mw["mobilenetv2_1.00_224"]
        for sub in mobilenet.layers:
            if sub.name not in mn_grp:
                continue
            grp = mn_grp[sub.name]
            w_dict = {}
            grp.visititems(lambda n, o: w_dict.update({n: o[()]})
                           if isinstance(o, h5py.Dataset) else None)
            kernels = [v for k, v in w_dict.items() if "kernel" in k]
            biases  = [v for k, v in w_dict.items() if "bias"   in k]
            gammas  = [v for k, v in w_dict.items() if "gamma"  in k]
            betas   = [v for k, v in w_dict.items() if "beta"   in k]
            means   = [v for k, v in w_dict.items() if "moving_mean"     in k]
            vars_   = [v for k, v in w_dict.items() if "moving_variance" in k]
            try:
                if kernels and biases:
                    sub.set_weights([kernels[0], biases[0]])
                elif kernels:
                    sub.set_weights(kernels)
                elif gammas and betas and means and vars_:
                    sub.set_weights([gammas[0], betas[0], means[0], vars_[0]])
            except Exception:
                pass

        # Load Dense weights
        for lname, layer in [("dense", model.layers[3]), ("dense_1", model.layers[4])]:
            grp = mw[lname]["sequential"]
            w_dict = {}
            grp.visititems(lambda n, o: w_dict.update({n: o[()]})
                           if isinstance(o, h5py.Dataset) else None)
            kernels = [v for k, v in w_dict.items() if "kernel" in k]
            biases  = [v for k, v in w_dict.items() if "bias"   in k]
            if kernels and biases:
                layer.set_weights([kernels[0], biases[0]])

    return model


@st.cache_resource(show_spinner=False)
def load_crop_model():
    """Load crop model. Tries crop_model_v2.keras first, then rebuilds from h5."""
    # Best option: pre-converted modern keras file
    if os.path.exists("crop_model_v2.keras"):
        try:
            model = tf.keras.models.load_model("crop_model_v2.keras", compile=False)
            return model, None
        except Exception:
            pass

    # Fallback: rebuild from raw weights
    src = "crop_model.h5"
    if not os.path.exists(src):
        return None, "No model file found. Place 'crop_model.h5' in the project folder."
    try:
        model = _rebuild_model_from_h5(src)
        return model, None
    except Exception as e:
        return None, f"Could not load model: {e}"


# ════════════════════════════════════════════════════════════════════════════
# INFERENCE — single model, three outputs
# ════════════════════════════════════════════════════════════════════════════

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """Resize, convert to array, and apply MobileNetV2 preprocessing."""
    img = image.resize(target_size)
    arr = img_to_array(img)
    arr = preprocess_input(np.expand_dims(arr, axis=0))
    return arr


def infer(image: Image.Image, model) -> dict:
    """
    Run the crop_model.h5 on the image.
    Returns a comprehensive result dict with:
      - crop_name, scientific_name, season
      - is_healthy, health_status, disease_name
      - severity, affected_area, insurance_flag
      - growth_stage, growth_index, growth_advice
      - confidence, low_conf
    """
    arr = preprocess_image(image)
    preds = model.predict(arr, verbose=0)
    pred_idx   = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))
    conf_pct   = int(confidence * 100)
    top5_idx   = np.argsort(preds[0])[::-1][:5]
    top5       = [(PLANT_CLASSES[i] if i < len(PLANT_CLASSES) else f"Class {i}",
                   round(float(preds[0][i]) * 100, 1)) for i in top5_idx]

    # ── Low confidence guard ──────────────────────────────────────────────
    if conf_pct < LOW_CONF_THRESHOLD:
        return {
            "low_conf": True,
            "confidence": conf_pct,
            "top5": top5,
            "error_msg": f"Model confidence is only {conf_pct}% — image may not show a supported crop leaf clearly.",
        }

    # ── Parse predicted class ─────────────────────────────────────────────
    predicted_class = PLANT_CLASSES[pred_idx] if pred_idx < len(PLANT_CLASSES) else f"Class {pred_idx}"
    parts = predicted_class.split("___")

    raw_crop  = parts[0]                              # e.g. "Corn_(maize)"
    raw_cond  = parts[1] if len(parts) > 1 else "healthy"

    # Clean crop name: remove parenthetical suffixes, underscores
    crop_name = raw_crop.split("(")[0].replace("_", " ").replace(",", "").strip()
    crop_key  = crop_name.split()[0]                  # first word for dict lookup

    disease_raw = raw_cond.replace("_", " ").strip()
    is_healthy  = "healthy" in raw_cond.lower()
    health_str  = "✅ Healthy" if is_healthy else f"⚠️ {disease_raw}"

    # ── Lookup metadata ───────────────────────────────────────────────────
    sci_name = CROP_SCIENTIFIC.get(crop_key, "Data unavailable")
    season   = CROP_SEASONS.get(crop_key, "Data unavailable")

    # ── Damage / severity ─────────────────────────────────────────────────
    if is_healthy:
        severity      = "None"
        affected_pct  = "0%"
        insurance_flag= "Not Applicable"
        dmg_label     = "None detected"
    else:
        if conf_pct >= 80:
            severity     = "High"
            affected_pct = "60–80%"
        elif conf_pct >= 60:
            severity     = "Moderate"
            affected_pct = "30–60%"
        else:
            severity     = "Low"
            affected_pct = "10–30%"
        insurance_flag = "Eligible for Claim"
        dmg_label      = disease_raw

    # ── Growth stage heuristic ────────────────────────────────────────────
    # Use image greenness + confidence to estimate stage (no separate model needed)
    img_arr   = np.array(image.resize((100, 100))).astype(float)
    green_ch  = img_arr[:, :, 1].mean() / 255.0
    red_ch    = img_arr[:, :, 0].mean() / 255.0
    blue_ch   = img_arr[:, :, 2].mean() / 255.0

    # Vegetation index (approximation of NDVI-like score from RGB)
    vi = (green_ch - red_ch) / (green_ch + red_ch + 1e-6)

    if vi > 0.15:
        # Deep green → vegetative / grain filling
        if green_ch > 0.5:
            growth_idx = 5  # Grain Filling / Fruiting
        else:
            growth_idx = 2  # Vegetative Growth / Tillering
    elif vi > 0.05:
        growth_idx = 4  # Heading / Flowering
    elif red_ch > 0.55:
        growth_idx = 6  # Maturity (yellowing/reddening)
    elif green_ch < 0.3 and red_ch < 0.3:
        growth_idx = 0  # Germination / Seedling (pale/young)
    else:
        growth_idx = 3  # Stem Extension (default mid-stage)

    # Damaged crops tend to be at or before their peak stage
    if not is_healthy and growth_idx > 4:
        growth_idx = 4

    growth_stage  = GROWTH_STAGES[growth_idx]
    growth_advice = GROWTH_ADVICE[growth_stage]
    growth_pct    = int((growth_idx + 1) / len(GROWTH_STAGES) * 100)

    return {
        "low_conf":      False,
        "confidence":    conf_pct,
        "top5":          top5,
        # Crop ID
        "crop_name":     crop_name,
        "scientific_name": sci_name,
        "season":        season,
        # Health / Damage
        "is_healthy":    is_healthy,
        "health_status": health_str,
        "disease_name":  dmg_label,
        "severity":      severity,
        "affected_area": affected_pct,
        "insurance_flag":insurance_flag,
        # Growth
        "growth_stage":  growth_stage,
        "growth_index":  growth_idx,
        "growth_total":  len(GROWTH_STAGES),
        "growth_pct":    growth_pct,
        "growth_advice": growth_advice,
    }


# ════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ════════════════════════════════════════════════════════════════════════════

left_col, right_col = st.columns([1.1, 1.9], gap="large")

# ── Left: Upload ──────────────────────────────────────────────────────────
with left_col:
    st.markdown("#### 📷 Upload Crop Image")
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.uploaded_image = image
        st.image(image, use_container_width=True, caption="Uploaded crop image")
        st.markdown(
            f'<p class="img-label">📁 {uploaded_file.name} &nbsp;|&nbsp; {image.size[0]}×{image.size[1]} px</p>',
            unsafe_allow_html=True,
        )
    else:
        st.session_state.uploaded_image = None
        st.session_state.analysis_result = None
        st.image(
            "https://images.unsplash.com/photo-1574943320219-553eb213f72d?w=600&q=80",
            use_container_width=True,
            caption="Sample crop field — upload your own above",
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("**📋 System Status**")

    model, model_err = load_crop_model()
    if model is not None:
        st.success("✅ Crop AI Model — Loaded")
    else:
        st.error(f"❌ Model Error: {model_err}")
        st.markdown("""
        <div class="warning-box">
            Place <code>crop_model.h5</code> in the same folder as <code>app.py</code> and restart.
        </div>
        """, unsafe_allow_html=True)

    st.info("ℹ️ PMFBY Ready")

    # Analyse button
    img_ready = st.session_state.uploaded_image is not None and model is not None
    st.markdown('<div class="analyze-btn-wrap">', unsafe_allow_html=True)
    run_analysis = st.button("🔬 Analyse Crop", disabled=not img_ready, key="btn_analyse")
    st.markdown('</div>', unsafe_allow_html=True)

    if not img_ready and model is not None:
        st.markdown("""
        <div class="info-box">Upload a crop leaf image above to enable analysis.</div>
        """, unsafe_allow_html=True)

    if run_analysis and img_ready:
        with st.spinner("🌿 Running AI analysis…"):
            result = infer(st.session_state.uploaded_image, model)
            st.session_state.analysis_result = result


# ── Right: Results ─────────────────────────────────────────────────────────
with right_col:
    st.markdown("#### 📊 AI Analysis Results")

    r = st.session_state.analysis_result

    if r is None:
        # Placeholder cards
        c1, c2, c3 = st.columns(3, gap="medium")
        for col, icon, title, desc in [
            (c1, "🌱", "Crop Identification", "Detects crop species from leaf image"),
            (c2, "🔍", "Health & Damage", "Identifies diseases & damage severity"),
            (c3, "📈", "Growth Stage", "Estimates current phenological stage"),
        ]:
            with col:
                st.markdown(f"""
                <div class="model-card">
                    <div class="model-icon">{icon}</div>
                    <div class="model-title">{title}</div>
                    <div class="model-desc">{desc}</div>
                    <span class="model-tag">Keras · crop_model.h5</span>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box" style="margin-top:1.5rem;">
            Upload a crop leaf image and click <b>🔬 Analyse Crop</b> to get results for all three modules simultaneously.
        </div>
        """, unsafe_allow_html=True)

    elif r.get("low_conf"):
        st.markdown(f"""
        <div class="warning-box" style="margin-top:1rem;">
            <b>⚠️ Low Confidence ({r['confidence']}%)</b><br>
            {r.get('error_msg','The model could not confidently identify this image.')}
            <br><br>
            <b>Tips:</b> Upload a clear, well-lit image of a single crop leaf. Avoid blurry or distant shots.
        </div>
        """, unsafe_allow_html=True)

        if r.get("top5"):
            st.markdown("**Top predictions:**")
            for cls, prob in r["top5"]:
                parts = cls.split("___")
                label = f"{parts[0].replace('_',' ')} — {parts[1].replace('_',' ')}" if len(parts)==2 else cls
                st.markdown(f"- `{label}` : **{prob}%**")

    else:
        # ── Three result cards ────────────────────────────────────────────
        c1, c2, c3 = st.columns(3, gap="medium")

        # Card 1: Crop Identification
        with c1:
            st.markdown(f"""
            <div class="result-box">
                <h4>🌱 Crop Identification</h4>
                <div class="result-item">
                    <span class="result-label">Crop Name</span>
                    <span class="result-value ok">{r['crop_name']}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Scientific Name</span>
                    <span class="result-value" style="font-style:italic;font-size:0.8rem;">{r['scientific_name']}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Growing Season</span>
                    <span class="result-value">{r['season']}</span>
                </div>
                <div style="margin-top:0.8rem;">
                    <span class="result-label" style="font-size:0.78rem;">Model Confidence: {r['confidence']}%</span>
                    <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{r['confidence']}%"></div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Card 2: Health & Damage
        with c2:
            dmg_color = "ok" if r["is_healthy"] else "danger"
            sev_color = "ok" if r["is_healthy"] else ("danger" if r["severity"]=="High" else "warn")
            ins_color = "ok" if r["is_healthy"] else "warn"
            st.markdown(f"""
            <div class="result-box">
                <h4>🔍 Health & Damage</h4>
                <div class="result-item">
                    <span class="result-label">Health Status</span>
                    <span class="result-value {dmg_color}">{r['health_status']}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Disease / Damage</span>
                    <span class="result-value">{r['disease_name']}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Severity</span>
                    <span class="result-value {sev_color}">{r['severity']}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Affected Area</span>
                    <span class="result-value">{r['affected_area']}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Insurance Flag</span>
                    <span class="result-value {ins_color}">{r['insurance_flag']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Card 3: Growth Stage
        with c3:
            dots = ""
            for i in range(r["growth_total"]):
                active = "background:var(--green-accent);border-color:var(--green-accent);" if i <= r["growth_index"] else ""
                dots  += f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;border:2px solid rgba(39,174,96,0.4);margin:0 2px;{active}"></span>'

            st.markdown(f"""
            <div class="result-box">
                <h4>📈 Growth Stage</h4>
                <div class="result-item">
                    <span class="result-label">Current Stage</span>
                    <span class="result-value ok">{r['growth_stage']}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Stage Progress</span>
                    <span class="result-value">Stage {r['growth_index']+1} of {r['growth_total']}</span>
                </div>
                <div style="margin-top:0.8rem;">
                    <span class="result-label" style="font-size:0.78rem;">Growth Progress: {r['growth_pct']}%</span>
                    <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{r['growth_pct']}%"></div></div>
                </div>
                <div style="margin-top:0.6rem;text-align:center;">{dots}</div>
            </div>
            """, unsafe_allow_html=True)

        # Advisory box
        st.markdown(f"""
        <div class="result-box" style="margin-top:1rem;border-color:var(--gold);">
            <h4 style="color:var(--gold);">📋 Agronomic Advisory</h4>
            <p style="font-size:0.88rem;color:var(--cream);margin:0;">{r['growth_advice']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Top-5 alternative predictions expander
        if r.get("top5"):
            with st.expander("🔎 View Top-5 Model Predictions"):
                for cls, prob in r["top5"]:
                    parts = cls.split("___")
                    crop_lbl = parts[0].replace("_"," ").split("(")[0].strip()
                    cond_lbl = parts[1].replace("_"," ") if len(parts)>1 else "healthy"
                    health_icon = "✅" if "healthy" in cond_lbl.lower() else "⚠️"
                    st.markdown(f"**{health_icon} {crop_lbl}** — {cond_lbl} &nbsp; `{prob}%`")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:rgba(169,223,191,0.4);font-size:0.72rem;padding:2rem 0 1rem;letter-spacing:0.05em;">
    🌾 AI Crop Analytics Platform &nbsp;·&nbsp; PMFBY - Pradhan Mantri Fasal Bima Yojana &nbsp;·&nbsp;
    Ministry of Agriculture &amp; Farmers Welfare, Government of India
</div>
""", unsafe_allow_html=True)
