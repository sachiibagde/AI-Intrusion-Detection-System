"""
app.py - Streamlit Dashboard
AI-Based Intrusion Detection System (IDS)

Interactive UI with:
  • Dataset upload & preview
  • Model training trigger
  • Performance metrics & charts
  • Confusion matrix viewer
  • Feature importance visualization
  • Real-time attack detection simulation
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import logging

# ── Must be first Streamlit call ─────────────────────────
st.set_page_config(
    page_title="AI-IDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from preprocess import (
    generate_synthetic_nslkdd,
    full_preprocess_pipeline,
    CATEGORY_COLUMN,
)
from train import train_all_models, get_models
from utils import (
    load_model, load_artifacts, save_artifacts,
    models_exist, build_metrics_table,
    plot_accuracy_comparison, plot_metrics_radar,
    plot_confusion_matrix, plot_feature_importance,
    plot_attack_distribution, simulate_traffic,
    predict_traffic, get_severity_color, format_timestamp,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Custom CSS – Dark industrial aesthetic
# ─────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');

  :root {
    --bg:      #0d1117;
    --surface: #161b22;
    --border:  #30363d;
    --accent:  #58a6ff;
    --green:   #3fb950;
    --red:     #f85149;
    --yellow:  #d29922;
    --text:    #e6edf3;
    --muted:   #8b949e;
  }

  html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
  }
  section[data-testid="stSidebar"] * { color: var(--text) !important; }

  /* Cards */
  .ids-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
  }

  /* Metric badge */
  .metric-badge {
    background: #1f2937;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.9rem 1rem;
    text-align: center;
  }
  .metric-badge .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--accent);
  }
  .metric-badge .label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
  }

  /* Alert chip */
  .alert-chip {
    display: inline-block;
    border-radius: 6px;
    padding: 0.35rem 0.9rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.04em;
  }

  /* Live feed row */
  .feed-row {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    border-bottom: 1px solid var(--border);
    padding: 0.45rem 0;
    display: flex;
    gap: 1rem;
    align-items: center;
  }

  /* Section title */
  .section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text);
    border-left: 3px solid var(--accent);
    padding-left: 0.7rem;
    margin-bottom: 1rem;
  }

  /* Streamlit overrides */
  .stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.4rem !important;
  }
  .stButton > button:hover { opacity: 0.85 !important; }

  div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--accent) !important;
  }
  div[data-testid="stMetricLabel"] { color: var(--muted) !important; }

  .stSelectbox > div > div { background: var(--surface) !important; border-color: var(--border) !important; }
  .stDataFrame { border: 1px solid var(--border); border-radius: 8px; }
  footer { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────

def _init_state():
    defaults = {
        "trained": False,
        "results": {},
        "artifacts": None,
        "models_cache": {},
        "df": None,
        "sim_log": [],
        "sim_running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _cached_train(df_hash, scaler_method, test_size):
    """Cache training result to avoid re-training on re-runs."""
    df_copy = st.session_state.df.copy()
    preprocessed = full_preprocess_pipeline(df_copy, scaler_method=scaler_method,
                                            test_size=test_size)
    results = train_all_models(preprocessed, verbose=False)
    artifacts = {
        "label_encoder":    preprocessed["label_encoder"],
        "feature_encoders": preprocessed["feature_encoders"],
        "scaler":           preprocessed["scaler"],
        "feature_cols":     preprocessed["feature_cols"],
        "classes":          list(preprocessed["label_encoder"].classes_),
        "results":          results,
    }
    save_artifacts(artifacts)
    return results, artifacts


def _load_trained_models(artifact_models: dict, model_names: list) -> dict:
    """Load trained sklearn models from disk."""
    loaded = {}
    for name in model_names:
        try:
            loaded[name] = load_model(name)
        except Exception:
            pass  # model not saved yet
    return loaded


def _severity_badge(category: str) -> str:
    color = get_severity_color(category)
    bg = color + "22"
    return (f'<span class="alert-chip" '
            f'style="background:{bg};color:{color};border:1px solid {color}44;">'
            f'{category}</span>')


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛡️ AI-IDS Control Panel")
    st.markdown("---")

    # ── Dataset Source ──────────────────────────
    st.markdown("### 📂 Dataset")
    data_source = st.radio("Source", ["Synthetic (Demo)", "Upload NSL-KDD"],
                           index=0, label_visibility="collapsed")

    if data_source == "Upload NSL-KDD":
        uploaded = st.file_uploader("Upload .csv / .txt", type=["csv", "txt"])
        if uploaded:
            from preprocess import load_nslkdd
            import tempfile, io
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                st.session_state.df = load_nslkdd(tmp_path)
                st.success(f"Loaded {len(st.session_state.df):,} rows")
            except Exception as e:
                st.error(f"Load error: {e}")
    else:
        n_samples = st.slider("Synthetic samples", 1000, 20000, 5000, step=500)
        if st.button("Generate Dataset"):
            with st.spinner("Generating..."):
                st.session_state.df = generate_synthetic_nslkdd(n_samples)
                st.session_state.trained = False
            st.success(f"✓ {n_samples:,} samples generated")

    st.markdown("---")

    # ── Training Config ─────────────────────────
    st.markdown("### ⚙️ Training Config")
    scaler_method = st.selectbox("Scaler", ["standard", "minmax"])
    test_size     = st.slider("Test split", 0.1, 0.4, 0.2, step=0.05)

    if st.session_state.df is not None:
        if st.button("🚀 Train Models"):
            with st.spinner("Training models…"):
                try:
                    df_hash = hash(str(st.session_state.df.shape) + scaler_method + str(test_size))
                    results, artifacts = _cached_train(df_hash, scaler_method, test_size)
                    st.session_state.results   = results
                    st.session_state.artifacts = artifacts
                    st.session_state.trained   = True
                    st.session_state.models_cache = _load_trained_models(
                        artifacts, list(results.keys()))
                except Exception as e:
                    st.error(f"Training error: {e}")
            if st.session_state.trained:
                st.success("✓ Training complete!")
    else:
        st.info("Generate or upload a dataset first.")

    st.markdown("---")

    # ── Model Selector ──────────────────────────
    st.markdown("### 🤖 Active Model")
    available_models = list(st.session_state.results.keys()) if st.session_state.results else ["(none)"]
    active_model = st.selectbox("Select model", available_models)

    st.markdown("---")
    st.markdown('<p style="color:#8b949e;font-size:0.75rem;text-align:center;">AI-IDS v1.0 · NSL-KDD</p>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

col_h1, col_h2 = st.columns([6, 1])
with col_h1:
    st.markdown("""
    <h1 style="font-family:'Sora',sans-serif;font-size:2rem;font-weight:700;
               color:#e6edf3;margin-bottom:0.2rem;">
      🛡️ AI-Based Intrusion Detection System
    </h1>
    <p style="color:#8b949e;font-size:0.9rem;margin-top:0;">
      Real-time network traffic classification · NSL-KDD Dataset · ML-Powered
    </p>
    """, unsafe_allow_html=True)

with col_h2:
    st.markdown(f"""
    <div style="text-align:right;color:#8b949e;font-size:0.75rem;
                font-family:'JetBrains Mono',monospace;padding-top:1rem;">
      {format_timestamp()}
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────────

tab_overview, tab_metrics, tab_cm, tab_features, tab_realtime = st.tabs([
    "📊 Overview",
    "📈 Model Metrics",
    "🔲 Confusion Matrix",
    "🔍 Feature Importance",
    "⚡ Live Detection",
])

# ════════════════════════════════════════════
# TAB 1: OVERVIEW
# ════════════════════════════════════════════

with tab_overview:
    if st.session_state.df is None:
        st.info("👈 Generate or upload a dataset using the sidebar to get started.")
    else:
        df = st.session_state.df

        # ── KPI Row ──────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        attack_col = CATEGORY_COLUMN if CATEGORY_COLUMN in df.columns else "label"
        n_attacks  = (df[attack_col] != "Normal").sum() if attack_col in df.columns else 0
        pct_attack = n_attacks / len(df) * 100 if len(df) > 0 else 0

        with col1:
            st.markdown(f"""<div class="metric-badge">
              <div class="value">{len(df):,}</div>
              <div class="label">Total Packets</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-badge">
              <div class="value">{df.shape[1]}</div>
              <div class="label">Features</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-badge">
              <div class="value" style="color:#f85149;">{n_attacks:,}</div>
              <div class="label">Attack Packets</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-badge">
              <div class="value" style="color:#d29922;">{pct_attack:.1f}%</div>
              <div class="label">Attack Rate</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Distribution Chart ───────────────────
        col_chart, col_preview = st.columns([3, 2])
        with col_chart:
            st.markdown('<div class="section-title">Attack Category Distribution</div>',
                        unsafe_allow_html=True)
            fig = plot_attack_distribution(df)
            st.pyplot(fig, use_container_width=True)

        with col_preview:
            st.markdown('<div class="section-title">Dataset Preview</div>',
                        unsafe_allow_html=True)
            preview_cols = ["protocol_type", "service", "src_bytes",
                            "dst_bytes", "serror_rate", attack_col]
            preview_cols = [c for c in preview_cols if c in df.columns]
            st.dataframe(df[preview_cols].head(20), height=350, use_container_width=True)

        # ── Category breakdown ───────────────────
        if attack_col in df.columns:
            st.markdown('<div class="section-title">Category Breakdown</div>',
                        unsafe_allow_html=True)
            cats = df[attack_col].value_counts().reset_index()
            cats.columns = ["Category", "Count"]
            cats["Percentage"] = (cats["Count"] / len(df) * 100).round(2).astype(str) + "%"
            st.dataframe(cats, hide_index=True, use_container_width=True)

# ════════════════════════════════════════════
# TAB 2: MODEL METRICS
# ════════════════════════════════════════════

with tab_metrics:
    if not st.session_state.trained:
        st.info("Train models first using the sidebar.")
    else:
        results = st.session_state.results

        st.markdown('<div class="section-title">Performance Summary</div>',
                    unsafe_allow_html=True)
        table = build_metrics_table(results)
        st.dataframe(table, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-title">Accuracy Comparison</div>',
                        unsafe_allow_html=True)
            fig = plot_accuracy_comparison(results)
            st.pyplot(fig, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-title">All Metrics Comparison</div>',
                        unsafe_allow_html=True)
            fig2 = plot_metrics_radar(results)
            st.pyplot(fig2, use_container_width=True)

        # ── Per-model detail ─────────────────────
        st.markdown('<div class="section-title">Detailed Model Report</div>',
                    unsafe_allow_html=True)
        selected = st.selectbox("Model", list(results.keys()), key="metrics_detail")
        m = results[selected]

        c1, c2, c3, c4 = st.columns(4)
        badges = [
            (c1, "Accuracy",  m["accuracy"]),
            (c2, "Precision", m["precision"]),
            (c3, "Recall",    m["recall"]),
            (c4, "F1-Score",  m["f1_score"]),
        ]
        for col, label, val in badges:
            with col:
                st.metric(label, f"{val*100:.2f}%")

        if "train_time_s" in m:
            st.caption(f"⏱ Training time: {m['train_time_s']}s")

        with st.expander("Classification Report"):
            st.code(m["report"], language="text")

# ════════════════════════════════════════════
# TAB 3: CONFUSION MATRIX
# ════════════════════════════════════════════

with tab_cm:
    if not st.session_state.trained:
        st.info("Train models first using the sidebar.")
    else:
        results  = st.session_state.results
        classes  = st.session_state.artifacts["classes"]

        model_sel = st.selectbox("Select model", list(results.keys()), key="cm_model")
        cm_data   = results[model_sel]["confusion_matrix"]

        st.markdown(f'<div class="section-title">Confusion Matrix — {model_sel}</div>',
                    unsafe_allow_html=True)
        fig = plot_confusion_matrix(cm_data, classes, title=f"Confusion Matrix — {model_sel}")
        st.pyplot(fig, use_container_width=False)

        # Normalised version
        with st.expander("Show normalized confusion matrix"):
            import numpy as np
            cm_arr = np.array(cm_data, dtype=float)
            row_sums = cm_arr.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm_arr, row_sums, where=row_sums != 0)
            fig_n = plot_confusion_matrix(
                cm_norm.round(2).tolist(), classes,
                title=f"Normalized Confusion Matrix — {model_sel}"
            )
            st.pyplot(fig_n, use_container_width=False)

# ════════════════════════════════════════════
# TAB 4: FEATURE IMPORTANCE
# ════════════════════════════════════════════

with tab_features:
    if not st.session_state.trained:
        st.info("Train models first using the sidebar.")
    else:
        artifacts = st.session_state.artifacts
        feature_cols = artifacts["feature_cols"]
        models_cache = st.session_state.models_cache

        model_sel = st.selectbox("Select model", list(models_cache.keys()), key="fi_model")
        top_n = st.slider("Top N features", 5, 30, 15)

        if model_sel in models_cache:
            model = models_cache[model_sel]
            fig   = plot_feature_importance(model, feature_cols, model_sel, top_n=top_n)
            if fig:
                st.markdown(f'<div class="section-title">Feature Importance — {model_sel}</div>',
                            unsafe_allow_html=True)
                st.pyplot(fig, use_container_width=True)
            else:
                st.warning(f"{model_sel} does not expose feature importances.")

# ════════════════════════════════════════════
# TAB 5: LIVE DETECTION
# ════════════════════════════════════════════

with tab_realtime:
    st.markdown('<div class="section-title">⚡ Real-Time Attack Detection Simulation</div>',
                unsafe_allow_html=True)

    if not st.session_state.trained:
        st.info("Train models first, then come back here for live simulation.")
    else:
        artifacts    = st.session_state.artifacts
        models_cache = st.session_state.models_cache

        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            sim_model_name = st.selectbox("Detection model", list(models_cache.keys()))
        with col_cfg2:
            sim_mode = st.selectbox("Traffic mode",
                                    ["Random Mix", "Normal only", "DoS only",
                                     "Probe only", "R2L only", "U2R only"])
        with col_cfg3:
            n_packets = st.slider("Packets per burst", 1, 20, 5)

        category_map = {
            "Random Mix": None, "Normal only": "Normal", "DoS only": "DoS",
            "Probe only": "Probe", "R2L only": "R2L", "U2R only": "U2R",
        }
        sim_category = category_map[sim_mode]

        # ── Live Feed Placeholder ────────────────
        feed_placeholder     = st.empty()
        counters_placeholder = st.empty()
        chart_placeholder    = st.empty()

        # Initialise log
        if "sim_log" not in st.session_state or not st.session_state.sim_log:
            st.session_state.sim_log = []

        col_start, col_stop, col_clear = st.columns([1, 1, 2])
        with col_start:
            start = st.button("▶ Start Simulation")
        with col_stop:
            stop = st.button("⏹ Stop")
        with col_clear:
            if st.button("🗑 Clear Log"):
                st.session_state.sim_log = []

        if stop:
            st.session_state.sim_running = False

        if start:
            st.session_state.sim_running = True

        if st.session_state.sim_running:
            model      = models_cache[sim_model_name]
            le         = artifacts["label_encoder"]
            feat_cols  = artifacts["feature_cols"]
            scaler     = artifacts.get("scaler")

            BURST = 8   # number of simulation rounds
            for _ in range(BURST):
                if not st.session_state.sim_running:
                    break

                packets = simulate_traffic(
                    n_packets=n_packets,
                    category=sim_category,
                    feature_cols=feat_cols,
                )
                preds = predict_traffic(model, packets.copy(), le, feat_cols, scaler)

                for _, row in preds.iterrows():
                    cat   = row.get("prediction", "Unknown")
                    conf  = row.get("confidence", 100.0)
                    true_ = row.get("true_category", "?")
                    ts    = format_timestamp()
                    st.session_state.sim_log.append({
                        "timestamp": ts,
                        "prediction": cat,
                        "true": true_,
                        "confidence": conf,
                        "correct": cat == true_,
                    })

                # ── Render log (last 40 entries) ──
                log_df = pd.DataFrame(st.session_state.sim_log[-40:])

                with feed_placeholder.container():
                    st.markdown("**Live Packet Feed** (latest 40)")
                    st.dataframe(
                        log_df[["timestamp", "prediction", "true", "confidence", "correct"]
                               ].rename(columns={
                                   "timestamp":  "Timestamp",
                                   "prediction": "Prediction",
                                   "true":       "True Label",
                                   "confidence": "Confidence (%)",
                                   "correct":    "Correct",
                               }).style.map(
                                   lambda v: f"color: {'#3fb950' if v else '#f85149'}",
                                   subset=["Correct"]
                               ),
                        use_container_width=True,
                        height=300,
                    )

                # ── Counters ──────────────────────
                all_log = pd.DataFrame(st.session_state.sim_log)
                with counters_placeholder.container():
                    cats = ["Normal", "DoS", "Probe", "R2L", "U2R"]
                    cols = st.columns(5)
                    for col, cat in zip(cols, cats):
                        cnt   = (all_log["prediction"] == cat).sum() if len(all_log) > 0 else 0
                        color = get_severity_color(cat)
                        with col:
                            st.markdown(f"""
                            <div class="metric-badge">
                              <div class="value" style="color:{color};">{cnt}</div>
                              <div class="label">{cat}</div>
                            </div>""", unsafe_allow_html=True)

                # ── Trend chart ───────────────────
                if len(all_log) >= 5:
                    recent = all_log.tail(50)
                    counts = recent["prediction"].value_counts()
                    with chart_placeholder.container():
                        import matplotlib.pyplot as plt
                        PALETTE_BG = "#0d1117"
                        PALETTE_SF = "#161b22"
                        PALETTE_TX = "#e6edf3"
                        PALETTE_BD = "#30363d"
                        CAT_COLORS = {
                            "Normal": "#3fb950", "DoS": "#f85149",
                            "Probe": "#d29922", "R2L": "#bc8cff", "U2R": "#ff7b72",
                        }
                        fig, ax = plt.subplots(figsize=(10, 2.5))
                        fig.patch.set_facecolor(PALETTE_BG)
                        ax.set_facecolor(PALETTE_SF)
                        bar_colors = [CAT_COLORS.get(c, "#58a6ff") for c in counts.index]
                        ax.bar(counts.index, counts.values, color=bar_colors,
                               edgecolor=PALETTE_BD, linewidth=0.6)
                        ax.set_ylabel("Count", color=PALETTE_TX, fontsize=9)
                        ax.set_title("Prediction Distribution (last 50 packets)",
                                     color=PALETTE_TX, fontsize=9)
                        ax.tick_params(colors=PALETTE_TX)
                        for sp in ax.spines.values():
                            sp.set_edgecolor(PALETTE_BD)
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                time.sleep(0.4)

            st.session_state.sim_running = False

        # ── Accuracy on log ──────────────────────
        if st.session_state.sim_log:
            log_df = pd.DataFrame(st.session_state.sim_log)
            acc = log_df["correct"].mean() * 100 if "correct" in log_df else 0
            st.markdown(f"**Simulation Accuracy:** `{acc:.1f}%` over "
                        f"`{len(log_df)}` simulated packets.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<p style="text-align:center;color:#8b949e;font-size:0.75rem;">
  AI-Based Intrusion Detection System · Built with Python, Scikit-learn, XGBoost & Streamlit
  · NSL-KDD Dataset · 🛡️
</p>
""", unsafe_allow_html=True)