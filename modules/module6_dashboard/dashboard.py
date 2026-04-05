"""
Module 6: Cyber Security Real-Time Dashboard
----------------------------------------------
Streamlit dashboard that:
  - Loads the trained sklearn RandomForest model (from Module 5) at startup.
  - Reads live_logs.csv (grown by Module 1) every 2 seconds.
  - Adds ML predictions + confidence % to every row.
  - Shows 6 Plotly charts (dark theme) and key metrics.

Run Module 1 in a separate terminal first, then:
    streamlit run modules/module6_dashboard/dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import time

# ── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_FILE     = os.path.join(PROJECT_ROOT, "logs",   "live_logs.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

CATEGORICAL_COLS = ["Protocol", "Packet Type", "Traffic Type", "Severity Level"]
NUMERIC_COLS     = ["Source Port", "Destination Port", "Packet Length", "Anomaly Scores"]
FEATURE_COLS     = NUMERIC_COLS + [c + "_enc" for c in CATEGORICAL_COLS]

PLOTLY_THEME = "plotly_dark"


# ── Model loading (cached – runs once per session) ────────────────
@st.cache_resource
def load_models():
    model_path   = os.path.join(MODELS_DIR, "rf_attack_model.joblib")
    label_path   = os.path.join(MODELS_DIR, "label_encoder.joblib")
    feature_path = os.path.join(MODELS_DIR, "feature_encoders.joblib")

    if not all(os.path.exists(p) for p in [model_path, label_path, feature_path]):
        return None, None, None

    rf        = joblib.load(model_path)
    label_enc = joblib.load(label_path)
    feat_enc  = joblib.load(feature_path)
    return rf, label_enc, feat_enc


def load_data() -> pd.DataFrame:
    if os.path.exists(LOG_FILE):
        try:
            return pd.read_csv(LOG_FILE)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def add_predictions(df: pd.DataFrame, rf, label_enc, feat_enc) -> pd.DataFrame:
    """Run inference and append Predicted_Attack + Confidence_Pct columns."""
    if rf is None or df.empty:
        return df

    try:
        dfc = df.copy()

        for col in NUMERIC_COLS:
            dfc[col] = pd.to_numeric(dfc[col], errors="coerce").fillna(0)

        for col in CATEGORICAL_COLS:
            if col in feat_enc:
                le = feat_enc[col]
                dfc[col] = dfc[col].fillna("Unknown").astype(str)
                # Handle unseen labels gracefully
                known = set(le.classes_)
                dfc[col] = dfc[col].apply(lambda v: v if v in known else le.classes_[0])
                dfc[col + "_enc"] = le.transform(dfc[col])
            else:
                dfc[col + "_enc"] = 0

        X = dfc[FEATURE_COLS]
        proba = rf.predict_proba(X)
        preds = rf.predict(X)

        dfc["Predicted_Attack"] = label_enc.inverse_transform(preds)
        dfc["Confidence_Pct"]   = (proba.max(axis=1) * 100).round(1)
        return dfc

    except Exception as e:
        df["Predicted_Attack"] = "Error"
        df["Confidence_Pct"]   = 0.0
        return df


# ── Dashboard layout ──────────────────────────────────────────────
def run_dashboard():
    st.set_page_config(
        page_title="CyberSecurity Analytics Dashboard",
        page_icon="🛡️",
        layout="wide",
    )

    # Custom CSS for dark premium feel
    st.markdown(
        """
        <style>
        body { background-color: #0e1117; }
        .metric-card { background: #1e2130; border-radius: 10px; padding: 12px; }
        .block-container { padding-top: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🛡️ CyberSecurity Attack Detection — Live Dashboard")
    st.caption("Auto-refreshes every 2 seconds. Run Module 1 to start the log stream.")

    rf, label_enc, feat_enc = load_models()

    model_loaded = rf is not None
    if model_loaded:
        st.success("✅ ML model loaded — live predictions active", icon="🤖")
    else:
        st.warning(
            "⚠️ Model not found. Run `py -3 modules/module5_ml/ml_attack_detection.py` first. "
            "Charts will still show without predictions.",
            icon="⚠️",
        )

    # Placeholders
    metrics_ph = st.empty()
    charts1_ph = st.empty()
    charts2_ph = st.empty()
    charts3_ph = st.empty()
    table_ph   = st.empty()

    while True:
        raw_df = load_data()

        if raw_df.empty:
            st.info("⏳ Waiting for log data… run Module 1 first.")
        else:
            df = add_predictions(raw_df, rf, label_enc, feat_enc)

            total_logs      = len(df)
            attack_types    = df["Attack Type"].nunique() if "Attack Type" in df.columns else 0
            high_severity   = int((df["Severity Level"] == "High").sum()) if "Severity Level" in df.columns else 0
            avg_anomaly     = round(df["Anomaly Scores"].mean(), 3) if "Anomaly Scores" in df.columns else 0

            # ── Metrics row ───────────────────────────────────────
            with metrics_ph.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("📋 Total Logs",          f"{total_logs:,}")
                c2.metric("🔣 Unique Attack Types",  attack_types)
                c3.metric("🚨 High Severity",        f"{high_severity:,}")
                c4.metric("📈 Avg Anomaly Score",    avg_anomaly)

            # ── Row 1: Attack distribution + Protocol ─────────────
            with charts1_ph.container():
                col1, col2 = st.columns(2)

                if "Attack Type" in df.columns:
                    fig_atk = px.pie(
                        df, names="Attack Type",
                        title="Attack Type Distribution",
                        template=PLOTLY_THEME,
                        hole=0.35,
                    )
                    col1.plotly_chart(fig_atk, use_container_width=True)

                if "Protocol" in df.columns:
                    proto_counts = df["Protocol"].value_counts().reset_index()
                    proto_counts.columns = ["Protocol", "count"]
                    fig_proto = px.bar(
                        proto_counts, x="Protocol", y="count",
                        title="Protocol Distribution",
                        template=PLOTLY_THEME,
                        color="count",
                        color_continuous_scale="Blues",
                    )
                    col2.plotly_chart(fig_proto, use_container_width=True)

            # ── Row 2: Severity bar + Traffic Type pie ────────────
            with charts2_ph.container():
                col1, col2 = st.columns(2)

                if "Severity Level" in df.columns:
                    sev_counts = df["Severity Level"].value_counts().reset_index()
                    sev_counts.columns = ["Severity Level", "count"]
                    color_map  = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}
                    fig_sev = px.bar(
                        sev_counts, x="Severity Level", y="count",
                        title="Severity Level Breakdown",
                        template=PLOTLY_THEME,
                        color="Severity Level",
                        color_discrete_map=color_map,
                    )
                    col1.plotly_chart(fig_sev, use_container_width=True)

                if "Traffic Type" in df.columns:
                    fig_traffic = px.pie(
                        df, names="Traffic Type",
                        title="Traffic Type Distribution",
                        template=PLOTLY_THEME,
                        hole=0.35,
                    )
                    col2.plotly_chart(fig_traffic, use_container_width=True)

            # ── Row 3: Anomaly Score histogram + Top Source IPs ───
            with charts3_ph.container():
                col1, col2 = st.columns(2)

                if "Anomaly Scores" in df.columns:
                    fig_hist = px.histogram(
                        df, x="Anomaly Scores",
                        nbins=40,
                        title="Anomaly Score Distribution",
                        template=PLOTLY_THEME,
                        color_discrete_sequence=["#6366f1"],
                    )
                    col1.plotly_chart(fig_hist, use_container_width=True)

                if "Source IP Address" in df.columns:
                    top_ips = (
                        df["Source IP Address"]
                        .value_counts()
                        .head(10)
                        .reset_index()
                    )
                    top_ips.columns = ["Source IP", "event_count"]
                    fig_ips = px.bar(
                        top_ips, x="event_count", y="Source IP",
                        orientation="h",
                        title="Top 10 Source IPs by Event Count",
                        template=PLOTLY_THEME,
                        color="event_count",
                        color_continuous_scale="Reds",
                    )
                    fig_ips.update_layout(yaxis=dict(autorange="reversed"))
                    col2.plotly_chart(fig_ips, use_container_width=True)

            # ── Recent Logs Table (with predictions if available) ─
            with table_ph.container():
                st.subheader("🗒️ Recent Activity Logs (last 15 rows)")
                display_cols = [
                    "Emit_Timestamp", "Source IP Address", "Attack Type",
                    "Severity Level", "Protocol", "Anomaly Scores",
                ]
                if model_loaded and "Predicted_Attack" in df.columns:
                    display_cols += ["Predicted_Attack", "Confidence_Pct"]

                available = [c for c in display_cols if c in df.columns]
                recent    = df.tail(15)[available]

                # Color-highlight Predicted_Attack column
                if "Predicted_Attack" in recent.columns:
                    st.dataframe(recent, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(recent, use_container_width=True, hide_index=True)

        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    run_dashboard()
