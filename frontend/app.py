import os
import requests
import streamlit as st
import plotly.graph_objects as go

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Diabetes Prediction",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<script>
    const root = document.documentElement;
    root.style.setProperty('--primary-color', '#3B82F6');
</script>
""", unsafe_allow_html=True)

# --- Flat style + theme-aware (dark/light) ---
st.markdown(
    """
<style>
  /* Uses Streamlit theme variables (dark/light) */
  :root {
    --bg: var(--background-color);
    --bg2: var(--secondary-background-color);
    --text: var(--text-color);
    --primary: var(--primary-color);
    --radius: 14px;
  }
  
  [data-testid="stHeader"] { background: transparent; }

  .block-container { padding-top: 1.75rem; padding-bottom: 2rem; max-width: 1400px; }

  .stTabs [data-baseweb="tab-list"] { gap: 1.25rem; padding-left: 0.25rem; }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: 0 !important;
    padding: 0.4rem 0.25rem !important;
  }
  .stTabs [aria-selected="true"] {
    border-bottom: 2px solid #3B82F6 !important;
    color: #3B82F6 !important;
  }

  div[data-testid="stMetric"] {
    background: transparent !important;
    border: 0 !important;
    padding: 0 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
  }
  div[data-testid="stMetricValue"] { font-weight: 750; }
  div[data-testid="stMetricLabel"] {
    opacity: 0.75;
    letter-spacing: .06em;
    text-transform: uppercase;
    font-size: 0.8rem;
  }

  .stPlotlyChart { background: transparent !important; border: 0 !important; padding: 0 !important; }

  .stButton > button, div[data-testid="stFormSubmitButton"] > button {
    border-radius: var(--radius) !important;
    padding: 0.9rem 1rem !important;
    font-weight: 700 !important;
    background-color: #3B82F6 !important;
    color: white !important;
  }
  .stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #2563EB !important;
  }

  /* Sliders - blue */
  .stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: #3B82F6 !important;
  }
  .stSlider [data-baseweb="slider"] [data-testid="stTickBar"] > div {
    background-color: #3B82F6 !important;
  }

  /* Progress bar - blue */
  .stProgress > div > div > div {
    background-color: #3B82F6 !important;
  }

  /* Links - blue */
  a, a:link, a:visited {
    color: #3B82F6 !important;
  }
  a:hover {
    color: #2563EB !important;
  }

  .pill {
    display: inline-flex;
    gap: .5rem;
    align-items: center;
    padding: .35rem .65rem;
    border-radius: 999px;
    background: var(--bg2);
    font-weight: 650;
  }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60, show_spinner=False)
def get_metrics():
    try:
        r = requests.get(f"{BACKEND_URL}/metrics", timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def make_prediction(payload: dict):
    try:
        r = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"Backend call error: {e}")
        return None


def _theme_is_dark() -> bool:
    base = st.get_option("theme.base")  # "light" | "dark" | "auto" | None
    if base == "dark":
        return True
    if base == "light":
        return False
    # fallback: si auto/None -> on reste neutre (Plotly transparent)
    return False


def _primary_color() -> str:
    return st.get_option("theme.primaryColor") or "#3b82f6"


def create_bar_chart(metrics: dict):
    labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    values = [
        float(metrics["accuracy"]),
        float(metrics["precision"]),
        float(metrics["recall"]),
        float(metrics["f1_score"]),
        float(metrics["roc_auc"]),
    ]

    primary = _primary_color()
    dark = _theme_is_dark()
    font_color = "rgba(255,255,255,.85)" if dark else "rgba(0,0,0,.75)"
    grid = "rgba(148,163,184,.25)" if dark else "rgba(15,23,42,.12)"

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker=dict(color=primary, cornerradius=8),
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=70, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui", color=font_color),
        xaxis=dict(range=[0, 1.05], tickformat=".0%", gridcolor=grid, zeroline=False),
        yaxis=dict(showgrid=False),
    )
    return fig


def create_gauge(probability: float):
    primary = _primary_color()
    dark = _theme_is_dark()
    font_color = "rgba(255,255,255,.9)" if dark else "rgba(0,0,0,.8)"
    tick = "rgba(148,163,184,.6)" if dark else "rgba(15,23,42,.35)"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 52, "color": font_color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": tick},
                "bar": {"color": primary, "thickness": 0.75},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(16,185,129,.18)"},
                    {"range": [30, 60], "color": "rgba(245,158,11,.18)"},
                    {"range": [60, 100], "color": "rgba(239,68,68,.18)"},
                ],
            },
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui"),
    )
    return fig


# --- Header ---
st.title("Diabetes Prediction")
st.caption("Medical decision support system based on machine learning")
st.divider()


tab_metrics, tab_pred = st.tabs(
    [":material/insights: Model Metrics", ":material/monitor_heart: Patient Assessment"]
)

# --- Tab 1: metrics ---
with tab_metrics:
    metrics = get_metrics()
    if not metrics:
        st.error("Unable to retrieve metrics from backend.")
        st.code("Check BACKEND_URL and GET /metrics endpoint.", language="bash")
    else:
        st.subheader("Performance")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        c2.metric("Precision", f"{metrics['precision']:.1%}")
        c3.metric("Recall", f"{metrics['recall']:.1%}")
        c4.metric("F1 Score", f"{metrics['f1_score']:.1%}")
        c5.metric("ROC AUC", f"{metrics['roc_auc']:.1%}")

        st.divider()

        left, right = st.columns([2, 1], vertical_alignment="top")
        with left:
            st.markdown("#### Metrics Comparison")
            st.plotly_chart(create_bar_chart(metrics), use_container_width=True)

        with right:
            st.markdown("#### Model Information")
            st.markdown(
                f"""
<span class="pill">:material/label: Name</span> **{metrics.get("model_name","-")}**
<span class="pill">:material/tag: Version</span> **{metrics.get("model_version","-")}**
<span class="pill">:material/tune: Algorithm</span> **Gradient Boosting**
<span class="pill">:material/fingerprint: Run</span> `{str(metrics.get("run_id",""))[:16]}…`
""",
                unsafe_allow_html=True,
            )

# --- Tab 2: prediction ---
with tab_pred:
    st.subheader("Clinical Data")
    st.caption("Adjust the values to estimate diabetes risk.")

    with st.form("prediction_form", border=False):
        colA, colB = st.columns(2)

        payload = {}

        with colA:
            st.markdown("#### :material/science: Metabolic Indicators")
            payload["Glucose"] = st.slider(
                "Plasma Glucose (mg/dL) :material/bloodtype:",
                50, 250, 120,
                help="Glucose concentration (e.g., oral test).",
            )
            payload["BMI"] = st.slider(
                "Body Mass Index (kg/m²) :material/monitor_weight:",
                15.0, 60.0, 25.0, 0.5,
                help="Weight / height².",
            )
            payload["DiabetesPedigreeFunction"] = st.slider(
                "Diabetes Pedigree Function :material/genetics:",
                0.05, 2.5, 0.50, 0.05,
                help="Genetic predisposition indicator.",
            )

        with colB:
            st.markdown("#### :material/favorite: Physiological Indicators")
            payload["BloodPressure"] = st.slider(
                "Diastolic Blood Pressure (mm Hg) :material/heartbeat:",
                40, 150, 70,
                help="Resting measurement.",
            )
            payload["Age"] = st.slider(
                "Age (years) :material/person:",
                18, 100, 35,
                help="Patient age.",
            )

        submitted = st.form_submit_button(
            ":material/analytics: Assess Risk",
            use_container_width=True,
        )

    if submitted:
        with st.spinner("Analyzing..."):
            result = make_prediction(payload)

        if not result:
            st.error("No valid response from backend (POST /predict).")
        else:
            prob = float(result["probability"])
            risk = str(result.get("risk_level", "")).lower()

            if risk == "low":
                titre = "Low Risk"
                reco = [
                    "Maintain regular physical activity.",
                    "Monitor diet and weight.",
                    "Periodic checkups according to clinical context.",
                ]
            elif risk == "moderate":
                titre = "Moderate Risk"
                reco = [
                    "Medical advice recommended if associated risk factors present.",
                    "Optimize lifestyle (activity, diet, sleep).",
                    "Consider additional tests according to protocol.",
                ]
            else:
                titre = "High Risk"
                reco = [
                    "Consult a healthcare professional for confirmation.",
                    "Perform additional biological tests.",
                    "Appropriate care according to clinical evaluation.",
                ]

            st.divider()
            st.subheader("Result")

            gcol, tcol = st.columns([1, 1], vertical_alignment="top")
            with gcol:
                st.plotly_chart(create_gauge(prob), use_container_width=True)

            with tcol:
                st.markdown(f"### {titre}")
                st.metric("Estimated Probability", f"{prob:.1%}")
                st.progress(prob)

                st.markdown("#### Recommendations")
                for r in reco:
                    st.write(f"- {r}")

                st.caption(f"Model: version {result.get('model_version','-')}")

st.divider()
st.caption("Ivan Gallo Pena — Educational Project | Technical Specialization in Artificial Intelligence")