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

# Minimal CSS for button styling
st.markdown("""
<style>
    /* Style submit button to be blue */
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #3B82F6 !important;
        color: white !important;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #2563EB !important;
    }
</style>
""", unsafe_allow_html=True)


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
    """Detect if the current theme is dark mode."""
    try:
        # Try to get theme from session state or config
        theme = st.get_option("theme.base")
        if theme == "dark":
            return True
        elif theme == "light":
            return False
    except:
        pass
    # Default to light theme
    return False


def create_bar_chart(metrics: dict):
    """Create horizontal bar chart for model metrics."""
    labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    values = [
        float(metrics["accuracy"]),
        float(metrics["precision"]),
        float(metrics["recall"]),
        float(metrics["f1_score"]),
        float(metrics["roc_auc"]),
    ]

    dark = _theme_is_dark()
    font_color = "white" if dark else "black"
    grid_color = "rgba(255,255,255,0.1)" if dark else "rgba(0,0,0,0.1)"
    bar_color = "#3B82F6"  # Blue color

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker=dict(color=bar_color),
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
            textfont=dict(color=font_color),
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=70, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=font_color),
        xaxis=dict(
            range=[0, 1.05],
            tickformat=".0%",
            gridcolor=grid_color,
            zeroline=False,
            tickfont=dict(color=font_color)
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color=font_color)
        ),
    )
    return fig


def create_gauge(probability: float):
    """Create gauge chart for probability visualization."""
    dark = _theme_is_dark()
    font_color = "white" if dark else "black"
    tick_color = "rgba(255,255,255,0.6)" if dark else "rgba(0,0,0,0.6)"
    bar_color = "#3B82F6"  # Blue color

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 52, "color": font_color}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": tick_color,
                    "tickfont": {"color": font_color}
                },
                "bar": {"color": bar_color, "thickness": 0.75},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(16,185,129,0.2)"},
                    {"range": [30, 60], "color": "rgba(245,158,11,0.2)"},
                    {"range": [60, 100], "color": "rgba(239,68,68,0.2)"},
                ],
            },
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=font_color),
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
            st.markdown(f"**Name:** {metrics.get('model_name','-')}")
            st.markdown(f"**Version:** {metrics.get('model_version','-')}")
            st.markdown(f"**Algorithm:** Gradient Boosting")
            st.markdown(f"**Run ID:** `{str(metrics.get('run_id',''))[:16]}...`")

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