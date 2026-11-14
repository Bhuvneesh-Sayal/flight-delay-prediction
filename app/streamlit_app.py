"""
app/streamlit_app.py
====================
Interactive Streamlit UI for real-time flight delay inference.

Loads the saved two-stage hurdle model (departure task, bts_weather condition)
and lets a user configure a flight, then returns:
  - P(delayed > 0)        — from the hurdle classifier
  - Expected delay (min)  — combined hurdle prediction

Run with:
    streamlit run app/streamlit_app.py

Requirements:
    pip install streamlit
    The hurdle model must have been saved by running:
        python main.py --steps hurdle
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make src/ and config/ importable when running from the app/ directory
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import (
    AIRPORT_RUNWAYS,
    PATHS,
    TOP_10_AIRPORTS,
    WEATHER_FEATURE_COLS,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Model loading  (cached so it only loads once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading hurdle model…")
def load_models():
    """Load the departure hurdle classifier and regressor PipelineModels."""
    try:
        from pyspark.ml import PipelineModel
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder
            .appName("StreamlitInference")
            .master("local[2]")
            .config("spark.driver.memory", "2g")
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")

        clf_path = PATHS["models"] / "departure_hurdle_classifier"
        reg_path = PATHS["models"] / "departure_hurdle_regressor"

        if not clf_path.exists() or not reg_path.exists():
            return None, None, None

        clf_model = PipelineModel.load(str(clf_path))
        reg_model = PipelineModel.load(str(reg_path))
        return spark, clf_model, reg_model

    except Exception as exc:
        st.error(f"Failed to load models: {exc}")
        return None, None, None


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def _build_input_row(
    spark,
    origin: str,
    dest: str,
    airline: str,
    month: int,
    day_of_week: int,
    dep_hour: float,
    elapsed_time: float,
    congestion: int,
    prior_delay: float,
    temp: float,
    dew: float,
    wind_speed: float,
    wind_dir: float,
    visibility: float,
    slp: float,
) -> object:
    """Construct a single-row Spark DataFrame matching the weather feature schema."""
    from pyspark.sql import types as T

    rad = np.pi / 180.0
    runway_hdg = AIRPORT_RUNWAYS.get(origin, 180.0)
    wind_diff = (wind_dir - runway_hdg) * rad
    crosswind = abs(wind_speed * np.sin(wind_diff))
    headwind  = wind_speed * np.cos(wind_diff)
    icing     = 1.0 if (temp <= 0.0 and (temp - dew) <= 2.0) else 0.0
    ifr       = 1.0 if visibility < 4800.0 else 0.0

    data = [{
        "Origin":                   origin,
        "Dest":                     dest,
        "Reporting_Airline":        airline,
        "scheduled_month":          month,
        "scheduled_day_of_week":    day_of_week,
        "scheduled_departure_hour": dep_hour,
        "CRSElapsedTime":           elapsed_time,
        "origin_congestion_2h":     congestion,
        "prior_leg_delay":          prior_delay,
        "w_temp":                   temp,
        "w_dew":                    dew,
        "w_wspeed":                 wind_speed,
        "w_slp":                    slp,
        "weather_crosswind":        crosswind,
        "weather_headwind":         headwind,
        "weather_icing_risk":       icing,
        "weather_ifr_conditions":   ifr,
        "weather_available":        1,
        "weather_age_minutes":      30.0,
    }]
    return spark.createDataFrame(pd.DataFrame(data))


def _predict(spark, clf_model, reg_model, input_df) -> tuple:
    """
    Run the two-stage hurdle prediction.
    Returns (p_delayed, expected_delay_minutes).
    """
    from pyspark.sql import types as T

    get_prob1 = st.session_state.get("_get_prob1")
    if get_prob1 is None:
        from pyspark.sql.functions import udf
        get_prob1 = udf(lambda v: float(v[1]), T.DoubleType())
        st.session_state["_get_prob1"] = get_prob1

    clf_out = clf_model.transform(input_df)
    p_delayed = float(clf_out.select(get_prob1("probability")).first()[0])

    reg_out = reg_model.transform(input_df)
    log_pred = float(reg_out.select("prediction").first()[0])
    cond_delay = float(np.expm1(max(log_pred, 0.0)))

    expected_delay = p_delayed * cond_delay
    return p_delayed, expected_delay


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main():
    st.title("✈️ Flight Delay Predictor")
    st.caption(
        "Two-stage hurdle model — DA231o Data Scaling project | "
        "Trained on 2025 US domestic flights (10 hub airports)"
    )

    spark, clf_model, reg_model = load_models()

    if spark is None:
        st.warning(
            "⚠️ Models not found. Run `python main.py --steps hurdle` first to train and save them."
        )
        st.stop()

    # ── Flight details ───────────────────────────────────────────────────────
    st.subheader("Flight Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        origin = st.selectbox("Origin Airport", TOP_10_AIRPORTS, index=0)
    with col2:
        dest_opts = [a for a in TOP_10_AIRPORTS if a != origin]
        dest = st.selectbox("Destination Airport", dest_opts, index=1)
    with col3:
        airline = st.selectbox("Airline", ["AA", "DL", "UA", "WN", "B6", "AS", "NK", "F9"])

    col4, col5, col6 = st.columns(3)
    with col4:
        month = st.slider("Month", 1, 12, 6)
    with col5:
        day_of_week = st.slider("Day of Week (1=Sun, 7=Sat)", 1, 7, 2)
    with col6:
        dep_hour = st.slider("Scheduled Departure Hour", 0.0, 23.0, 8.0, step=0.5)

    col7, col8 = st.columns(2)
    with col7:
        elapsed_time = st.number_input("Scheduled Flight Duration (min)", 60, 600, 180)
    with col8:
        congestion = st.slider("Origin Airport Congestion (flights in past 2h)", 0, 50, 10)

    prior_delay = st.slider("Prior Leg Arrival Delay (min)", -30.0, 120.0, 0.0, step=1.0)

    # ── Weather conditions ───────────────────────────────────────────────────
    st.subheader("Weather at Origin")
    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        temp      = st.number_input("Temperature (°C)", -30.0, 45.0, 15.0)
        dew       = st.number_input("Dew Point (°C)",   -30.0, 30.0, 8.0)
    with wc2:
        wind_speed = st.number_input("Wind Speed (m/s)", 0.0, 30.0, 5.0)
        wind_dir   = st.number_input("Wind Direction (°)", 0.0, 360.0, 180.0)
    with wc3:
        visibility = st.number_input("Visibility (m)", 0.0, 10000.0, 9000.0)
        slp        = st.number_input("Sea Level Pressure (hPa)", 950.0, 1050.0, 1013.0)

    # ── Prediction ──────────────────────────────────────────────────────────
    st.divider()
    if st.button("🔮 Predict Delay", type="primary", use_container_width=True):
        with st.spinner("Running model…"):
            try:
                input_df = _build_input_row(
                    spark, origin, dest, airline, month, day_of_week, dep_hour,
                    elapsed_time, congestion, prior_delay,
                    temp, dew, wind_speed, wind_dir, visibility, slp,
                )
                p_delayed, expected_delay = _predict(spark, clf_model, reg_model, input_df)

                # ── Results display ──────────────────────────────────────────
                r1, r2 = st.columns(2)
                with r1:
                    st.metric(
                        label="P(Departure Delayed)",
                        value=f"{p_delayed:.1%}",
                        delta=f"{'High risk' if p_delayed > 0.5 else 'Low risk'}",
                        delta_color="inverse",
                    )
                with r2:
                    st.metric(
                        label="Expected Departure Delay",
                        value=f"{expected_delay:.1f} min",
                        delta=f"{'Significant' if expected_delay > 15 else 'Minor'}",
                        delta_color="inverse",
                    )

                # Colour-coded assessment
                if expected_delay < 5:
                    st.success("✅ Flight is likely on time.")
                elif expected_delay < 15:
                    st.info("⏱ Minor delay possible — plan for a small buffer.")
                elif expected_delay < 45:
                    st.warning("⚠️ Moderate delay expected — notify connecting passengers.")
                else:
                    st.error("🚨 Significant delay likely — consider rebooking options.")

                # Breakdown
                with st.expander("Model details"):
                    st.write(f"**Stage 1 — P(delay > 0):** `{p_delayed:.4f}`")
                    cond = expected_delay / p_delayed if p_delayed > 1e-6 else 0.0
                    st.write(f"**Stage 2 — E[delay | delay > 0]:** `{cond:.1f} min`")
                    st.write(f"**Combined — E[delay]:** `{expected_delay:.1f} min`")
                    st.caption(
                        "Note: R² ≈ 0.05 across all models — flight delay is "
                        "driven largely by unobserved factors (ATC, crew, gate conflicts). "
                        "Use this as a probabilistic signal, not a precise forecast."
                    )

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")


if __name__ == "__main__":
    main()
