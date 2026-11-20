"""
app/streamlit_app.py
====================
Interactive Streamlit UI for real-time flight delay inference.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.settings import AIRPORT_RUNWAYS, PATHS, TOP_10_AIRPORTS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Weather presets — map human-readable conditions to sensor values
# ---------------------------------------------------------------------------
WEATHER_PRESETS = {
    "☀️  Clear":        dict(temp=20.0,  dew=-5.0,  wind_speed=4.0,  wind_dir=180.0, visibility=9999.0, slp=1015.0),
    "⛅  Partly Cloudy": dict(temp=15.0,  dew=5.0,   wind_speed=7.0,  wind_dir=200.0, visibility=9000.0, slp=1012.0),
    "🌧️  Rainy":         dict(temp=10.0,  dew=8.0,   wind_speed=10.0, wind_dir=220.0, visibility=4000.0, slp=1005.0),
    "⛈️  Stormy":        dict(temp=8.0,   dew=7.0,   wind_speed=18.0, wind_dir=250.0, visibility=1500.0, slp=995.0),
    "🌫️  Foggy":         dict(temp=12.0,  dew=11.0,  wind_speed=3.0,  wind_dir=180.0, visibility=800.0,  slp=1010.0),
    "❄️  Icy / Snow":    dict(temp=-3.0,  dew=-5.0,  wind_speed=8.0,  wind_dir=300.0, visibility=3000.0, slp=1020.0),
}

# Approximate average flights per 2-hour window by departure hour
CONGESTION_BY_HOUR = {
    0: 2,  1: 1,  2: 1,  3: 1,  4: 2,  5: 5,
    6: 12, 7: 15, 8: 14, 9: 13, 10: 13, 11: 14,
    12: 15, 13: 15, 14: 15, 15: 16, 16: 16, 17: 17,
    18: 16, 19: 14, 20: 12, 21: 10, 22: 7,  23: 4,
}

AIRLINES = {
    "AA — American Airlines": "AA",
    "DL — Delta Air Lines":   "DL",
    "UA — United Airlines":   "UA",
    "WN — Southwest Airlines":"WN",
    "B6 — JetBlue":           "B6",
    "AS — Alaska Airlines":   "AS",
    "NK — Spirit Airlines":   "NK",
    "F9 — Frontier Airlines": "F9",
}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading hurdle model…")
def load_models():
    try:
        from pyspark.ml import PipelineModel
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder
            .appName("StreamlitInference")
            .master("local[2]")
            .config("spark.driver.memory", "2g")
            .config("spark.sql.shuffle.partitions", "4")
            # Security: disable the Spark UI so port 4040 is not opened.
            # The UI would otherwise bind to all interfaces and expose job
            # details, config, and local paths to anyone on the network.
            .config("spark.ui.enabled", "false")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.driver.host", "127.0.0.1")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")

        clf_path = PATHS["models"] / "hurdle_departure_bts_weather_classifier"
        reg_path = PATHS["models"] / "hurdle_departure_bts_weather_regressor"

        # Fall back to legacy naming if bts_weather model not found
        if not clf_path.exists():
            clf_path = PATHS["models"] / "hurdle_departure_full_classifier"
            reg_path = PATHS["models"] / "hurdle_departure_full_regressor"
        if not clf_path.exists():
            clf_path = PATHS["models"] / "hurdle_departure_classifier"
            reg_path = PATHS["models"] / "hurdle_departure_regressor"

        if not clf_path.exists() or not reg_path.exists():
            return None, None, None

        clf_model = PipelineModel.load(str(clf_path))
        reg_model = PipelineModel.load(str(reg_path))
        return spark, clf_model, reg_model

    except Exception as exc:
        # Log full detail internally; show a generic message to the user
        # to avoid leaking internal paths or stack traces via the UI.
        import logging as _logging
        _logging.getLogger(__name__).exception("Failed to load models")
        st.error("Failed to load models — check logs/pipeline.log for details.")
        return None, None, None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _build_input_row(spark, origin, dest, airline, flight_date, dep_time,
                     elapsed_time, congestion, prior_delay, weather):
    rad = np.pi / 180.0
    runway_hdg = AIRPORT_RUNWAYS.get(origin, 180.0)
    wind_diff = (weather["wind_dir"] - runway_hdg) * rad
    crosswind = abs(weather["wind_speed"] * np.sin(wind_diff))
    headwind  = weather["wind_speed"] * np.cos(wind_diff)
    icing     = 1.0 if (weather["temp"] <= 0.0 and (weather["temp"] - weather["dew"]) <= 2.0) else 0.0
    ifr       = 1.0 if weather["visibility"] < 4800.0 else 0.0

    data = [{
        "Origin":                   origin,
        "Dest":                     dest,
        "Reporting_Airline":        airline,
        "scheduled_month":          flight_date.month,
        "scheduled_day_of_week":    flight_date.isoweekday() % 7 + 1,  # Sun=1 .. Sat=7
        "scheduled_departure_hour": float(dep_time.hour),
        "CRSElapsedTime":           float(elapsed_time),
        "origin_congestion_2h":     float(congestion),
        "prior_leg_delay":          float(prior_delay),
        "w_temp":                   weather["temp"],
        "w_dew":                    weather["dew"],
        "w_wspeed":                 weather["wind_speed"],
        "w_slp":                    weather["slp"],
        "weather_crosswind":        crosswind,
        "weather_headwind":         headwind,
        "weather_icing_risk":       icing,
        "weather_ifr_conditions":   ifr,
        "weather_available":        1,
        "weather_age_minutes":      30.0,
    }]
    return spark.createDataFrame(pd.DataFrame(data))


def _predict(spark, clf_model, reg_model, input_df):
    from pyspark.sql import types as T
    from pyspark.sql.functions import udf

    get_prob1 = udf(lambda v: float(v[1]), T.DoubleType())
    clf_out   = clf_model.transform(input_df)
    p_delayed = float(clf_out.select(get_prob1("probability")).first()[0])

    reg_out   = reg_model.transform(input_df)
    log_pred  = float(reg_out.select("prediction").first()[0])
    cond_delay = float(np.expm1(max(log_pred, 0.0)))

    return p_delayed, p_delayed * cond_delay


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
        st.warning("⚠️ Models not found. Run `python main.py --steps hurdle` first.")
        st.stop()

    # ── Flight Details ───────────────────────────────────────────────────────
    st.subheader("🛫 Flight Details")

    c1, c2, c3 = st.columns(3)
    with c1:
        origin = st.selectbox("Origin Airport", TOP_10_AIRPORTS)
    with c2:
        dest = st.selectbox("Destination Airport",
                            [a for a in TOP_10_AIRPORTS if a != origin],
                            index=1)
    with c3:
        airline_label = st.selectbox("Airline", list(AIRLINES.keys()))
        airline = AIRLINES[airline_label]

    c4, c5 = st.columns(2)
    with c4:
        flight_date = st.date_input(
            "Departure Date",
            value=date.today(),
            min_value=date(2024, 1, 1),
            max_value=date(2026, 12, 31),
        )
    with c5:
        dep_time = st.time_input("Scheduled Departure Time", value=time(8, 0), step=900)

    c6, c7 = st.columns(2)
    with c6:
        elapsed_time = st.number_input("Scheduled Flight Duration (min)", 60, 600, 180, step=5)
    with c7:
        prev_delayed = st.checkbox("Previous flight on this aircraft was delayed")
        prior_delay  = st.slider("Previous flight delay (min)", 0, 180, 30,
                                 disabled=not prev_delayed)
        if not prev_delayed:
            prior_delay = 0.0

    # Auto-estimate congestion from departure hour
    congestion = CONGESTION_BY_HOUR.get(dep_time.hour, 10)
    st.caption(f"ℹ️ Estimated airport congestion at {dep_time.strftime('%H:%M')}: "
               f"~{congestion} flights in the past 2 hours")

    # ── Weather Conditions ───────────────────────────────────────────────────
    st.subheader("🌤️ Weather Conditions at Origin")
    st.caption("Select the general weather — exact sensor values are estimated automatically.")

    weather_label = st.radio(
        "Current conditions",
        options=list(WEATHER_PRESETS.keys()),
        horizontal=True,
    )
    weather = WEATHER_PRESETS[weather_label]

    # Show what values will be used (collapsed by default)
    with st.expander("View estimated sensor values"):
        wc1, wc2, wc3 = st.columns(3)
        wc1.metric("Temperature", f"{weather['temp']}°C")
        wc1.metric("Dew Point",   f"{weather['dew']}°C")
        wc2.metric("Wind Speed",  f"{weather['wind_speed']} m/s")
        wc2.metric("Visibility",  f"{weather['visibility']:.0f} m")
        wc3.metric("Sea Level Pressure", f"{weather['slp']} hPa")

    # ── Predict ──────────────────────────────────────────────────────────────
    st.divider()
    if st.button("🔮 Predict Delay", type="primary", use_container_width=True):
        with st.spinner("Running model…"):
            try:
                input_df = _build_input_row(
                    spark, origin, dest, airline, flight_date, dep_time,
                    elapsed_time, congestion, prior_delay, weather,
                )
                p_delayed, expected_delay = _predict(spark, clf_model, reg_model, input_df)

                r1, r2 = st.columns(2)
                with r1:
                    st.metric("P(Departure Delayed)",
                              f"{p_delayed:.1%}",
                              delta="High risk" if p_delayed > 0.5 else "Low risk",
                              delta_color="inverse")
                with r2:
                    st.metric("Expected Departure Delay",
                              f"{expected_delay:.1f} min",
                              delta="Significant" if expected_delay > 15 else "Minor",
                              delta_color="inverse")

                if expected_delay < 5:
                    st.success("✅ Flight is likely on time.")
                elif expected_delay < 15:
                    st.info("⏱ Minor delay possible — plan a small buffer.")
                elif expected_delay < 45:
                    st.warning("⚠️ Moderate delay expected.")
                else:
                    st.error("🚨 Significant delay likely.")

                with st.expander("Model details"):
                    cond = expected_delay / p_delayed if p_delayed > 1e-6 else 0.0
                    st.write(f"**P(delay > 0):** `{p_delayed:.4f}`")
                    st.write(f"**E[delay | delay > 0]:** `{cond:.1f} min`")
                    st.write(f"**Combined E[delay]:** `{expected_delay:.1f} min`")
                    st.write(f"**Date:** {flight_date.strftime('%A, %d %b %Y')}  "
                             f"**Departure:** {dep_time.strftime('%H:%M')}")
                    st.caption(
                        "R² ≈ 0.05 across all models. Flight delay is driven largely by "
                        "unobserved factors (ATC, crew, gate conflicts). Use this as a "
                        "probabilistic signal, not a precise forecast."
                    )

            except Exception as exc:
                import logging as _logging
                _logging.getLogger(__name__).exception("Prediction failed")
                st.error("Prediction failed — check logs/pipeline.log for details.")


if __name__ == "__main__":
    main()
