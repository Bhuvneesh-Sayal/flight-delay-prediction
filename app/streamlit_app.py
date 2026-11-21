"""
app/streamlit_app.py
====================
Two-tab Streamlit UI:

  Tab 1 — Instant single-flight prediction
    Uses the saved hurdle model directly (local Spark, < 1 second).

  Tab 2 — Real-time Kafka prediction
    1. User selects a flight and clicks "Predict via Kafka"
    2. App publishes the flight event to the `flight-events` Kafka topic
    3. The Spark Structured Streaming pipeline picks it up, scores it,
       and publishes the result to `delay-predictions`
    4. App polls `delay-predictions` for the matching flight_key and
       displays the result (up to 60 s timeout)

    Requires:
      - docker compose up -d  (Kafka running)
      - python main.py --steps stream --kafka  (scoring pipeline)

Run with:
    streamlit run app/streamlit_app.py
"""

import json
import sys
import time
import uuid
from datetime import date, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.settings import (
    AIRPORT_RUNWAYS,
    KAFKA_FLIGHT_TOPIC,
    KAFKA_SCORES_TOPIC,
    PATHS,
    TOP_10_AIRPORTS,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
WEATHER_PRESETS = {
    "☀️  Clear":         dict(temp=20.0,  dew=-5.0,  wind_speed=4.0,  wind_dir=180.0, visibility=9999.0, slp=1015.0),
    "⛅  Partly Cloudy":  dict(temp=15.0,  dew=5.0,   wind_speed=7.0,  wind_dir=200.0, visibility=9000.0, slp=1012.0),
    "🌧️  Rainy":          dict(temp=10.0,  dew=8.0,   wind_speed=10.0, wind_dir=220.0, visibility=4000.0, slp=1005.0),
    "⛈️  Stormy":         dict(temp=8.0,   dew=7.0,   wind_speed=18.0, wind_dir=250.0, visibility=1500.0, slp=995.0),
    "🌫️  Foggy":          dict(temp=12.0,  dew=11.0,  wind_speed=3.0,  wind_dir=180.0, visibility=800.0,  slp=1010.0),
    "❄️  Icy / Snow":     dict(temp=-3.0,  dew=-5.0,  wind_speed=8.0,  wind_dir=300.0, visibility=3000.0, slp=1020.0),
}

CONGESTION_BY_HOUR = {
    0: 2,  1: 1,  2: 1,  3: 1,  4: 2,  5: 5,
    6: 12, 7: 15, 8: 14, 9: 13, 10: 13, 11: 14,
    12: 15, 13: 15, 14: 15, 15: 16, 16: 16, 17: 17,
    18: 16, 19: 14, 20: 12, 21: 10, 22: 7,  23: 4,
}

AIRLINES = {
    "AA — American Airlines":  "AA",
    "DL — Delta Air Lines":    "DL",
    "UA — United Airlines":    "UA",
    "WN — Southwest Airlines": "WN",
    "B6 — JetBlue":            "B6",
    "AS — Alaska Airlines":    "AS",
    "NK — Spirit Airlines":    "NK",
    "F9 — Frontier Airlines":  "F9",
}

# Both producer and consumer use kafka:9092 — resolves via /etc/hosts to 127.0.0.1
# This matches what Kafka advertises internally so there's no reconnect confusion.
_KAFKA_BOOTSTRAP = "kafka:9092"
_KAFKA_API_VER   = (2, 5, 0)   # skip broker version negotiation


# ---------------------------------------------------------------------------
# Spark + model loading (Tab 1)
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
            .config("spark.ui.enabled", "false")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.driver.host", "127.0.0.1")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")

        clf_path = PATHS["models"] / "hurdle_departure_bts_weather_classifier"
        reg_path = PATHS["models"] / "hurdle_departure_bts_weather_regressor"
        if not clf_path.exists():
            clf_path = PATHS["models"] / "hurdle_departure_full_classifier"
            reg_path = PATHS["models"] / "hurdle_departure_full_regressor"
        if not clf_path.exists():
            clf_path = PATHS["models"] / "hurdle_departure_classifier"
            reg_path = PATHS["models"] / "hurdle_departure_regressor"

        if not clf_path.exists() or not reg_path.exists():
            return None, None, None

        return spark, PipelineModel.load(str(clf_path)), PipelineModel.load(str(reg_path))

    except Exception:
        import logging as _l
        _l.getLogger(__name__).exception("Failed to load models")
        st.error("Failed to load models — check logs/pipeline.log for details.")
        return None, None, None


def _build_input_row(spark, origin, dest, airline, flight_date, dep_time,
                     elapsed_time, congestion, prior_delay, weather):
    rad = np.pi / 180.0
    rwy = AIRPORT_RUNWAYS.get(origin, 180.0)
    wd  = (weather["wind_dir"] - rwy) * rad
    data = [{
        "Origin":                   origin,
        "Dest":                     dest,
        "Reporting_Airline":        airline,
        "scheduled_month":          flight_date.month,
        "scheduled_day_of_week":    flight_date.isoweekday() % 7 + 1,
        "scheduled_departure_hour": float(dep_time.hour),
        "CRSElapsedTime":           float(elapsed_time),
        "origin_congestion_2h":     float(congestion),
        "prior_leg_delay":          float(prior_delay),
        "w_temp":                   weather["temp"],
        "w_dew":                    weather["dew"],
        "w_wspeed":                 weather["wind_speed"],
        "w_slp":                    weather["slp"],
        "weather_crosswind":        abs(weather["wind_speed"] * np.sin(wd)),
        "weather_headwind":         weather["wind_speed"] * np.cos(wd),
        "weather_icing_risk":       1.0 if (weather["temp"] <= 0.0 and
                                            (weather["temp"] - weather["dew"]) <= 2.0) else 0.0,
        "weather_ifr_conditions":   1.0 if weather["visibility"] < 4800.0 else 0.0,
        "weather_available":        1,
        "weather_age_minutes":      30.0,
    }]
    return spark.createDataFrame(pd.DataFrame(data))


def _predict_local(spark, clf_model, reg_model, input_df):
    from pyspark.sql import types as T
    from pyspark.sql.functions import udf
    get_p1 = udf(lambda v: float(v[1]), T.DoubleType())
    p  = float(clf_model.transform(input_df).select(get_p1("probability")).first()[0])
    lp = float(reg_model.transform(input_df).select("prediction").first()[0])
    return p, p * float(np.expm1(max(lp, 0.0)))


# ---------------------------------------------------------------------------
# Kafka helpers
# ---------------------------------------------------------------------------
def _kafka_available() -> bool:
    """Quick check: can we connect to Kafka?"""
    try:
        from kafka import KafkaProducer
        p = KafkaProducer(
            bootstrap_servers=_KAFKA_BOOTSTRAP,
            api_version=_KAFKA_API_VER,
        )
        p.close()
        return True
    except Exception:
        return False


def _publish_flight_event(flight_key: int, origin: str, dest: str, airline: str,
                          flight_date: date, dep_time: dtime, elapsed_time: float,
                          congestion: float, prior_delay: float, weather: dict) -> bool:
    """Publish one flight event to the flight-events Kafka topic."""
    try:
        from kafka import KafkaProducer
        rad = np.pi / 180.0
        rwy = AIRPORT_RUNWAYS.get(origin, 180.0)
        wd  = (weather["wind_dir"] - rwy) * rad

        msg = {
            "flight_key":               flight_key,
            "FlightDate":               flight_date.strftime("%Y-%m-%d"),
            "Reporting_Airline":        airline,
            "Origin":                   origin,
            "Dest":                     dest,
            "CRSDepTime":               dep_time.hour * 100 + dep_time.minute,
            "CRSElapsedTime":           float(elapsed_time),
            "scheduled_departure_utc":  f"{flight_date}T{dep_time.strftime('%H:%M:%S')}",
            "origin_congestion_2h":     int(congestion),
            "prior_leg_delay":          float(prior_delay),
            "scheduled_month":          flight_date.month,
            "scheduled_day_of_week":    flight_date.isoweekday() % 7 + 1,
            "scheduled_departure_hour": float(dep_time.hour),
        }

        producer = KafkaProducer(
            bootstrap_servers=_KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8"),
            api_version=_KAFKA_API_VER,
        )
        producer.send(KAFKA_FLIGHT_TOPIC, key=flight_key, value=msg)
        producer.flush()
        producer.close()
        return True
    except Exception as e:
        st.error(f"Failed to publish to Kafka: {e}")
        return False


def _create_result_consumer():
    """
    Create a Kafka consumer seeked to the current end of delay-predictions.
    Must be called BEFORE publishing the flight event to avoid missing the response.
    Returns the consumer, or None on failure.
    """
    try:
        from kafka import KafkaConsumer, TopicPartition
        consumer = KafkaConsumer(
            bootstrap_servers=_KAFKA_BOOTSTRAP,
            value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            enable_auto_commit=False,
            group_id=None,
            consumer_timeout_ms=500,
            api_version=_KAFKA_API_VER,
        )

        # poll() triggers metadata fetch — partitions_for_topic returns None
        # without it because kafka-python fetches metadata lazily.
        consumer.poll(timeout_ms=3000)

        partitions = consumer.partitions_for_topic(KAFKA_SCORES_TOPIC)
        if not partitions:
            consumer.close()
            return None

        tps = [TopicPartition(KAFKA_SCORES_TOPIC, p) for p in partitions]
        consumer.assign(tps)
        consumer.seek_to_end(*tps)   # seek_to_end works; end_offsets has a bug in 2.0.5
        return consumer
    except Exception:
        return None


def _wait_for_result(consumer, flight_key: int, timeout_secs: int = 60) -> dict | None:
    """
    Poll *consumer* (already seeked to end) for a message matching *flight_key*.
    Returns the parsed message dict, or None on timeout.
    """
    deadline = time.monotonic() + timeout_secs
    try:
        while time.monotonic() < deadline:
            for msg in consumer:
                v = msg.value
                if v.get("flight_key") == flight_key:
                    return v
        return None
    except Exception:
        return None
    finally:
        consumer.close()


# ---------------------------------------------------------------------------
# Shared flight-details form (used in both tabs)
# ---------------------------------------------------------------------------
def _render_flight_form(key_prefix: str):
    """Render flight + weather inputs. Returns all values."""
    c1, c2, c3 = st.columns(3)
    with c1:
        origin = st.selectbox("Origin Airport", TOP_10_AIRPORTS, key=f"{key_prefix}_origin")
    with c2:
        dest = st.selectbox("Destination Airport",
                            [a for a in TOP_10_AIRPORTS if a != origin],
                            index=1, key=f"{key_prefix}_dest")
    with c3:
        airline_label = st.selectbox("Airline", list(AIRLINES.keys()), key=f"{key_prefix}_airline")
        airline = AIRLINES[airline_label]

    c4, c5 = st.columns(2)
    with c4:
        flight_date = st.date_input("Departure Date", value=date.today(),
                                    min_value=date(2024, 1, 1), max_value=date(2026, 12, 31),
                                    key=f"{key_prefix}_date")
    with c5:
        dep_time = st.time_input("Scheduled Departure Time", value=dtime(8, 0),
                                 step=900, key=f"{key_prefix}_time")

    c6, c7 = st.columns(2)
    with c6:
        elapsed_time = st.number_input("Flight Duration (min)", 60, 600, 180,
                                       step=5, key=f"{key_prefix}_dur")
    with c7:
        prev_delayed = st.checkbox("Previous aircraft leg was delayed",
                                   key=f"{key_prefix}_prev")
        prior_delay  = st.slider("Prior leg delay (min)", 0, 180, 30,
                                 disabled=not prev_delayed, key=f"{key_prefix}_pdelay")
        if not prev_delayed:
            prior_delay = 0.0

    congestion = CONGESTION_BY_HOUR.get(dep_time.hour, 10)
    st.caption(f"ℹ️ Estimated congestion at {dep_time.strftime('%H:%M')}: "
               f"~{congestion} flights in past 2h")

    st.subheader("🌤️ Weather at Origin")
    weather_label = st.radio("Conditions", list(WEATHER_PRESETS.keys()),
                             horizontal=True, key=f"{key_prefix}_weather")
    weather = WEATHER_PRESETS[weather_label]

    with st.expander("Sensor values"):
        w1, w2, w3 = st.columns(3)
        w1.metric("Temperature", f"{weather['temp']}°C")
        w1.metric("Dew Point",   f"{weather['dew']}°C")
        w2.metric("Wind Speed",  f"{weather['wind_speed']} m/s")
        w2.metric("Visibility",  f"{weather['visibility']:.0f} m")
        w3.metric("Pressure",    f"{weather['slp']} hPa")

    return origin, dest, airline, flight_date, dep_time, elapsed_time, congestion, prior_delay, weather


def _render_result(p: float, expected: float, source_label: str = ""):
    """Display prediction result metrics and assessment."""
    if source_label:
        st.caption(f"Source: {source_label}")

    r1, r2 = st.columns(2)
    r1.metric("P(Departure Delayed)", f"{p:.1%}",
              delta="High risk" if p > 0.5 else "Low risk",
              delta_color="inverse")
    r2.metric("Expected Delay", f"{expected:.1f} min",
              delta="Significant" if expected > 15 else "Minor",
              delta_color="inverse")

    if expected < 5:
        st.success("✅ Flight is likely on time.")
    elif expected < 15:
        st.info("⏱ Minor delay possible.")
    elif expected < 45:
        st.warning("⚠️ Moderate delay expected.")
    else:
        st.error("🚨 Significant delay likely.")

    with st.expander("Model details"):
        cond = expected / p if p > 1e-6 else 0.0
        st.write(f"**P(delay > 0):** `{p:.4f}`")
        st.write(f"**E[delay | delay > 0]:** `{cond:.1f} min`")
        st.write(f"**Combined E[delay]:** `{expected:.1f} min`")
        st.caption("R² ≈ 0.02 — delay driven largely by unobservable factors.")


# ---------------------------------------------------------------------------
# Tab 1 — Instant local prediction
# ---------------------------------------------------------------------------
def render_predictor_tab(spark, clf_model, reg_model):
    st.subheader("🛫 Flight Details")
    origin, dest, airline, flight_date, dep_time, elapsed_time, congestion, prior_delay, weather = \
        _render_flight_form("t1")

    st.divider()
    if st.button("🔮 Predict Delay", type="primary", use_container_width=True, key="t1_btn"):
        with st.spinner("Running model…"):
            try:
                row = _build_input_row(spark, origin, dest, airline, flight_date,
                                       dep_time, elapsed_time, congestion,
                                       prior_delay, weather)
                p, expected = _predict_local(spark, clf_model, reg_model, row)
                _render_result(p, expected, source_label="Local Spark (instant)")
            except Exception:
                import logging as _l
                _l.getLogger(__name__).exception("Prediction failed")
                st.error("Prediction failed — check logs/pipeline.log for details.")


# ---------------------------------------------------------------------------
# Tab 2 — Real-time prediction via Kafka
# ---------------------------------------------------------------------------
def render_kafka_tab():
    st.subheader("📡 Real-time Prediction via Kafka")
    st.caption(
        "Your flight is published to **`flight-events`** → "
        "scored by the Spark Structured Streaming pipeline → "
        "result returned from **`delay-predictions`** (up to 60 s)."
    )

    # Pre-flight check
    kafka_ok = _kafka_available()
    if not kafka_ok:
        st.error(
            "❌ Cannot connect to Kafka at `localhost:9092`.  \n"
            "Start the Docker cluster first:\n"
            "```\ndocker compose up -d\n```"
        )
        return

    # Check if streaming pipeline looks like it's running
    st.success("✅ Kafka is reachable at `localhost:9092`")
    st.info(
        "Make sure the scoring pipeline is running in another terminal:  \n"
        "```\nexport JAVA_HOME=$(/usr/libexec/java_home -v 17)\n"
        "export KAFKA_BOOTSTRAP_SERVERS=localhost:9094\n"
        "python main.py --steps stream --kafka\n```"
    )

    st.divider()
    st.subheader("🛫 Flight Details")
    origin, dest, airline, flight_date, dep_time, elapsed_time, congestion, prior_delay, weather = \
        _render_flight_form("t2")

    st.divider()
    if st.button("🚀 Predict via Kafka", type="primary", use_container_width=True, key="t2_btn"):

        # Generate a unique flight_key for this request
        flight_key = abs(hash(f"{origin}{dest}{airline}{flight_date}{dep_time}{time.time()}")) % (2**31)

        with st.status("Sending flight to Kafka pipeline…", expanded=True) as status:

            # Step 1: create consumer and seek to end BEFORE publishing
            # (avoids race condition where response arrives before consumer starts)
            st.write("🔌 Connecting to Kafka consumer…")
            consumer = _create_result_consumer()
            if consumer is None:
                status.update(label="Failed to connect to Kafka", state="error")
                st.error("Could not connect to `delay-predictions` topic. Is Kafka running?")
                return

            # Step 2: publish the flight event
            st.write(f"→ Publishing to `{KAFKA_FLIGHT_TOPIC}` (key: `{flight_key}`)")
            ok = _publish_flight_event(
                flight_key, origin, dest, airline, flight_date, dep_time,
                elapsed_time, congestion, prior_delay, weather,
            )
            if not ok:
                consumer.close()
                status.update(label="Failed to publish", state="error")
                return

            st.write("✅ Published. Waiting for Spark to score…")
            st.write("⏳ Streaming pipeline triggers every 5 s — please wait up to 30 s.")

            # Step 3: wait for the scored result
            result = _wait_for_result(consumer, flight_key, timeout_secs=30)

            if result is None:
                status.update(label="Timed out — no response from pipeline", state="error")
                st.warning(
                    "No result received within 60 s.  \n"
                    "Check that the streaming pipeline is running:\n"
                    "```\npython main.py --steps stream --kafka\n```"
                )
                return

            status.update(label="✅ Score received from Kafka!", state="complete")

        # Step 4: display result
        p_delayed      = result.get("p_delayed", 0.0)
        expected_delay = result.get("expected_delay", p_delayed * 47.0)

        scored_at = result.get("scored_at", "")
        st.caption(f"Scored at: `{scored_at}`  |  flight_key: `{flight_key}`")
        _render_result(p_delayed, expected_delay,
                       source_label="Spark Structured Streaming via Kafka")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.title("✈️ Flight Delay Predictor")
    st.caption(
        "Two-stage hurdle model — DA231o Data Scaling project | "
        "Trained on 2025 US domestic flights (10 hub airports)"
    )

    tab1, tab2 = st.tabs(["🔮 Instant Prediction", "📡 Real-time via Kafka"])

    with tab1:
        spark, clf_model, reg_model = load_models()
        if spark is None:
            st.warning("⚠️ Models not found. Run `python main.py --steps hurdle` first.")
        else:
            render_predictor_tab(spark, clf_model, reg_model)

    with tab2:
        render_kafka_tab()


if __name__ == "__main__":
    main()
