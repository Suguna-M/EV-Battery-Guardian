import streamlit as st
import pandas as pd
import joblib
import os

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="EV Battery Guardian",
    page_icon="🚗",
    layout="wide"
)

# =========================================================
# LOAD AI MODEL
# =========================================================

@st.cache_resource
def load_model():
    model = joblib.load(
        "models/battery_anomaly_model.pkl"
    )

    scaler = joblib.load(
        "models/battery_scaler.pkl"
    )

    features = joblib.load(
        "models/battery_features.pkl"
    )

    return model, scaler, features


model, scaler, features = load_model()


# =========================================================
# BATTERY HEALTH CALCULATION
# =========================================================

def calculate_health(voltage, current, temperature):

    health = 100

    # Voltage Check
    if voltage < 11:
        health -= 25

    elif voltage > 13.5:
        health -= 15

    # Current Check
    if abs(current) > 5:
        health -= 20

    # Temperature Check
    if temperature > 45:
        health -= 30

    elif temperature > 40:
        health -= 15

    elif temperature < 0:
        health -= 10

    return max(0, min(100, health))


# =========================================================
# RISK LEVEL
# =========================================================

def get_risk_level(prediction, health, temperature):

    if prediction == -1:
        return "HIGH 🔴"

    if temperature > 45:
        return "HIGH 🔴"

    elif health < 50:
        return "HIGH 🔴"

    elif health < 80:
        return "MEDIUM 🟡"

    else:
        return "LOW 🟢"


# =========================================================
# PAGE TITLE
# =========================================================

st.title("🚗 EV BATTERY GUARDIAN")

st.markdown(
    "### AI-Powered Battery Health Monitoring & Anomaly Detection System"
)

st.divider()


# =========================================================
# SIDEBAR INPUT
# =========================================================

st.sidebar.header("🔧 Battery Sensor Input")

voltage = st.sidebar.slider(
    "⚡ Voltage (V)",
    min_value=8.0,
    max_value=15.0,
    value=12.4,
    step=0.1
)

current = st.sidebar.slider(
    "🔌 Current (A)",
    min_value=-10.0,
    max_value=10.0,
    value=2.1,
    step=0.1
)

temperature = st.sidebar.slider(
    "🌡️ Temperature (°C)",
    min_value=0.0,
    max_value=100.0,
    value=34.0,
    step=1.0
)


# =========================================================
# CREATE INPUT DATA
# =========================================================

input_data = pd.DataFrame(
    [[voltage, current, temperature]],
    columns=features
)


# =========================================================
# SCALE DATA
# =========================================================

input_scaled = scaler.transform(
    input_data
)


# =========================================================
# AI PREDICTION
# =========================================================

prediction = model.predict(
    input_scaled
)[0]

anomaly_score = model.decision_function(
    input_scaled
)[0]


# =========================================================
# HEALTH & RISK
# =========================================================

battery_health = calculate_health(
    voltage,
    current,
    temperature
)

risk_level = get_risk_level(
    prediction,
    battery_health,
    temperature
)


# =========================================================
# STATUS
# =========================================================

if prediction == 1:

    ai_status = "NORMAL OPERATION"

else:

    ai_status = "ANOMALY DETECTED"


# =========================================================
# DASHBOARD METRICS - ROW 1
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🔋 Battery SOC",
        "78%"
    )

with col2:
    st.metric(
        "🌡️ Temperature",
        f"{temperature} °C"
    )

with col3:
    st.metric(
        "⚡ Voltage",
        f"{voltage} V"
    )


# =========================================================
# DASHBOARD METRICS - ROW 2
# =========================================================

col4, col5, col6 = st.columns(3)

with col4:
    st.metric(
        "🔌 Current",
        f"{current} A"
    )

with col5:
    st.metric(
        "❤️ Battery Health",
        f"{battery_health}%"
    )

with col6:
    st.metric(
        "⚠️ Risk Level",
        risk_level
    )


st.divider()


# =========================================================
# AI STATUS
# =========================================================

st.subheader("🤖 AI STATUS")

if prediction == 1:

    st.success(
        "✓ Normal Battery Operation"
    )

else:

    st.error(
        "⚠️ ANOMALY DETECTED!"
    )

    # Temperature Warning
    if temperature > 45:

        st.warning(
            "🔥 Abnormal Temperature Rise Detected"
        )

    # Voltage Warning
    if voltage < 11:

        st.warning(
            "⚡ Abnormal Voltage Drop Detected"
        )

    elif voltage > 13.5:

        st.warning(
            "⚡ Abnormal Voltage Increase Detected"
        )

    # Current Warning
    if abs(current) > 5:

        st.warning(
            "🔌 Unusual Current / Charging Pattern Detected"
        )


# =========================================================
# AI DETAILS
# =========================================================

st.divider()

st.subheader("📊 AI Analysis")

col1, col2 = st.columns(2)

with col1:

    st.write("**AI Prediction:**")

    if prediction == 1:

        st.success("NORMAL")

    else:

        st.error("ANOMALY")


with col2:

    st.write("**Anomaly Score:**")

    st.info(
        round(float(anomaly_score), 4)
    )


# =========================================================
# SENSOR DATA TABLE
# =========================================================

st.divider()

st.subheader("📋 Current Battery Sensor Data")

st.dataframe(
    input_data,
    use_container_width=True
)


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.divider()

st.subheader("ℹ️ About EV Battery Guardian")

st.write(
    """
    This system uses Machine Learning to monitor EV battery
    parameters and identify abnormal operating conditions.

    The AI model analyzes:

    • Voltage

    • Current

    • Temperature

    The system uses Isolation Forest for anomaly detection.
    """
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "EV Battery Guardian | AI + Machine Learning + Streamlit"
)
