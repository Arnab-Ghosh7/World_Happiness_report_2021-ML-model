import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import pairwise_distances

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Happiness Predictor",
    page_icon="😊",
    layout="centered"
)

# --------------------------------------------------
# Load Model Artifact
# --------------------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "happiness_model.pkl")
    artifact = joblib.load(model_path)
    return artifact["model"], artifact["features"]

# --------------------------------------------------
# Load Dataset (for country inference)
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("world-happiness-report-2021.csv")

try:
    model, FEATURES = load_model()
    data = load_data()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# --------------------------------------------------
# Helper: Find Closest Country
# --------------------------------------------------
def find_closest_country(input_df, dataset, features):
    distances = pairwise_distances(
        dataset[features],
        input_df,
        metric="euclidean"
    )
    closest_index = distances.argmin()
    return dataset.iloc[closest_index]["Country name"]

# --------------------------------------------------
# Title & Description
# --------------------------------------------------
st.title("🌍 World Happiness Score Predictor")
st.markdown("""
This application predicts the **Happiness (Ladder) Score** using a  
**Random Forest Regression model** trained on the  
**World Happiness Report 2021** dataset.

The country name is **automatically inferred** based on the closest
matching socio-economic profile.
""")

st.divider()

# --------------------------------------------------
# Input Form
# --------------------------------------------------
st.header("Socio-Economic Indicators")

col1, col2 = st.columns(2)

with col1:
    log_gdp = st.number_input(
        "Logged GDP per capita",
        min_value=0.0, max_value=20.0,
        value=10.0, step=0.1
    )
    social_support = st.slider(
        "Social support",
        min_value=0.0, max_value=1.0,
        value=0.8, step=0.01
    )
    healthy_life = st.number_input(
        "Healthy life expectancy",
        min_value=0.0, max_value=100.0,
        value=70.0, step=0.1
    )

with col2:
    freedom = st.slider(
        "Freedom to make life choices",
        min_value=0.0, max_value=1.0,
        value=0.8, step=0.01
    )
    generosity = st.number_input(
        "Generosity",
        min_value=-1.0, max_value=1.0,
        value=0.0, step=0.01
    )
    corruption = st.slider(
        "Perceptions of corruption",
        min_value=0.0, max_value=1.0,
        value=0.7, step=0.01
    )

st.divider()

# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------
if st.button("Predict Happiness Score", type="primary", use_container_width=True):

    # Prepare input
    input_data = pd.DataFrame({
        "Logged GDP per capita": [log_gdp],
        "Social support": [social_support],
        "Healthy life expectancy": [healthy_life],
        "Freedom to make life choices": [freedom],
        "Generosity": [generosity],
        "Perceptions of corruption": [corruption]
    })

    # Ensure correct feature order
    input_data = input_data[FEATURES]

    # Predict score
    prediction = model.predict(input_data)[0]

    # Infer closest country
    closest_country = find_closest_country(
        input_data,
        data,
        FEATURES
    )

    # Display result
    st.success(
        f"### 🌍 Predicted Happiness Score for **{closest_country}**: {prediction:.3f}"
    )

    # Interpretation
    if prediction > 7.5:
        st.balloons()
        st.markdown("**Very happy country** 🌈 (Top global performers)")
    elif prediction < 4.0:
        st.markdown("**Significant challenges** ⚠️")
    else:
        st.markdown("**Average happiness level** 🙂")

    # Show input data
    with st.expander("Show Features Used"):
        st.dataframe(input_data)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Model trained on World Happiness Report 2021 data.")
