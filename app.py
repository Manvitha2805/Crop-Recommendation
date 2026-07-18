import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ---------- Page Config ----------
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# ---------- Load Model & Encoder ----------
@st.cache_resource
def load_model():
    model = joblib.load("random_forest_crop_model.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, le

model, le = load_model()

FEATURES = ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "pH_Value", "Rainfall"]

# Typical ranges (used for slider bounds), based on training data
RANGES = {
    "Nitrogen":    (0, 140, 50),
    "Phosphorus":  (5, 145, 53),
    "Potassium":   (5, 205, 48),
    "Temperature": (8.0, 44.0, 25.0),
    "Humidity":    (14.0, 100.0, 71.0),
    "pH_Value":    (3.5, 10.0, 6.5),
    "Rainfall":    (20.0, 300.0, 103.0),
}

# ---------- Sidebar ----------
st.sidebar.title("🌾 Crop Recommender")
st.sidebar.markdown(
    "Enter your soil and climate parameters to get an AI-powered crop recommendation, "
    "powered by a Random Forest model."
)
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🔮 Predict", "📊 About the Data", "ℹ️ About the Model"])

# ---------- Predict Page ----------
if page == "🔮 Predict":
    st.title("🌾 Crop Recommendation System")
    st.write("Fill in the soil nutrients and climate conditions below to get a recommended crop.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Soil Nutrients")
        nitrogen = st.slider("Nitrogen (N)", *RANGES["Nitrogen"])
        phosphorus = st.slider("Phosphorus (P)", *RANGES["Phosphorus"])
        potassium = st.slider("Potassium (K)", *RANGES["Potassium"])
        ph_value = st.slider("pH Value", *RANGES["pH_Value"], step=0.1)

    with col2:
        st.subheader("Climate Conditions")
        temperature = st.slider("Temperature (°C)", *RANGES["Temperature"], step=0.1)
        humidity = st.slider("Humidity (%)", *RANGES["Humidity"], step=0.1)
        rainfall = st.slider("Rainfall (mm)", *RANGES["Rainfall"], step=0.1)

    st.markdown("---")

    if st.button("🔍 Recommend Crop", type="primary", use_container_width=True):
        input_df = pd.DataFrame(
            [[nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall]],
            columns=FEATURES
        )

        prediction = model.predict(input_df)
        crop = le.inverse_transform(prediction)[0]

        probabilities = model.predict_proba(input_df)[0]
        prob_df = pd.DataFrame({
            "Crop": le.classes_,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False).head(5)

        st.success(f"### ✅ Recommended Crop: **{crop}**")

        col_a, col_b = st.columns([1, 1.3])
        with col_a:
            st.metric("Confidence", f"{prob_df.iloc[0]['Probability']*100:.1f}%")
            st.write("Your input:")
            st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

        with col_b:
            fig = px.bar(
                prob_df,
                x="Probability",
                y="Crop",
                orientation="h",
                title="Top 5 Predicted Crops",
                color="Probability",
                color_continuous_scale="Greens",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

# ---------- About the Data Page ----------
elif page == "📊 About the Data":
    st.title("📊 About the Dataset")
    df = pd.read_csv("Crop_Recommendation.csv")

    st.write(f"The model was trained on **{len(df)} records** covering **{df['Crop'].nunique()} crop types**.")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Crop Distribution")
    crop_counts = df["Crop"].value_counts().reset_index()
    crop_counts.columns = ["Crop", "Count"]
    fig = px.bar(crop_counts, x="Crop", y="Count", color="Crop")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Distributions")
    feature = st.selectbox("Choose a feature", FEATURES)
    fig2 = px.histogram(df, x=feature, nbins=30, color_discrete_sequence=["#2E8B57"])
    st.plotly_chart(fig2, use_container_width=True)

# ---------- About the Model Page ----------
elif page == "ℹ️ About the Model":
    st.title("ℹ️ About the Model")
    st.markdown("""
    This app uses a **Random Forest Classifier** trained on soil and climate data to
    recommend the most suitable crop to grow.

    **Model details:**
    - Algorithm: `RandomForestClassifier` (scikit-learn)
    - Trees: 100
    - Test accuracy: ~99.5%
    - Features used: Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH Value, Rainfall
    - Target: Crop type (22 classes)

    **How it works:**
    1. You provide soil nutrient levels and climate conditions.
    2. The trained model predicts the crop most suited to those conditions.
    3. The app also shows the top 5 most likely crops with their probabilities.
    """)
