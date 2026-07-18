# 🌾 Crop Recommendation System

A machine learning powered web application that recommends the most suitable crop to cultivate based on soil nutrient levels and climatic conditions. Built with a **Random Forest Classifier** and an interactive **Streamlit** frontend.

---

## 📌 Overview

Choosing the right crop for a given plot of land is a critical decision for farmers, and it depends on multiple interacting factors — soil nutrient composition, temperature, humidity, soil pH, and rainfall. This project uses a supervised machine learning model trained on historical agricultural data to recommend the crop best suited to a given set of conditions, and presents it through a simple, interactive web interface.

**Live demo:** (Streamlit Cloud link here once deployed)

---

## ✨ Features

- 🔢 Interactive sliders to input soil nutrients (N, P, K, pH) and climate data (temperature, humidity, rainfall)
- 🌱 Instant crop recommendation powered by a trained Random Forest model
- 📊 Top-5 probability breakdown showing the model's confidence across the most likely crops
- 📈 Dataset exploration page with crop distribution and feature histograms
- ℹ️ Model information page describing the algorithm, features, and performance
- 🎨 Clean, responsive UI built entirely in Python (no separate frontend framework needed)

---

## 🧠 Model Details

| Attribute | Value |
|---|---|
| Algorithm | Random Forest Classifier (scikit-learn) |
| Number of trees | 100 |
| Number of crop classes | 22 |
| Dataset size | 2,200 records (post-cleaning: 2,173) |
| Train / Test split | 80% training / 20% testing |

**Input Features**

| Feature | Description | Typical Range |
|---|---|---|
| Nitrogen (N) | Soil nitrogen content | 0 – 140 |
| Phosphorus (P) | Soil phosphorus content | 5 – 145 |
| Potassium (K) | Soil potassium content | 5 – 205 |
| Temperature | Ambient temperature (°C) | 8 – 44 |
| Humidity | Relative humidity (%) | 14 – 100 |
| pH Value | Soil pH | 3.5 – 10.0 |
| Rainfall | Rainfall (mm) | 20 – 300 |

**Target:** Crop type (22 classes, e.g. Rice, Maize, Banana, Coffee, Cotton, Jute, etc.)

The full data cleaning, exploratory analysis, model training, and evaluation process (including cross-validation batches and confusion matrices) is documented in [`notebook/cropcode.ipynb`](notebook/cropcode.ipynb).

---

## 📓 Notebook Details

The [`notebook/cropcode.ipynb`](notebook/cropcode.ipynb) file contains the complete end-to-end machine learning workflow used to build this project, including:

- **Data loading** – reading the raw `Crop_Recommendation.csv` dataset (2,200 rows, 8 columns)
- **Data cleaning** – removing duplicate rows and rows with invalid zero values, resulting in a cleaned dataset of 2,173 records
- **Train/test split** – splitting the cleaned data into training (80%) and testing (20%) sets
- **Label encoding** – encoding the categorical `Crop` target column into numeric labels using `LabelEncoder`
- **Model training** – training a `RandomForestClassifier` (100 trees, `random_state=42`) on the training set
- **Model evaluation** – computing accuracy, a full classification report, and confusion matrices on the test set
- **Cross-validation batches** – using `StratifiedKFold` (5 splits) to create balanced data batches and validate model performance across each one, confirming consistent accuracy (~99.5–100%)
- **Model export** – saving the final trained model and label encoder as `random_forest_crop_model.pkl` and `label_encoder.pkl` using `joblib`, which are then loaded directly by `app.py` for real-time predictions
- **Manual prediction testing** – a final cell for testing the saved model with custom, user-entered input values

This notebook is meant to document the reasoning and experimentation behind the model — the deployed app (`app.py`) uses the exported `.pkl` files directly and does not re-run the notebook.

---

## 🗂️ Project Structure

```
crop-recommendation-app/
├── app.py                         # Streamlit application (frontend + inference)
├── random_forest_crop_model.pkl   # Trained Random Forest model
├── label_encoder.pkl              # Label encoder for crop class names
├── Crop_Recommendation.csv        # Training dataset
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── .gitignore                     # Files/folders excluded from version control
└── notebook/
    └── cropcode.ipynb             # Data cleaning, model training & evaluation notebook
```

---

## ⚙️ Tech Stack

- **Language:** Python 3.11
- **ML Library:** scikit-learn
- **Data Handling:** pandas, NumPy
- **Frontend / App Framework:** Streamlit
- **Visualization:** Plotly, Matplotlib
- **Model Persistence:** joblib

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+ installed
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/crop-recommendation-app.git
cd crop-recommendation-app

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`. If it doesn't, copy that URL manually into your browser.

---

## 🌐 Deployment

This app can be deployed for free using **[Streamlit Community Cloud](https://streamlit.io/cloud)**:

1. Push this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repository, branch `main`, and file `app.py`.
4. Click **Deploy**.

Your app will be live at a public URL (e.g. `https://your-app-name.streamlit.app`) that anyone can access without installing anything.

---

## 🔮 How It Works

1. The user provides soil nutrient values (N, P, K, pH) and climate conditions (temperature, humidity, rainfall) via interactive sliders.
2. The input is passed to the pre-trained Random Forest model.
3. The model predicts the most probable crop class and computes prediction probabilities for all 22 crops.
4. The app displays the top recommendation along with a confidence score and a bar chart of the top 5 most likely crops.

---

## 📊 Dataset

The model is trained on a dataset of 2,200 records covering 22 crop types, with each record containing soil nutrient levels (N, P, K), climate readings (temperature, humidity, rainfall), and soil pH. Duplicate and invalid (zero-value) rows were removed during preprocessing, resulting in a cleaned dataset of 2,173 records used for training and evaluation.

---

## 🛣️ Future Improvements

- Add fertilizer and soil-treatment recommendations alongside crop suggestions
- Integrate live weather API data to auto-fill temperature, humidity, and rainfall
- Add geolocation-based recommendations
- Experiment with additional models (XGBoost, Gradient Boosting) for comparison
- Add unit tests and CI/CD pipeline for automated testing

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
