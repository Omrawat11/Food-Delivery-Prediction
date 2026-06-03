# 🛵 DeliveryIQ — Food Delivery Time Predictor

> A professional, portfolio-grade ML web application built with Streamlit and scikit-learn.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/sklearn-LinearRegression-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Project Summary

**DeliveryIQ** predicts food delivery ETAs using a trained **Linear Regression** model.
The app processes 7 real-world features — distance, weather, traffic, time of day, vehicle type, prep time, and courier experience — then outputs an estimated delivery time with a confidence interval.

| Metric | Value |
|--------|-------|
| R² Score | 0.775 |
| MAE | 6.96 min |
| RMSE | 9.60 min |
| MSE | 92.13 |
| Dataset Size | 1,000 orders |
| Features | 7 |

---

## 🗂 Folder Structure

```
food_delivery_app/
├── app.py                    ← Main Streamlit application (5 pages)
├── requirements.txt          ← Python dependencies
├── README.md                 ← Project documentation (this file)
├── Food_Delivery_Times.csv   ← Dataset (1000 rows × 9 columns)
├── model/
│   └── LinearRegression.pkl  ← Trained sklearn model (retrained at runtime)
└── assets/
    └── (icons, images)       ← Static assets (optional)
```

---

## 🖥 App Pages

| Page | Description |
|------|-------------|
| 🏠 **Home** | KPI cards, ETA distribution, vehicle/weather/traffic charts, scatter plot |
| 🔮 **Predict ETA** | Live prediction form, gauge chart, feature contributions, CSV download |
| 📊 **Dataset Insights** | Raw data explorer, distributions, correlation heatmap, category analysis |
| 📈 **Model Performance** | Actual vs predicted, residuals, feature importance, model equation |
| 📘 **About Project** | Methodology, tech stack, project structure, deployment guide |

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- pip

### Local Setup

```bash
# 1. Navigate into project folder
cd food_delivery_app

# 2. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**

---

## ☁️ Deploy on Streamlit Cloud

1. Push this project to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** → Done! 🎉

> **Note:** Make sure `Food_Delivery_Times.csv` is in the repo root and `requirements.txt` is complete.

---

## 🔬 Dataset

**File:** `Food_Delivery_Times.csv`

| Column | Type | Description |
|--------|------|-------------|
| `Order_ID` | int | Unique order identifier (dropped for training) |
| `Distance_km` | float | Delivery distance in km |
| `Weather` | categorical | Clear / Foggy / Rainy / Snowy / Windy |
| `Traffic_Level` | categorical | Low / Medium / High |
| `Time_of_Day` | categorical | Morning / Afternoon / Evening / Night |
| `Vehicle_Type` | categorical | Scooter / Bike / Car |
| `Preparation_Time_min` | int | Restaurant prep time (minutes) |
| `Courier_Experience_yrs` | float | Courier work experience (years) |
| `Delivery_Time_min` | int | **Target** — actual delivery time (minutes) |

---

## 🤖 Model Details

- **Algorithm:** Ordinary Least Squares Linear Regression (`sklearn.linear_model.LinearRegression`)
- **Preprocessing:** LabelEncoder for categorical features, dropna for missing values
- **Split:** 80% train / 20% test, `random_state=42`
- **Regularization:** None (pure OLS)

### Model Equation (approximate)
```
ETA = 15.66
    + 2.97 × Distance_km
    + 1.63 × Weather
    - 2.36 × Traffic_Level
    + 0.91 × Preparation_Time_min
    - 0.64 × Courier_Experience_yrs
    ...
```

---

## 🛠 Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.10+ | Core language |
| Streamlit | Interactive web app framework |
| scikit-learn | Machine learning model |
| Pandas | Data manipulation |
| NumPy | Numerical computations |
| Plotly | Interactive visualizations |
| Custom CSS | Premium dark UI theme |

---

## 🎓 Learning Outcomes

This project covers:
- ✅ End-to-end ML pipeline (clean → train → evaluate → deploy)
- ✅ Streamlit multi-page app architecture
- ✅ Professional dark-themed UI with custom CSS
- ✅ Interactive Plotly charts (scatter, histogram, violin, gauge, heatmap)
- ✅ Real-time prediction with confidence intervals
- ✅ CSV download of predictions
- ✅ Model evaluation (MAE, MSE, RMSE, R²)
- ✅ Feature contribution visualization

---

## 👤 Author

Built as a portfolio project for AIML students.
Feel free to fork, extend, and deploy.

---

## 📄 License

MIT License — free to use, modify, and distribute.
