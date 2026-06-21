# smart-agriculture-yield-prediction
Machine Learning project using Random Forest to predict crop yield.
# 🌾 Smart Agriculture Yield Prediction Using Machine Learning (Random Forest)

![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-RandomForest-green?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-2E4A62?style=for-the-badge)
![Data Science](https://img.shields.io/badge/Data%20Science-blue?style=for-the-badge)
![PyCharm](https://img.shields.io/badge/PyCharm-000000?style=for-the-badge&logo=pycharm&logoColor=marun)
---

# 📌 Project Overview

The **Smart Agriculture Yield Prediction System** is a Machine Learning project designed to predict crop yield based on real-world agricultural factors such as crop type, season, state, area, fertilizer usage, and pesticide usage.

The system is built using **Random Forest Regressor**, which is a powerful ensemble learning algorithm capable of handling complex non-linear relationships in agricultural datasets.

This project demonstrates how **AI and Machine Learning can be used in agriculture** to help farmers and policymakers make data-driven decisions for better crop production planning.

---

# 🚀 Problem Statement

Traditional farming decisions are mostly based on experience and guesswork, which leads to:

- Unpredictable crop yield
- Inefficient fertilizer usage
- Poor resource planning
- Low productivity in some regions

This project solves this problem by using **Machine Learning to predict crop yield accurately before cultivation**.

---

# ⚙️ How the Model Works

1. Data is collected from agricultural records (Kaggle dataset)
2. Data preprocessing and cleaning is performed
3. Categorical features are encoded
4. Dataset is split into training and testing sets
5. Random Forest model is trained
6. Predictions are made on unseen data
7. Model performance is evaluated using R², MAE, RMSE
8. Insights are visualized using graphs

---

# ❌ Initial Approach vs ✅ Improved Approach

## ❌ Initial Model (With Data Leakage)

Initially, the model included the **production column**, which directly influenced yield prediction.

### Issues:
- Unrealistically high accuracy
- Data leakage problem
- Not suitable for real-world deployment

### Model Performance:
- R² Score: **0.91**

---

## ✅ Improved Model (After Fixing Leakage)

Production feature was removed to ensure realistic prediction.

### Final Features Used:
- Crop
- Year
- Season
- State
- Area
- Fertilizer
- Pesticide

### Model Performance:
- R² Score: **~0.84**
- More realistic and production-ready model

---

# 📊 Model Evaluation Comparison

| Model Version | Features Used | R² Score | Remarks |
|---|---|---|---|
| Initial Model | Includes Production | 0.91 | Data Leakage (Not valid) |
| Final Model | Without Production | ~0.84 | Realistic & Production Ready |

---

# 📁 Dataset Information

- Source: Kaggle Crop Yield Dataset  
- Size: ~19,000 records  
- Features: 9 columns  
- Target Variable: Yield  

---

# 🧠 Machine Learning Model

### Algorithm Used:
- Random Forest Regressor

### Why Random Forest?
- Handles non-linear relationships
- Reduces overfitting
- Works well with mixed data types
- High accuracy for tabular datasets

---

# 📊 Visualizations Included

- Yield Distribution
- Top Crops by Yield
- Yield Trends Over Years
- Fertilizer vs Yield
- Pesticide vs Yield
- Area vs Yield
- Correlation Heatmap
- Feature Importance Analysis
- Actual vs Predicted Yield

All graphs are saved in the `/plots` folder.

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data handling |
| NumPy | Numerical computations |
| Scikit-Learn | Machine Learning |
| Random Forest | Prediction model |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |

---

# 📌 Features of This Project

- End-to-end ML pipeline
- Real-world agricultural prediction system
- Data preprocessing + encoding
- Model training and evaluation
- Feature importance analysis
- Professional visualizations
- Leakage-free ML model
- Export-ready plots for portfolio

---

# 🌍 Applications

- Smart Farming Systems
- Agricultural Planning
- Crop Yield Forecasting
- Government Policy Planning
- AgriTech Solutions
- Data-driven farming decisions

---

# 📈 Key Insights

- Production caused data leakage (removed in final model)
- Area and fertilizer strongly influence yield
- Crop type significantly affects yield variation
- Model becomes realistic after removing leakage features

---

# 📂 Project Structure

agrii/
│
├── data/
│   └── crop_yield.csv
│
├── src/
│   └── train.py
│
├── plots/
│   └── (all generated graphs)
│
├── requirements.txt
└── README.md

---

# 🔮 Future Improvements

- Streamlit web app deployment
- Real-time prediction system
- Weather API integration
- Advanced models (XGBoost / LightGBM)
- Crop recommendation system
- Farmer advisory system

---

# ⚙️ Installation

pip install pandas numpy matplotlib seaborn scikit-learn

---

# ▶️ Run Project

python src/train.py

---

# 🧑‍💻 Author

Built as a Machine Learning portfolio project to demonstrate:
- Data Science skills
- ML pipeline understanding
- Real-world problem solving
- Model evaluation knowledge
