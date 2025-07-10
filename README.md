
# 🇮🇳 ML-Based Poverty Signal Detector from Non-Financial Clues

A machine learning project to predict whether a household is **Below Poverty Line (BPL)** using **non-financial features** such as education level, housing type, sanitation, electricity access, and more — no direct income data used.

---

## 🎯 Objective

To develop a classification model that uses **observable, survey-friendly socio-economic indicators** to predict poverty status. This approach supports fast, ethical, and scalable poverty detection, especially in rural or undocumented regions of India.

---

## 📊 Dataset Overview

- 🔢 **Rows:** 5,000 (synthetically generated with realistic distribution)
- 📁 **Columns:** 15 total (categorical + numerical)
- 🎯 **Target:** `poverty_status` (1 = BPL, 0 = Non-BPL)

### 📌 Sample Features:
| Feature Name          | Type         | Description                              |
|----------------------|--------------|------------------------------------------|
| `education_level`     | Categorical  | Highest education attained in household   |
| `housing_type`        | Categorical  | Kutcha, Semi-Pucca, Pucca                |
| `electricity_access`  | Binary       | 1 = Yes, 0 = No                           |
| `infrastructure_score`| Numerical    | Score from 0 to 100                       |
| `internet_access`     | Binary       | 1 = Yes, 0 = No                           |
| `toilet_access`       | Categorical  | None, Shared, Private                     |
| `employment_type`     | Categorical  | Government, Informal, Unemployed, etc.    |
| `health_index`        | Numerical    | Score from 0.0 to 1.0                     |

---

## 🤖 Models Trained and Evaluated

I implemented multiple models to predict poverty status. All were evaluated on a 20% test set (1,000 samples) using precision, recall, F1-score, and accuracy.

| Model                 | Accuracy | Precision (BPL) | Recall (BPL) | F1-Score (BPL) | Notes |
|-----------------------|----------|------------------|---------------|----------------|-------|
| 🏆 **XGBoost**              | **1.00**     | **1.00**           | **1.00**         | **1.00**          | Perfect classification |
| ✅ Support Vector Machine (SVM) | 0.98     | 0.98             | 0.99           | 0.99          | High performer |
| ✅ Logistic Regression          | 0.95     | 0.96             | 0.99           | 0.97          | Lightweight and fast |
| ⚠️ Linear Regression (Thresholded) | 0.90     | 0.90             | 1.00           | 0.95          | Overpredicts BPL class |

---

### 📉 Confusion Matrices

#### 🏆 XGBoost
```
[[105   0]
 [  0 895]]
```

#### 🔷 SVM
```
[[ 88  17]
 [  5 890]]
```

#### 🔷 Logistic Regression
```
[[ 66  39]
 [ 11 884]]
```

#### 🔷 Linear Regression (threshold ≥ 0.5)
```
[[  4 101]
 [  0 895]]
```

---

## 🔍 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- Feature Importance (from XGBoost)

---

## 📊 Visualizations

Created using **Seaborn** and **Matplotlib**:
- Countplots for education, housing, transport mode, etc.
- Boxplots comparing infra score and health index by poverty status
- Heatmap of feature correlations
- Feature importance chart
- Confusion matrix for all models

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`
- **Environment:** Jupyter Notebook / Google Colab

---

## 🧪 File Structure

```
ML-Poverty-Detector/
│
├── data/
│   └── india_poverty_signal_dataset_5000_enhanced.csv
├── notebooks/
│   └── poverty_prediction_models.ipynb
├── models/
│   └── xgboost_model.pkl
├── README.md
├── requirements.txt
└── assets/
    ├── feature_importance.png
    └── confusion_matrix_xgb.png
```

---

## 📍 Future Scope

- 🔁 Add real-world financial features (income, subsidies)
- 🌐 Deploy as a Streamlit or Flask web app
- 🛰️ Integrate with remote sensing or government survey data
- 📱 Mobile app interface for NGOs to conduct real-time poverty classification

---

## 👩‍💻   NAME

- **Juhi** – ML Model Developer ,Data Visualization & EDA  
*B.Tech 2nd Year Project – 2025*

---
## 🙏 Acknowledgements

Inspired by India’s digital inclusion goals, rural development challenges, and the power of data to drive change.

