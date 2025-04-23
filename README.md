# Wine-Quality-Prediction
https://colab.research.google.com/drive/1l5Toi2FN7Wc3zEyO_MbSuhS987KGkU2I?usp=sharing 

---

<img width="1470" alt="Screenshot 2025-04-23 at 5 11 53 PM" src="https://github.com/user-attachments/assets/27896eb3-c69e-4889-ad18-30695316d2ac" />
<img width="1470" alt="Screenshot 2025-04-23 at 5 12 46 PM" src="https://github.com/user-attachments/assets/74bcbc00-da10-40c5-93ae-3123c59945c9" />

---

**Data Sources**

https://www.kaggle.com/datasets/yasserh/wine-quality-dataset

---

**Workflow Overview**

1. Data Loading
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Training & Validation
6. Choose algorithm:
Regression: RandomForestRegressor for predicting numeric quality.
Classification: RandomForestClassifier on a binary good-vs-bad target.
7. Fit the model on training data.
8. Evaluate with appropriate metrics:
Regression → MAE/R²
Classification → accuracy_score, ROC-AUC
9. Model Persistence - Serialize the trained model with pickle to wine_model.pkl.
10. Streamlit App Development

---

**TECH STACK**

1. Programming Language : Python

2. Core Libraries : numpy – Numerical computations , pandas – Data manipulation & analysis , Visualization , matplotlib – Basic plotting , seaborn – Heatmaps, pairplots, distribution plots
 , Modeling & Evaluation , scikit-learn , RandomForestRegressor / RandomForestClassifier , train_test_split – Data splitting , accuracy_score, MAE, R², ROC-AUC – Performance metrics

3. Web App : Streamlit – Rapid UI for sliders, buttons, and result display

---
