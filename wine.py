import streamlit as st
import pandas as pd
import pickle

with open("wine_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🍷 Wine Quality Classifier")
st.markdown("""
Adjust the sliders below to describe a wine sample.  
We'll predict the **quality score** and tell you if it's a **Good** or **Bad** wine,
along with confidence scores for each.
""")

fixed_acidity        = st.slider("Fixed Acidity",        4.0, 16.0,  8.0)
volatile_acidity     = st.slider("Volatile Acidity",     0.1,  1.5,  0.5)
citric_acid          = st.slider("Citric Acid",          0.0,  1.0,  0.3)
residual_sugar       = st.slider("Residual Sugar",       0.5, 15.0,  2.5)
chlorides            = st.slider("Chlorides",            0.01, 0.2,  0.05)
free_sulfur_dioxide  = st.slider("Free Sulfur Dioxide",    1,   72,   15)
total_sulfur_dioxide = st.slider("Total Sulfur Dioxide",   6,  289,   46)
density              = st.slider("Density",            0.9900,1.0040,0.9960, step=0.00001)
pH                   = st.slider("pH",                   2.5,  4.5,  3.3)
sulphates            = st.slider("Sulphates",             0.3,  2.0,  0.6)
alcohol              = st.slider("Alcohol %",            8.0, 15.0, 10.0)

dummy_id = 0
input_data = pd.DataFrame(
    [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
      free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates,
      alcohol, dummy_id]],
    columns=[
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
        "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates",
        "alcohol", "Id"
    ]
)

if st.button("Predict"):
    pred_score = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    classes = model.classes_
    good_prob = proba[classes >= 6].sum()
    bad_prob  = proba[classes <  6].sum()
    label = "👍 Good Quality" if pred_score >= 6 else "👎 Bad Quality"

    st.subheader("🍷 Prediction Result")
    st.write(f"**Predicted Quality Score:** `{pred_score}`")
    st.write(f"**Classification:** {label}")
    st.write(f"**Confidence for Good (≥6):** `{good_prob:.2f}`")
    st.write(f"**Confidence for Bad (<6):**  `{bad_prob:.2f}`")


