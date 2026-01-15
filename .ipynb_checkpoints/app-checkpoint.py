import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_model():
    return joblib.load('ckd_model_package.pkl')

model_data = load_model()

st.title("🏥 CKD & Diabetes Risk Assessment")

# 初始化 session state
if 'ckd_result' not in st.session_state:
    st.session_state.ckd_result = None
if 'diabetes_result' not in st.session_state:
    st.session_state.diabetes_result = None

# ==========================================
# CKD Risk Assessment
# ==========================================
st.header("1️⃣ Chronic Kidney Disease (CKD) Risk")

col1, col2, col3 = st.columns(3)

with col1:
    sg = st.number_input("Specific Gravity (sg)", 1.005, 1.030, 1.020, 0.005)
    al = st.selectbox("Albumin (al)", [0, 1, 2, 3, 4, 5])
    bgr = st.number_input("Blood Glucose Random (bgr) mg/dL", 50, 500, 120)
    bu = st.number_input("Blood Urea (bu) mg/dL", 10, 200, 40)

with col2:
    sc = st.number_input("Serum Creatinine (sc) mg/dL", 0.5, 15.0, 1.2, 0.1)
    sod = st.number_input("Sodium (sod) mEq/L", 100, 160, 140)
    hemo = st.number_input("Hemoglobin (hemo) g/dL", 3.0, 18.0, 12.0, 0.1)
    pcv = st.number_input("Packed Cell Volume (pcv) %", 10, 60, 40)

with col3:
    rbcc = st.number_input("Red Blood Cell Count (rbcc) millions/cmm", 2.0, 8.0, 4.5, 0.1)
    htn = st.selectbox("Hypertension (htn)", ["No", "Yes"])
    dm = st.selectbox("Diabetes Mellitus (dm)", ["No", "Yes"])
    pe = st.selectbox("Pedal Edema (pe)", ["No", "Yes"])

if st.button("🔍 Assess CKD Risk"):
    input_data = pd.DataFrame([[
        sg, al, bgr, bu, sc, sod, hemo, pcv, rbcc,
        1 if htn == "Yes" else 0,
        1 if dm == "Yes" else 0,
        1 if pe == "Yes" else 0
    ]], columns=['sg', 'al', 'bgr', 'bu', 'sc', 'sod', 'hemo', 'pcv', 'rbcc', 'htn', 'dm', 'pe'])
    
    scaled = model_data['scaler'].transform(input_data)
    prob = model_data['model'].predict_proba(scaled)[0][1]
    st.session_state.ckd_result = prob

# 显示 CKD 结果
if st.session_state.ckd_result is not None:
    prob = st.session_state.ckd_result
    st.subheader("CKD Result")
    if prob >= 0.7:
        st.error(f"⚠️ High Risk: {prob*100:.1f}%")
    elif prob >= 0.3:
        st.warning(f"⚡ Moderate Risk: {prob*100:.1f}%")
    else:
        st.success(f"✅ Low Risk: {prob*100:.1f}%")
    st.progress(prob)

st.divider()

# ==========================================
# Diabetes Risk Assessment
# ==========================================
st.header("2️⃣ Diabetes Risk Assessment")

col_a, col_b = st.columns(2)
with col_a:
    fasting_glucose = st.number_input("Fasting Glucose (mmol/L)", 3.0, 20.0, 5.0, 0.1)
with col_b:
    fasting_insulin = st.number_input("Fasting Insulin (μU/mL)", 1.0, 100.0, 10.0, 0.1)

if st.button("🔍 Assess Diabetes Risk"):
    homa_ir = (fasting_glucose * fasting_insulin) / 22.5
    st.session_state.diabetes_result = {'glucose': fasting_glucose, 'homa_ir': homa_ir}

# 显示糖尿病结果
if st.session_state.diabetes_result is not None:
    res = st.session_state.diabetes_result
    st.subheader("Diabetes Diagnosis Result")
    st.write(f"**Your Fasting Glucose:** {res['glucose']} mmol/L")
    
    if res['glucose'] >= 7:
        st.error("⚠️ **Diagnosis: Diabetes**")
    elif res['glucose'] >= 5.6:
        st.warning("⚡ **Diagnosis: Prediabetes**")
    else:
        st.success("✅ **Diagnosis: Normal**")
    
    st.metric("HOMA-IR Index", f"{res['homa_ir']:.2f}")
    if res['homa_ir'] > 2.5:
        st.warning("HOMA-IR > 2.5 suggests insulin resistance.")
