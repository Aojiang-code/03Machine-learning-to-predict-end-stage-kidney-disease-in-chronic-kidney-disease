# pip install streamlit

# app.py
import streamlit as st
import joblib
import numpy as np
import os

# 加载模型和标准化器
model_path = "/workspace/models/best_model.pkl"
scaler_path = "/workspace/models/scaler.pkl"
features_path = "/workspace/models/features.pkl"

best_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
features = joblib.load(features_path)

# 创建 Streamlit 应用
st.title("CKD Prediction Model")

# 输入数据
data = {}
for feature in features:
    data[feature] = st.number_input(feature)

# 预测按钮
if st.button("Predict"):
    try:
        input_data = np.array([data[feature] for feature in features]).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data)
        prediction = best_model.predict(input_data_scaled)
        result = "ESKD+" if prediction[0] == 1 else "ESKD-"
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(str(e))


# streamlit run app2.py