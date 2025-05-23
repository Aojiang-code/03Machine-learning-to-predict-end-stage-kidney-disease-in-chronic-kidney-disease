# pip install scipy

import numpy as np
import pandas as pd
from scipy.stats import norm, uniform
import os

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 模拟数据的样本量
n_samples = 748

# 创建保存路径
save_path = r"/workspace/data"
os.makedirs(save_path, exist_ok=True)

# 模拟连续变量
continuous_data = {
    "Age": norm.rvs(loc=57.8, scale=17.6, size=n_samples),
    "Systolic Blood Pressure (SBP)": norm.rvs(loc=129.5, scale=17.8, size=n_samples),
    "Diastolic Blood Pressure (DBP)": norm.rvs(loc=77.7, scale=11.1, size=n_samples),
    "BMI": norm.rvs(loc=24.8, scale=3.7, size=n_samples),
    "Total Protein": norm.rvs(loc=71.6, scale=8.4, size=n_samples),
    "Albumin": norm.rvs(loc=42.2, scale=5.6, size=n_samples),
    "Calcium": norm.rvs(loc=2.2, scale=0.1, size=n_samples),
    "Phosphorous": norm.rvs(loc=1.2, scale=0.2, size=n_samples),
    "Calcium-Phosphorus Product (Ca x P)": norm.rvs(loc=33.5, scale=5.6, size=n_samples),
    "Blood Leukocyte Count": norm.rvs(loc=7.1, scale=2.4, size=n_samples),
    "Hemoglobin": norm.rvs(loc=131.0, scale=20.3, size=n_samples),
    "Platelet Count": norm.rvs(loc=209.8, scale=57.1, size=n_samples),
    "Potassium": norm.rvs(loc=4.3, scale=0.5, size=n_samples),
    "Sodium": norm.rvs(loc=140.2, scale=2.8, size=n_samples),
    "Chlorine": norm.rvs(loc=106.9, scale=3.7, size=n_samples),
    "Bicarbonate": norm.rvs(loc=25.9, scale=3.6, size=n_samples)
}

# 模拟分类变量
categorical_data = {
    "Gender": np.random.choice(["Male", "Female"], size=n_samples, p=[419/748, 329/748]),
    "Primary Disease": np.random.choice(
        ["Primary GN", "Diabetes", "Hypertension", "CIN", "Others", "Unknown"],
        size=n_samples,
        p=[292/748, 224/748, 97/748, 64/748, 18/748, 53/748]
    ),
    "Hypertension History": np.random.choice(["Yes", "No"], size=n_samples, p=[558/748, 1 - 558/748]),
    "Diabetes Mellitus History": np.random.choice(["Yes", "No"], size=n_samples, p=[415/748, 1 - 415/748]),
    "Cardiovascular or Cerebrovascular Disease History": np.random.choice(["Yes", "No"], size=n_samples, p=[177/748, 1 - 177/748]),
    "Smoking History": np.random.choice(["Yes", "No"], size=n_samples, p=[91/748, 1 - 91/748])
}

# 模拟肾病分期
ckd_stages = np.random.choice(
    ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"],
    size=n_samples,
    p=[58/748, 183/748, 352/748, 119/748, 36/748]
)

# 模拟目标变量
eskd_status = np.random.choice(["ESKD+", "ESKD-"], size=n_samples, p=[70/748, 1 - 70/748])

# 合并所有数据
simulated_data = pd.DataFrame({**continuous_data, **categorical_data, "CKD Stage": ckd_stages, "ESKD Status": eskd_status})

# 保存模拟数据
file_path = os.path.join(save_path, "ckd_data.csv")
simulated_data.to_csv(file_path, index=False)
print(f"Simulated data saved to {file_path}")