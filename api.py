# pip install fastapi uvicorn
# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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

# 创建 FastAPI 应用
app = FastAPI()

# 定义输入数据的结构
class PredictionRequest(BaseModel):
    data: dict

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # 提取输入数据
        input_data = np.array([request.data[feature] for feature in features]).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data)
        prediction = best_model.predict(input_data_scaled)
        return {"prediction": "ESKD+" if prediction[0] == 1 else "ESKD-"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# uvicorn api:app --reload