from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 从环境变量中获取模型文件路径，默认路径为 '/workspace/models/'
model_path = os.getenv('MODEL_PATH', '/workspace/models/best_model.pkl')
scaler_path = os.getenv('SCALER_PATH', '/workspace/models/scaler.pkl')
features_path = os.getenv('FEATURES_PATH', '/workspace/models/features.pkl')

# 加载模型、标准化器和特征列
try:
    best_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    features = joblib.load(features_path)
    logger.info(f"Models loaded successfully from {model_path}, {scaler_path}, {features_path}")
except Exception as e:
    logger.error(f"Error loading model or scaler files: {str(e)}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取输入数据
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # 检查是否有缺失特征
        input_data = np.array([data.get(feature) for feature in features])
        if None in input_data:
            return jsonify({"error": "Missing data for one or more features"}), 400

        # 预处理数据
        input_data = input_data.reshape(1, -1)
        input_data_scaled = scaler.transform(input_data)

        # 进行预测
        prediction = best_model.predict(input_data_scaled)
        result = "ESKD+" if prediction[0] == 1 else "ESKD-"

        # 记录请求和结果
        logger.info(f"Received input data: {data}")
        logger.info(f"Prediction result: {result}")

        return jsonify({"prediction": result})

    except Exception as e:
        # 捕获异常并记录错误
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=5001)

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')

