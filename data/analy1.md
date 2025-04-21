### 第一部分：数据准备
```python
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# 加载数据
data_path = "/workspace/data/simulated_ckd_data_with_missing_and_imputation.csv"
data = pd.read_csv(data_path)

# 分离特征和目标变量
X = data.drop(columns=["ESKD Status"])
y = data["ESKD Status"].map({"ESKD+": 1, "ESKD-": 0})

# 分离连续变量和分类变量
continuous_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

# 处理缺失值：多重插补
imputer = IterativeImputer(random_state=42, max_iter=10, n_nearest_features=5)
X_imputed = pd.DataFrame(imputer.fit_transform(X[continuous_features]), columns=continuous_features)

# 编码分类变量：独热编码
encoder = OneHotEncoder(handle_unknown="ignore")
X_encoded = encoder.fit_transform(X[categorical_features]).toarray()
encoded_feature_names = encoder.get_feature_names_out(categorical_features)
X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_feature_names)

# 合并处理后的数据
X_prepared = pd.concat([X_imputed, X_encoded_df], axis=1)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_prepared)

# 数据分割：训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
```

### 第二部分：数据预处理
```python
from sklearn.model_selection import StratifiedKFold

# 五折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### 第三部分：模型开发

超参数选择（来自文献原文）

| Algorithms         | Hyperparameters                                                                 |
|--------------------|---------------------------------------------------------------------------------|
| Logistic regression | penalty='l2', class_weight='balanced', max_iter=100000, C=10, solver='liblinear' |
| Naive Bayes        | type='multinomial', alpha=150                                                   |
| Decision tree      | criterion='gini', splitter='best', max_depth=16, max_features=15, min_samples_leaf=5, min_samples_split=0.0001 |
| Random forest      | class_weight='balanced', criterion='gini', max_depth=9, max_features=17, min_samples_leaf=6, min_samples_split=30, n_estimators=32 |
| K-nearest neighbors| weights='distance', metric='minkowski', n_neighbors=16, leaf_size=10              |

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# 定义模型和参数网格
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

param_grids = {
    "Logistic Regression": {"C": [0.1, 1, 10]},
    "Naive Bayes": {},
    "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
    "Decision Tree": {"max_depth": [None, 5, 10]},
    "K-Nearest Neighbors": {"n_neighbors": [3, 5, 7]}
}

# 模型训练和超参数优化
best_models = {}
for name, model in models.items():
    print(f"Training {name}...")
    grid_search = GridSearchCV(model, param_grids[name], cv=cv, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    print(f"Best parameters for {name}: {grid_search.best_params_}")
```

### 第四部分：性能评估
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

# 定义评估函数
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    return accuracy, precision, recall, f1, roc_auc

# 评估每个模型
results = {}
for name, model in best_models.items():
    accuracy, precision, recall, f1, roc_auc = evaluate_model(model, X_test, y_test)
    results[name] = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1, "ROC AUC": roc_auc}
    print(f"{name} results: {results[name]}")

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
for name, model in best_models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()
```

### 第五部分：结果分析
```python
# 性能比较
best_model_name = max(results, key=lambda k: results[k]["F1"])
best_model = best_models[best_model_name]
print(f"Best model: {best_model_name}")

# 敏感性分析
thresholds = np.arange(0.1, 1, 0.1)
sensitivity = []
specificity = []
for threshold in thresholds:
    y_pred = (best_model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    tp = np.sum((y_test == 1) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    sensitivity.append(tp / (tp + fn))
    specificity.append(tn / (tn + fp))

plt.figure(figsize=(8, 6))
plt.plot(thresholds, sensitivity, label="Sensitivity")
plt.plot(thresholds, specificity, label="Specificity")
plt.xlabel("Threshold")
plt.ylabel("Rate")
plt.title("Sensitivity and Specificity Analysis")
plt.legend()
plt.show()
```

### 第六部分：外部验证
```python
# 加载外部数据集
external_data_path = "/path/to/external_data.csv"
external_data = pd.read_csv(external_data_path)

# 应用相同的预处理步骤
external_X = external_data.drop(columns=["ESKD Status"])
external_y = external_data["ESKD Status"].map({"ESKD+": 1, "ESKD-": 0})

# 处理缺失值、编码和标准化
external_X_imputed = pd.DataFrame(imputer.transform(external_X[continuous_features]), columns=continuous_features)
external_X_encoded = encoder.transform(external_X[categorical_features]).toarray()
external_X_encoded_df = pd.DataFrame(external_X_encoded, columns=encoded_feature_names)
external_X_prepared = pd.concat([external_X_imputed, external_X_encoded_df], axis=1)
external_X_scaled = scaler.transform(external_X_prepared)

# 在外部数据集上验证模型
external_accuracy, external_precision, external_recall, external_f1, external_roc_auc = evaluate_model(best_model, external_X_scaled, external_y)
print(f"External validation results: Accuracy={external_accuracy:.2f}, Precision={external_precision:.2f}, Recall={external_recall:.2f}, F1={external_f1:.2f}, ROC AUC={external_roc_auc:.2f}")
```

### 第七部分：模型改进
```python
# 纳入更多特征
# 假设新增了尿液检测指标ACR
data["ACR"] = np.random.normal(loc=10, scale=5, size=len(data))  # 示例数据
continuous_features.append("ACR")

# 重新处理数据
X_improved = data.drop(columns=["ESKD Status"])
X_improved_imputed = pd.DataFrame(imputer.fit_transform(X_improved[continuous_features]), columns=continuous_features)
X_improved_encoded = encoder.fit_transform(X_improved[categorical_features]).toarray()
X_improved_encoded_df = pd.DataFrame(X_improved_encoded, columns=encoded_feature_names)
X_improved_prepared = pd.concat([X_improved_imputed, X_improved_encoded_df], axis=1)
X_improved_scaled = scaler.fit_transform(X_improved_prepared)

# 重新训练模型
X_train_improved, X_test_improved, y_train_improved, y_test_improved = train_test_split(X_improved_scaled, y, test_size=0.2, stratify=y, random_state=42)
best_models_improved = {}
for name, model in models.items():
    print(f"Training improved {name}...")
    grid_search = GridSearchCV(model, param_grids[name], cv=cv, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train_improved, y_train_improved)
    best_models_improved[name] = grid_search.best_estimator_
    print(f"Best parameters for improved {name}: {grid_search.best_params_}")

# 评估改进后的模型
results_improved = {}
for name, model in best_models_improved.items():
    accuracy, precision, recall, f1, roc_auc = evaluate_model(model, X_test_improved, y_test_improved)
    results_improved[name] = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1, "ROC AUC": roc_auc}
    print(f"Improved {name} results: {results_improved[name]}")
```

#### 保存最佳模型参数
```python
import joblib
import os

# 确保保存模型的目录存在
model_dir = "/workspace/models/"
os.makedirs(model_dir, exist_ok=True)

# 保存模型
model_path = os.path.join(model_dir, "best_model.pkl")
joblib.dump(best_model, model_path)
print(f"Model saved to {model_path}")
```

### 第八部分：临床应用
```python
# 模型部署
# 假设使用Flask框架部署模型
# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# 加载模型、标准化器和特征列
model_path = "/workspace/models/best_model.pkl"
scaler_path = "/workspace/models/scaler.pkl"
features_path = "/workspace/models/features.pkl"

best_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
features = joblib.load(features_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = np.array([data[feature] for feature in features]).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    prediction = best_model.predict(input_data_scaled)
    return jsonify({"prediction": "ESKD+" if prediction[0] == 1 else "ESKD-"})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
```

### 总结
以上代码涵盖了从数据准备到模型部署的完整流程。您可以根据实际情况调整代码，例如添加更多特征、尝试其他算法或优化模型部署。希望这些代码对您的项目有所帮助！