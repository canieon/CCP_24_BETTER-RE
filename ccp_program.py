# 데이터 분석 및 모델링 라이브러리
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 회귀 모델
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor

# 군집화
from sklearn.cluster import KMeans

# Flask 웹 앱
from flask import Flask, render_template_string, request, jsonify
import webbrowser
from threading import Timer

#pip install pandas
#pip install numpy
#pip install scikit-learn
#pip install xgboost
#pip install flask

RANDOM_STATE = 110

# 데이터 불러오기
data = pd.read_csv("C:/Users/canieon/Downloads/1231_5th.csv")#파일 위치
data_1 = data
data_2 = pd.read_csv("C:/Users/canieon/Downloads/result_ccp.csv")#파일 위치




data = data.dropna(how='any')

# 필요한 칼럼 선택
features = ['Min Voltage (V)', 'Max Voltage (V)', 'Charge Capacity (Ah)',
            'Discharge Capacity (Ah)', 'Charge Energy (Wh)', 'Discharge Energy (Wh)',
            'Charge-Discharge Efficiency (%)', 'Energy Efficiency (%)','SOH (%)']
target = 'Cycle'

# 특성과 타겟 분리
X = data[features].values
y = data[target].values

# 학습 데이터 분리리
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# 1. GBM 모델 생성 및 학습
gbm = GradientBoostingRegressor(
    n_estimators=326,          # 트리 개수
    learning_rate=0.1279293174000281,        # 학습률
    max_depth=8,                # 트리의 최대 깊이
    min_samples_split=3,# 분할 노드의 최소 샘플 수
    min_samples_leaf=8,  # 리프 노드의 최소 샘플 수
    subsample=0.8010548372,                # 각 트리에 사용할 샘플 비율
    random_state=RANDOM_STATE
)

gbm.fit(X_train, y_train)
y_pred_gbm = gbm.predict(X_test)

# 2. XGBoost 모델 생성 및 학습
xgb = XGBRegressor(
    n_estimators=500,    # 트리 개수
    learning_rate=0.1,  # 학습률
    max_depth=7,          # 트리의 최대 깊이
    random_state=RANDOM_STATE
)

xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
ensemble_weight_gbm = 0.56
ensemble_weight_xgb = 0.44

# SOH 계산 함수
def calculate_soh(discharge_capacity, fresh_capacity, battery_id=None):
    try:
        discharge_capacity = pd.to_numeric(discharge_capacity, errors='coerce')
        fresh_capacity = pd.to_numeric(fresh_capacity, errors='coerce')
        if pd.isna(discharge_capacity) or pd.isna(fresh_capacity) or fresh_capacity == 0:
            return np.nan
        return (discharge_capacity / fresh_capacity) * 100
    except Exception as e:
        print(f"Error in SOH calculation for battery {battery_id}: {e}")
        return np.nan


# 충방전 효율 계산 함수
def calculate_charge_discharge_efficiency(discharge_capacity, charge_capacity):
    discharge_capacity = pd.to_numeric(discharge_capacity, errors='coerce')
    charge_capacity = pd.to_numeric(charge_capacity, errors='coerce')
    if charge_capacity > 0:
        return (discharge_capacity / charge_capacity) * 100
    else:
        return 0

# 에너지 효율 계산 함수
def calculate_energy_efficiency(discharge_energy, charge_energy):
    discharge_energy = pd.to_numeric(discharge_energy, errors='coerce')
    charge_energy = pd.to_numeric(charge_energy, errors='coerce')
    if charge_energy > 0:
        return (discharge_energy / charge_energy) * 100
    else:
        return 0
  
connect_model_data = pd.merge(data_1, data_2, how='outer', on='Battery ID')
# 예측 대상 칼럼 ('at 80'이 붙은 칼럼)
target_columns = [col for col in connect_model_data.columns if col.endswith('at 80')]

# 입력 데이터 칼럼 (제외할 칼럼 제외)
exclude_columns = ['Battery ID', 'Internal Resistance (Ohms)'] + target_columns
feature_columns = [col for col in connect_model_data.columns if col not in exclude_columns]

# 데이터셋 분할
X = connect_model_data[feature_columns]  # 입력 데이터
y = connect_model_data[target_columns]  # 출력 데이터

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 예측 및 평가
y_pred_rf = rf_model.predict(X_test)

def predict_at_80(model, new_record):
    input_df = pd.DataFrame([new_record])
    prediction = model.predict(input_df)
    prediction_df = pd.DataFrame(prediction, columns=target_columns)
    return prediction_df

# 숫자형 데이터만 선택 및 스케일링
numerical_data = data_2.select_dtypes(include=['float64', 'int64'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# KMeans 군집화
best_k = 3  # 최적의 k 값
kmeans = KMeans(n_clusters=best_k, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
data_2['Cluster'] = clusters

# Flask 앱 생성
app = Flask(__name__)

# HTML 템플릿 문자열
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BETTER:RE</title>
    <style>
        .input-container {
            font-family: Arial, sans-serif;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            max-width: 400px;
            margin: 20px auto;
            background-color: #f9f9f9;
        }
        .input-container label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #333;
        }
        .input-container input {
            width: calc(100% - 20px);
            padding: 10px;
            margin-bottom: 15px;
            border: 1px solid #ccc;
            border-radius: 5px;
            font-size: 14px;
        }
        .input-container button {
            padding: 10px 15px;
            font-size: 16px;
            color: #fff;
            background-color: #2b5931;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .input-container button:hover {
            background-color: #2b5931;
        }
        .result-container {
            font-family: Arial, sans-serif;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            max-width: 400px;
            margin: 20px auto;
            background-color: #e6f4ea;
            text-align: center;
            color: #2b5931;
        }
        .result-container h3 {
            margin: 10px 0;
            font-size: 20px;
            font-weight: bold;
        }
        .result-container p {
            font-size: 16px;
            color: #555;
            margin: 5px 0;
        }
        .info-container {
            font-family: Arial, sans-serif;
            border: 1px solid #bbb;
            border-radius: 8px;
            padding: 20px;
            max-width: 400px;
            margin: 10px auto;
            background-color: #c8e6c9;
            color: #1b5e20;
        }
        .info-container h4 {
            margin-top: 0;
            font-size: 18px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="input-container">
        <form id="batteryForm">
            <label>Min Voltage (V): </label>
            <input name="var0" type="number" step="any" placeholder="Input Min Voltage" required>

            <label>Max Voltage (V): </label>
            <input name="var1" type="number" step="any" placeholder="Input Max Voltage" required>

            <label>Charge Capacity (Ah): </label>
            <input name="var2" type="number" step="any" placeholder="Input Charge Capacity" required>

            <label>Discharge Capacity (Ah): </label>
            <input name="var3" type="number" step="any" placeholder="Input Discharge Capacity" required>

            <label>Charge Energy (Wh): </label>
            <input name="var4" type="number" step="any" placeholder="Input Charge Energy" required>

            <label>Discharge Energy (Wh): </label>
            <input name="var5" type="number" step="any" placeholder="Input Discharge Energy" required>

            <label>Fresh Capacity (Ah): </label>
            <input name="var6" type="number" step="any" placeholder="Input Fresh Capacity" required>

            <button type="submit">Enter</button>
        </form>
    </div>

    <div id="result" class="result-container" style="display: none;">
        <h3>🔋 Battery Cycle Count</h3>
        <p>Your battery has completed approximately:</p>
        <h3 style="color: #007bff;" id="cycleCount"></h3>
        <p><strong>State of Health (SOH):</strong> <span id="soh"></span>%</p>
        <p><strong>Charge-Discharge Efficiency:</strong> <span id="cde"></span>%</p>
        <p><strong>Energy Efficiency:</strong> <span id="ee"></span>%</p>
    </div>
        <div id="info" class="info-container" style="display: none;">
        <h3><span style="font-weight: bold;">🔋 Your Battery Type:</span> <span style="font-weight: normal;" id="cluster"></span></h3>
        <p><span style="font-weight: bold;">Description:</span> <span style="font-weight: normal;" id="description"></span></p>
        <h3><span style="font-weight: bold;">Use Cases:<br></span> <span style="font-weight: normal;" id="case"></span></h3>
    </div>

    <script>
        document.getElementById('batteryForm').addEventListener('submit', async function (event) {
            event.preventDefault();

            const formData = new FormData(event.target);
            const data = Object.fromEntries(formData.entries());

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            document.getElementById('cycleCount').innerText = result.cycle_count;
            document.getElementById('soh').innerText = result.soh.toFixed(2);
            document.getElementById('cde').innerText = result.cde.toFixed(2);
            document.getElementById('ee').innerText = result.ee.toFixed(2);
            document.getElementById('cluster').innerText = result.cluster;
            document.getElementById('description').innerText = result.description;
            document.getElementById('case').innerHTML = result.case;
            document.getElementById('result').style.display = 'block';
            document.getElementById('info').style.display = 'block';
        });
    </script>
</body>
</html>
"""

# Flask 라우트 설정
@app.route('/')
def index():
    return render_template_string(html_template)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    var0, var1, var2, var3, var4, var5, var6 = map(float, [
        data['var0'], data['var1'], data['var2'], data['var3'], data['var4'], data['var5'], data['var6']
    ])

    soh = calculate_soh(var3, var6)
    cde = calculate_charge_discharge_efficiency(var3, var2)
    ee = calculate_energy_efficiency(var5, var4)

    y_pred = (
        ensemble_weight_gbm * gbm.predict(np.array([[var0, var1, var2, var3, var4, var5, soh, cde, ee]]))[0] +
        ensemble_weight_xgb * xgb.predict(np.array([[var0, var1, var2, var3, var4, var5, soh, cde, ee]]))[0]
    )
    cycle_count = max(0, int(y_pred + 0.5))
    new_record = {
        'Cycle': y_pred,
        'Min Voltage (V)': var0,
        'Max Voltage (V)': var1,
        'Charge Capacity (Ah)': var2,
        'Discharge Capacity (Ah)': var3,
        'Charge Energy (Wh)': var4,
        'Discharge Energy (Wh)': var5,
        'SOH (%)': soh,
        'Charge-Discharge Efficiency (%)': cde,
        'Energy Efficiency (%)': ee
    }

    # Random Forest로 예측
    predicted_rf = predict_at_80(rf_model, new_record)
    DE80 = float(predicted_rf['Discharge Energy at 80'])
    EE80 = float(predicted_rf['energy efficiency at 80'])
    

    # 새 입력 데이터 예측
    input_data = [[DE80, EE80]]
    scaled_input_data = scaler.transform(input_data) 
    predicted_cluster = kmeans.predict(scaled_input_data)[0]
    dict_name = {0:'Efficient Battery', 1:'Balanced Battery', 2:'High-Capacity Battery'}
    dict_case = {0: '''Portable electronics (smartphones, tablets)
IoT devices (smart home sensors, wearable devices)
Electric bicycles (to maximize mileage per charge)
Healthcare equipment (portable diagnostic tools)
Renewable energy storage systems (home solar battery packs)
''', 1:'''Power banks and backup batteries
Auxiliary power units for EV charging stations (short-term power backup)
Low-power devices (wireless speakers, small portable tools)
Emergency lighting and UPS systems
''', 2:'''Large-scale energy storage systems (ESS) (for homes, industries, and grids)
Power grid support systems (stabilizing grid demand during peak hours)
EV charging station buffers (to handle peak power demands)
Backup power supplies (for critical infrastructure)
Off-grid renewable energy systems (for remote or large-scale applications)'''}
    dict_desc = {0:f'your battery has high energy efficiency (about {EE80:.2f}% at 80% SOH) and good discharge energy (about {DE80:.2f} Wh at 80% SOH), making it suitable for environments where energy management is critical.',1:f'your battery has a moderate energy efficiency (about {EE80:.2f}% at 80% SOH) and a lower discharge energy (about {DE80:.2f} Wh at 80% SOH). This battery is designed for balanced performance, suitable for low-power systems where efficiency is important but capacity is less critical.', 2:f'Your battery has high discharge energy (about {DE80:.2f} Wh at 80% SOH) but lower energy efficiency (about {EE80:.2f}% at 80% SOH). This battery is optimal for applications requiring long-lasting power rather than maximizing energy usage efficiency.'}
    return jsonify({
        'cycle_count': cycle_count,
        'soh': min(100, soh),
        'cde': min(100, cde),
        'ee': min(100, ee),
        'cluster': str(dict_name[predicted_cluster]),
        'case': str(dict_case[predicted_cluster]).replace('\n', '<br>'),
        'description': str(dict_desc[predicted_cluster])
    })


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    Timer(1, open_browser).start()  # 1초 후 브라우저 자동 실행
    app.run(debug=False)