from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model dan tools
knn_model = joblib.load('model_knn.pkl')
gnb_model = joblib.load('model_gnb.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Daftar fitur sesuai urutan training
feature_names = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
                 'Oldpeak', 'ST_Slope']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = []
    for feature in feature_names:
        value = request.form.get(feature)
        if feature in label_encoders:
            value = label_encoders[feature].transform([value])[0]
        else:
            value = float(value)
        input_data.append(value)

    # Scaling numerik
    input_array = np.array([input_data])
    df_input = pd.DataFrame(input_array, columns=feature_names)
    df_input[scaler.feature_names_in_] = scaler.transform(df_input[scaler.feature_names_in_])

    knn_pred = knn_model.predict(df_input)[0]
    gnb_pred = gnb_model.predict(df_input)[0]

    return render_template('index.html',
                           knn_result="Berisiko" if knn_pred == 1 else "Tidak Berisiko",
                           gnb_result="Berisiko" if gnb_pred == 1 else "Tidak Berisiko")

if __name__ == '__main__':
    app.run(debug=True)
