from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd
import sklearn
import json

app = Flask(__name__)
cors = CORS(app, origins='http://localhost:5173', supports_credentials=True)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json 
    data_class = {
        'KELEMBAPAN': [(float(data['kelembapan']) - 78.235806) / (88.754356 - 78.235806)],
        'TEMPREATUR': [(float(data['temperature']) - 24.066584) / (28.292788 - 24.066584)],
        'SOIL TEMPERATURE': [(float(data['soilTemperature']) - 23.204248) / (29.536764 - 23.204248)],
        'SOIL MOIS': [(float(data['soilMoisture']) - 35.377044) / (43.559612 - 35.377044)],
        'LUAS': [(float(data['luas']) - 4.45) / (2849 - 4.45)]
    }

    data_reg = {
        'CURAH HUJAN': [(float(data['curahHujan']) - 78.235806) / (88.754356 - 78.235806)]
    }

    tanaman = {0: 'JAGUNG', 1: 'KACANG HIJAU', 2: 'KACANG TANAH', 3: 'KEDELAI', 4: 'PADI', 5: 'UBI JALAR', 6: 'UBI KAYU'}

    model_svm = joblib.load('model_ml/SVM.joblib')
    model_lr = joblib.load('model_ml/LR.joblib')

    df_class = pd.DataFrame.from_dict(data_class)
    df_reg = pd.DataFrame.from_dict(data_reg)

    predictions_class = model_svm.predict(df_class)
    df_reg['TANAMAN'] = (predictions_class - 0) / (6 - 0)
    predictions_reg = model_lr.predict(df_reg)
    predictions_reg = json.dumps(predictions_reg[0])
    predictions_trans = tanaman.get(predictions_class[0], 'Tanaman tidak ditemukan')


    response = {
        'tanaman' : predictions_trans,
        'produktivitas' : predictions_reg
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run()
