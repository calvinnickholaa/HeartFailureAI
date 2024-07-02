from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import logging

app = Flask(__name__, template_folder='views')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Load the model and preprocessing objects
try:
    with open('model/heartFailure.pkl', 'rb') as file:
        loaded_objects = pickle.load(file)
        scaler = loaded_objects['scaler']
        rfe = loaded_objects['rfe']
        model = loaded_objects['grid_search']
except Exception as e:
    logger.error(f"Error loading model: {e}")
    scaler = rfe = model = None

@app.route('/submit', methods=['POST'])
def submit_form():
    try:
        if model is None or scaler is None or rfe is None:
            raise ValueError("Model or preprocessing objects are not loaded correctly")

        # Get JSON data from the request
        data = request.get_json()

        # Extract form data
        age = data.get('age')
        anaemia = data.get('anaemia')
        creatinine_phosphokinase = data.get('creatinine_phosphokinase')
        diabetes = data.get('diabetes')
        ejection_fraction = data.get('ejection_fraction')
        high_blood_pressure = data.get('high_blood_pressure')
        platelets = data.get('platelets')
        serum_creatinine = data.get('serum_creatinine')
        serum_sodium = data.get('serum_sodium')
        sex = data.get('sex')
        smoking = data.get('smoking')
        time = data.get('time')

        # Ensure no values are None
        if None in [age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time, ]:
            raise ValueError("One or more form fields are empty")

        # Convert data to appropriate types and validate ranges
        try:
            data_list = [
                float(age),
                int(anaemia),
                int(creatinine_phosphokinase),
                int(diabetes),
                int(ejection_fraction),
                int(high_blood_pressure),
                float(platelets),
                float(serum_creatinine),
                int(serum_sodium),
                int(sex),
                int(smoking),
                int(time),
            ]
        except ValueError as ve:
            raise ValueError("Invalid input: all values must be numeric")

        # Ensure the feature names are in the correct order
        feature_names = [
            'age',
            'anaemia',
            'creatinine_phosphokinase',
            'diabetes',
            'ejection_fraction',
            'high_blood_pressure',
            'platelets',
            'serum_creatinine',
            'serum_sodium',
            'sex',
            'smoking',
            'time',
        ]

        # Create DataFrame with appropriate feature names
        data_df = pd.DataFrame([data_list], columns=feature_names)
        logger.info(f"Data DataFrame: {data_df}")

        # Scale the data
        data_scaled = scaler.transform(data_df)
        logger.info(f"Data scaled: {data_scaled}")

        # Apply RFE to the scaled data
        data_selected = rfe.transform(data_scaled)
        logger.info(f"Data selected with RFE: {data_selected}")

        # Make prediction
        prediction = model.predict(data_selected)
        logger.info(f"Prediction result: {prediction}")

        # Convert prediction to native Python type
        result = int(prediction[0])
        return jsonify(result=result)
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        return jsonify(result=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
