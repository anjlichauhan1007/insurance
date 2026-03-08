from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
model_path = os.path.join(BASE_DIR, 'loan_prediction_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')

# Load model safely
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Dictionaries
loan_intent_dict = {
    'PERSONAL': 1, 'EDUCATION': 2, 'MEDICAL': 3,
    'VENTURE': 4, 'HOMEIMPROVEMENT': 5, 'DEBTCONSOLIDATION': 6
}

person_gender_dict = {
    'female': 1, 'male': 0
}

person_edu_dict = {
    'Master': 1, 'High School': 2, 'Bachelor': 3, 'Associate': 4, 'Doctorate': 5
}

person_home_ownership_dict = {
    'RENT': 1, 'OWN': 2, 'MORTGAGE': 3, 'OTHER': 4
}

previous_loan_defaults_dict = {
    'No': 0, 'Yes': 1
}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:

        # Form data
        person_age = float(request.form.get('person_age'))
        person_gender = request.form.get('person_gender')
        person_education = request.form.get('person_education')
        person_income = float(request.form.get('person_income'))
        person_emp_exp = int(request.form.get('person_emp_exp'))
        person_home_ownership = request.form.get('person_home_ownership')
        loan_amnt = float(request.form.get('loan_amnt'))
        loan_intent = request.form.get('loan_intent')
        loan_int_rate = float(request.form.get('loan_int_rate'))
        loan_percent_income = float(request.form.get('loan_percent_income'))
        cb_person_cred_hist_length = float(request.form.get('cb_person_cred_hist_length'))
        credit_score = int(request.form.get('credit_score'))
        previous_loan_defaults_on_file = request.form.get('previous_loan_defaults_on_file')

        # Encoding
        person_gender_encoded = person_gender_dict.get(person_gender, 0)
        person_education_encoded = person_edu_dict.get(person_education, 2)
        person_home_ownership_encoded = person_home_ownership_dict.get(person_home_ownership, 1)
        loan_intent_encoded = loan_intent_dict.get(loan_intent, 1)
        previous_defaults_encoded = previous_loan_defaults_dict.get(previous_loan_defaults_on_file, 0)

        # Feature array
        features = np.array([[
            person_age,
            person_gender_encoded,
            person_education_encoded,
            person_income,
            person_emp_exp,
            person_home_ownership_encoded,
            loan_amnt,
            loan_intent_encoded,
            loan_int_rate,
            loan_percent_income,
            cb_person_cred_hist_length,
            credit_score,
            previous_defaults_encoded
        ]])

        # Scaling
        features_scaled = scaler.transform(features)

        # Prediction
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)

        result = "Approved" if prediction[0] == 1 else "Rejected"
        confidence = float(probability[0][prediction[0]] * 100)

        return render_template(
            'result.html',
            prediction=result,
            confidence=f"{confidence:.2f}",
            person_age=person_age,
            person_gender=person_gender,
            person_education=person_education,
            person_income=person_income,
            loan_amnt=loan_amnt,
            loan_intent=loan_intent,
            loan_percent_income=loan_percent_income
        )

    except Exception as e:
        return render_template('error.html', error=str(e))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:

        data = request.get_json()

        person_age = float(data['person_age'])
        person_gender = data['person_gender']
        person_education = data['person_education']
        person_income = float(data['person_income'])
        person_emp_exp = int(data['person_emp_exp'])
        person_home_ownership = data['person_home_ownership']
        loan_amnt = float(data['loan_amnt'])
        loan_intent = data['loan_intent']
        loan_int_rate = float(data['loan_int_rate'])
        loan_percent_income = float(data['loan_percent_income'])
        cb_person_cred_hist_length = float(data['cb_person_cred_hist_length'])
        credit_score = int(data['credit_score'])
        previous_loan_defaults_on_file = data['previous_loan_defaults_on_file']

        # Encoding
        person_gender_encoded = person_gender_dict.get(person_gender, 0)
        person_education_encoded = person_edu_dict.get(person_education, 2)
        person_home_ownership_encoded = person_home_ownership_dict.get(person_home_ownership, 1)
        loan_intent_encoded = loan_intent_dict.get(loan_intent, 1)
        previous_defaults_encoded = previous_loan_defaults_dict.get(previous_loan_defaults_on_file, 0)

        features = np.array([[
            person_age,
            person_gender_encoded,
            person_education_encoded,
            person_income,
            person_emp_exp,
            person_home_ownership_encoded,
            loan_amnt,
            loan_intent_encoded,
            loan_int_rate,
            loan_percent_income,
            cb_person_cred_hist_length,
            credit_score,
            previous_defaults_encoded
        ]])

        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)

        result = "Approved" if prediction[0] == 1 else "Rejected"
        confidence = float(probability[0][prediction[0]] * 100)

        return jsonify({
            "prediction": result,
            "confidence": confidence,
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        })


if __name__ == "__main__":
    app.run(debug=True)