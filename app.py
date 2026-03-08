from flask import Flask, request, render_template
import joblib
import numpy as np
import os

# locate base directory (project6)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# configure Flask to use the shared templates folder
template_dir = os.path.join(BASE_DIR, 'templates')
app = Flask(__name__, template_folder=template_dir)

# load the model using absolute path
model_path = os.path.join(BASE_DIR, 'cost-predictor')
model = joblib.load(model_path)

# Mappings
gender_dict = {'female': 0, 'male': 1}
region_dict = {'southwest': 1, 'southeast': 2, 'northwest': 3, 'northeast': 4}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    age = float(request.form['age'])
    gender = request.form['gender']
    bmi = float(request.form['bmi'])
    region = request.form['region']
    
    # Convert to numbers
    gender_num = gender_dict[gender]
    region_num = region_dict[region]
    
    # Prepare input
    features = np.array([[age, gender_num, bmi, region_num]])
    
    # Predict
    prediction = model.predict(features)[0]
    
    return render_template('index.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)