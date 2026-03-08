from flask import Flask, render_template, request
import joblib
import numpy as np

# ==========================
# LOAD MODEL
# ==========================

model = joblib.load('linear_model.lib')

app = Flask(__name__)

# ==========================
# HOME PAGE
# ==========================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def project():
    return render_template('project.html')


@app.route('/predict_result', methods=['POST'])
def predict():

    brand_name = request.form.get('brand_name')
    owner = int(request.form.get('owner'))
    age = int(request.form.get('age'))
    power = int(request.form.get('power'))
    kms_driven = int(request.form.get('kms_driven'))

    brand_dict = {
        'TVS':1,'Royal Enfield':2,'Triumph':3,'Yamaha':4,
        'Honda':5,'Hero':6,'Bajaj':7,'Suzuki':8,
        'Benelli':9,'KTM':10,'Mahindra':11,'Kawasaki':12,
        'Ducati':13,'Hyosung':14,'Harley-Davidson':15,'Jawa':16,
        'BMW':17,'Indian':18,'Rajdoot':19,'LML':20,
        'Yezdi':21,'MV':22,'Ideal':23
    }

    brand_name = brand_dict[brand_name]

    labels = [[brand_name, owner, age, power, kms_driven]]

    prediction = model.predict(labels)
    prediction = np.ravel(prediction)[0]

    return render_template("project.html", prediction_text=f"Estimated Bike Price: {prediction}")


if __name__ == '__main__':
    app.run(debug=True)