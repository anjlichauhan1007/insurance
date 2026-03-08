
from flask import Flask , render_template ,url_for,request# flask library se Flask ko import kar rahe h ko import kiya h
import joblib#model ko load or save karne ke liye
import numpy as np
model = joblib.load('liner_model.lb')
# model=joblib.load('liner_model.lb')
app=Flask(__name__)#app se object banaya h 
@app.route('/')# ye decorator h iska kaam hota h function ka kaam change kar dena
def home():
    return render_template("index.html")#render tamplate wale folder se template(html)ko leke aata h

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/project')
def project():
    return render_template("project.html")

@app.route('/predict' ,methods = ['POST','GET'])
def predict():
    if request.method=='POST':#request libray frontend se backend me data lata h
        brand_name=request.form['brand_name']
        owner=int(request.form['owner'])
        age=int(request.form['age'])
        power=int(request.form['power'])
        kms_driven=int(request.form['kms_driven'])
        print("mera data upload ho gya")
        # print("lebels-",brand_name,owner,age,power,kms_driven)
        brand_dict={  'TVS':1,   'Royal Enfield':2,         'Triumph':3,          'Yamaha':4,
           'Honda':5,            'Hero':6,           'Bajaj':7,          'Suzuki':8,
         'Benelli':9,             'KTM':10,        'Mahindra':11,        'Kawasaki':12,
          'Ducati':13,         'Hyosung':14, 'Harley-Davidson':15,            'Jawa':16,
             'BMW':17,          'Indian':18,         'Rajdoot':19,             'LML':20,
           'Yezdi':21,              'MV':22,           'Ideal':23

             }
        brand_name=brand_dict[brand_name]
        print("labels:",brand_name,owner,age,power,kms_driven)
        #2d me conver and predict ke liy
        lebels=[[brand_name,owner,age,power,kms_driven]]
        pred=model.predict(lebels)
        print(type(pred))
        # pred=np.ravel(pred)
        print("OUTPUT:",pred)
        # pred=pred[0]
        return render_template('project.html',prediction=pred)

    return render_template('project.html')

if __name__=="__main__":
     app.run(debug=True)