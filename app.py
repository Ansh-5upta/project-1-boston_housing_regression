import pickle
from flask import Flask,render_template,request,app,jsonify,url_for
import numpy as np
import pandas as pd

app = Flask(__name__)

#load the model
remodel = pickle.load(open('regmodel.pkl','rb'))
scaler = pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict_api',methods=['POST'])

def predict_api():
    data = request.json['data']
    print("Input JSON :",data)
    input_data = np.array(list(data.values())).reshape(1,-1)
    print("Input Data Array:", input_data)

    new_data = scaler.transform(input_data)
    print("Scaler Data:", new_data)
    output = remodel.predict(new_data)
    print("Prediction : ",output[0])
    return jsonify({'perdiction': float(output[0])})

@app.route('/predict', methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = remodel.predict(final_input)
    return render_template("home.html", prediction_text="The predicted value is {}".format(output))

if(__name__=="__main__"):
    app.run(debug=True)