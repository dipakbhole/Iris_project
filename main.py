from flask import Flask,render_template,request
import numpy as np

import json
import pickle
import CONFIG 

with open (CONFIG.MODEL_PATH,'rb') as file:
    model = pickle.load(file)

with open (CONFIG.ASSET_PATH,'r') as file:
    asset = json.load(file) 

col = asset['columns']       

app = Flask (__name__)

@app.route('/')
def start():
    return render_template('flow.html')

@app.route('/predict',methods= ['POST'])
def predict_species():

    input_data = request.form
    
    data = np.zeros(len(col))

    data[0]  = input_data['html_sl']
    data[1]  = input_data['html_sw']
    data[2]  = input_data['html_pl']
    data[3]  = input_data['html_pw']


    result= model.predict([data])

    print(result)

    if result[0] == 0:
        iris_value = "SETOSA"
    if result[0] == 1:
        iris_value = "VERSICOLOR"
    if result[0] == 2:
        iris_value = "VIRGINICA"

    return render_template("flow.html",PREDICT_VALUE=iris_value) 


if __name__ == '__main__':
    app.run(host=CONFIG.HOST_NAME,port=CONFIG.PORT_NUMBER)


