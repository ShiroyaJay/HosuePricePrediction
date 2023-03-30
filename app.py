import pickle
from flask import Flask,request,app,jsonify,url_for,render_template,escape
import numpy as np
import pandas as pd

app=Flask(__name__)
model=pickle.load(open('model_reg_house_prediction','rb'))
scaler=pickle.load(open('scale','rb'))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict_api",methods=["POST"])
def predict_api():
    data= request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data= scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)[0]
    return jsonify(output)

if __name__=="__main__":
    app.run(debug=True)