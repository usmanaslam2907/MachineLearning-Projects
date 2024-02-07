# from turtle import pd
import pandas as pd
from flask import Flask, render_template, request
import numpy as np
import pickle

with open('irismodel.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/Predict', methods=['POST'])
def predict_flower():
    sepallength =request.form.get('sepallength')
    sepalwidth = request.form.get('sepalwidth')
    petallength =request.form.get('petallength')
    petalwidth =request.form.get('petalwidth')
    data = {
        'sepal_length': [sepallength],
        'sepal_width': [sepalwidth],
        'petal_length': [petallength],
        'petal_width': [petalwidth]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)
    result = model.predict(df)
    result=str(result)
    return result


if __name__ == '__main__':
    app.run(debug=True)
