import numpy as np
from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pandas as pd
import pickle

from pandas.core.interchange import column

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('LinearModel.pkl','rb'))
car=pd.read_csv('Cleaned_Data.csv')
@app.route('/',methods=['GET','POST'])
def index():
    companies = sorted(car['company'].unique())
    models  = sorted(car['name'].unique())
    year = sorted(car['year'].unique(),reverse=True)
    fuel_type = car['fuel_type'].unique()

    return render_template('index.html',companies=companies,models=models,year=year,fuel_type=fuel_type)

@app.route('/predict',methods=['POST'])
def predict():
    company=request.form.get('company')
    car_model=request.form.get('model')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel')
    km_driven=request.form.get('km_driven')

    print(company,car_model,year,fuel_type,km_driven)

    prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],data=np.array([car_model,company,year,km_driven,fuel_type]).reshape(1, 5)))
    print(prediction)
    return  str(np.round(prediction[0],2))

if __name__=='__main__':
    app.run(debug=True)