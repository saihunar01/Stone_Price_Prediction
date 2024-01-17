from flask import Flask,request,render_template,jsonify
from src.Pipeline.prediction_pipeline import CustomeData,PredictPipeline
'''import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion
from src.components.mode_trainer import ModelTrainer'''


application=Flask(__name__)

app=application


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomeData(
            carat=float(request.form.get('carat')),
            depth=float(request.form.get('depth')),
            table=float(request.form.get('table')),
            x=float(request.form.get('x')),
            y=float(request.form.get('y')),
            z=float(request.form.get('z')),
            cut=request.form.get('cut'),
            color=request.form.get('color'),
            clarity=request.form.get('clarity')
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)
        results=round(pred[0],2)
        return render_template('result.html',final_result=results)



if __name__=="__main__":
    '''obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    Model_trainer=ModelTrainer()
    Model_trainer.initiate_model_training(train_arr,test_arr)'''
    app.run(host='0.0.0.0',debug=True)