import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from src.utils import save_object , evaluate_models

@dataclass
class ModelTrainerConfig:
    trainer_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and independent variable')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            ## Train multiple models

            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet()
        }
            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print("========================================")
            logging.info(f'model_report : {model_report}')
            
            #Get the best model 
            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]
            print(f'Best Model Found ,Model Name: {best_model_name} ,R2_score : {best_model_score}')
            print('\n=========================================')
            logging.info(f'Best Model Found ,Model Name: {best_model_name} ,R2_score : {best_model_score}')
            
            save_object(
                file_path=self.model_trainer_config.trainer_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info("Exception occured at Model Training")
            raise CustomException(e,sys)
