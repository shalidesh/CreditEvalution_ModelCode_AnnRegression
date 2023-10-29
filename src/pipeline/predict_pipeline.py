import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object,load_object_h5
import os
import datetime


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.h5")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object_h5(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        yom: str,
        milage: str,
        strock:str,
        light_type: str,
        ):

        self.mileage = milage

        self.yom = yom

        self.curr_year=datetime.datetime.now().year

        self.age=self.curr_year - int(self.yom) if int(self.yom) else 0

        self.strock = strock

        self.light_type = light_type


    def get_data_as_data_frame(self):
        try:

            custom_data_input_dict = {
                "mileage": [int(self.mileage)],
                "Age": [self.age],
                "stroke_values": [self.strock],
                "Light Type": [self.light_type]      
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
