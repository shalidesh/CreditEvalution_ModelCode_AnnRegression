import os
import sys
from tensorflow.keras.models import load_model
import pickle
import tensorflow as tf
import pandas as pd

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        print(e)

def load_object_h5(file_path):
    try:
        model = tf.keras.models.load_model(file_path)
        return model

    except Exception as e:
        print(e)

custom_data_input_dict = {
                "mileage": [25000],
                "Age": [5],
                "stroke_values": ['2 stroke'],
                "Light Type": ['Double Light']      
            }

df=pd.DataFrame(custom_data_input_dict)

    
model_path=os.path.join("artifacts","model.h5")
preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
print("Before Loading")
model=load_object_h5(file_path=model_path)
preprocessor=load_object(file_path=preprocessor_path)
print("After Loading")
data_scaled=preprocessor.transform(df)
preds=model.predict(data_scaled)
print(preds)