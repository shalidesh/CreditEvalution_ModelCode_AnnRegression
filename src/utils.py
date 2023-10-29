import os
import sys
from tensorflow.keras.models import load_model
import pickle
import tensorflow as tf

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def save_object_h5(file_path, model):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        # Save the model in HDF5 format
        model.save(file_path)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object_h5(file_path):
    try:
        model = tf.keras.models.load_model(file_path)
        return model

    except Exception as e:
        raise CustomException(e, sys)
