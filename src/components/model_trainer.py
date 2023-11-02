import os
import sys
from dataclasses import dataclass
from sklearn.metrics import r2_score

# Neural Net modules
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from src.components.model_building import ModelBuilding

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object_h5
from sklearn.metrics import mean_absolute_error

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.h5")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            input_value=X_train.shape[1]
            
            logging.info("Model Building")

            model_instance=ModelBuilding()
            model=model_instance.model_create(input_value)

            # compile the model
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            # early stopping callback
            es = EarlyStopping(monitor='val_loss',
                            mode='min',
                            patience=50,
                            restore_best_weights = True)

            # fit the model!
            logging.info("Model training start")

            history = model.fit(X_train, y_train,
                                validation_data = (X_test, y_test),
                                callbacks=[es],
                                epochs=1000,
                                batch_size=32,
                                verbose=1)

            logging.info("Model training finished")

            try:
                # Save the model in HDF5 format
                model.save(self.model_trainer_config.trained_model_file_path)

            except Exception as e:
                raise CustomException(e, sys)
            
            logging.info("Model Saved in artifact folder")
 
            pred = model.predict(X_test)
            mae_value=mean_absolute_error(y_test, pred)

            return mae_value
            
            
        except Exception as e:
            raise CustomException(e,sys)