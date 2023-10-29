# Neural Net modules
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

class ModelBuilding:

    def model_create(self,input_value:int):

        self.input_value=input_value
        
        model = Sequential()
        model.add(Dense(120, input_shape=(self.input_value,), activation='relu')) # (features,)
        model.add(Dense(60, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(1, activation='linear')) # output node
        model.summary() # see what your model looks like

        return model


    