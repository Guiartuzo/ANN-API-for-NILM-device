# from data_process import ProcessData
import time
from keras.layers.core import Dense, Activation, Dropout

from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam


class Model:


    def ModelConstruct(self, layers):
      
        model = Sequential()
      
        for i in range(len(layers)-1):
            model.add( Dense(input_shape=(layers[i],), units = (layers[i+1])))
            model.add( Dropout(0.5) )
      
            if i < (len(layers) - 2):
                model.add( Activation('relu') )

        model.summary()
        return model


    def ModelFit(self, name, model, x_Train, x_Test, y_Train, y_Test):
        
        adam = Adam(lr = 1e-5)
        model.compile(loss='mean_squared_error', optimizer=adam)
        
        start = time.time()
        
        checkpointer = ModelCheckpoint(filepath=name, verbose=0, save_best_only=True)
        
        model.fit( x_Train, y_Train, batch_size=512, verbose=1, epochs=200, validation_data=(x_Test,y_Test), callbacks=[checkpointer])
        
        print('Finish trainning. Time: ', time.time() - start)


    def train(self, name, model, x_Train, y_Train):

        start = time.time()
        adam = Adam(lr = 5e-5)
        model.compile(loss='mean_squared_error', optimizer=adam)
        checkpointer = ModelCheckpoint(filepath=name, verbose=0, save_best_only=True)
        hist_model = model.fit(
                    x_Train,
                    y_Train,
                    batch_size=512,
                    verbose=1,
                    epochs=400,
                    validation_split=0.3,
                    callbacks=[checkpointer])
        print('Finish trainning. Time: ', time.time() - start)