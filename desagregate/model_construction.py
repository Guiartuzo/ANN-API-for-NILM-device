from data_process import ProcessData
from data_plotter import Plotter
from model_predict import Predictions
import time
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import numpy


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

    def ModelConstruct_2(self, layers, inputShape):
      
        model = Sequential()
      
        for i in range(len(layers)-1):
            model.add( Dense(input_shape=(inputShape,), units = (layers[i+1])))
            model.add( Dropout(0.5) )
      
            if i < (len(layers) - 2):
                model.add( Activation('relu') )

        model.summary()
        return model


    def ModelFit(self, model, x_Train, x_Test, y_Train, y_Test):
        
        adam = Adam(lr = 1e-5)
        model.compile(loss='mean_squared_error', optimizer=adam)
        
        start = time.time()
        
        checkpointer = ModelCheckpoint(filepath="4rd_model.hdf5", verbose=0, save_best_only=True)
        
        model.fit( x_Train, y_Train, batch_size=512, verbose=1, epochs=200, validation_data=(x_Test,y_Test), callbacks=[checkpointer])
        
        print('Finish trainning. Time: ', time.time() - start)



if __name__ == "__main__":

    pd = ProcessData()
    plt = Plotter()
    md = Model()
    predict = Predictions()

    pd.LoadApplianceValues(1)

    dataset_house = pd.OpenHouseData(1)
    # numpy.savetxt('dataset_house.txt', dataset_house.values, fmt='%d')

    house_dates = pd.GetDates(dataset_house)
    print(house_dates)
    
    # resampled_dataframe = pd.Resampler(dataset_house, '3s', 'refrigerator-5')
    # numpy.savetxt('resampled_dataframe.txt', resampled_dataframe.values, fmt='%d')

    processed_dataset = pd.AddDifAverage(dataset_house)
    numpy.savetxt('processed_dataset.txt', processed_dataset.values, fmt='%d')

    data_train = processed_dataset.loc[house_dates[0] : house_dates[17]]
    data_test = processed_dataset.loc[house_dates[18] : ]
    # data_val = processed_dataset.loc[house_dates[11] : house_dates[16]]

    print(data_train)
    # print(data_val)
    print(data_test)

    # numpy.savetxt('data_train.txt', data_train.values, fmt='%d')
    # numpy.savetxt('data_val.txt', data_val.values, fmt='%d')
    # numpy.savetxt('data_test.txt', data_test.values, fmt='%d')

    # print(data_train.shape)
    # print(data_val.shape)
    # print(data_test.shape)
    

    x_Train = data_train[['mains-1',  'mains-2', 'mains-dif', 'mains-avg']]
    y_Train = data_train['refrigerator-5']

    # x_Val = data_val[['mains-1',  'mains-2', 'mains-dif', 'mains-avg']]
    # y_Val = data_val['refrigerator-5']

    x_Test = data_test[['mains-1',  'mains-2', 'mains-dif', 'mains-avg']]
    y_Test = data_test['refrigerator-5']

    # X_predict = data_val[['mains-1',  'mains-2']]

    print(x_Train)
    print(y_Train)

    # print(x_Val)
    # print(y_Val)

    print(x_Test)
    print(y_Test)

    # print(x_Train.shape, y_Train.shape, x_Val.shape, y_Val.shape, x_Test.shape, y_Test.shape)

    # x_Scaled = pd.Scaler(x)

    # x_Train, x_Test, y_Train, y_Test, inputShape = pd.Spliter(x_Scaled, y)
    # x_Train, x_Test, y_Train, y_Test = train_test_split(x_Scaled, Y, test_size=0.33, random_state=42)


    # y_plot = y.loc[house_dates[1] : house_dates[4]]
    # house_plot = processed_dataset.loc[house_dates[1] : house_dates[4]]
    # print(y_plot)
    # print(house_plot)


    prediction, model_mse_loss, model_mae_loss = predict.ModelTest('3rd_model.hdf5', x_Test, y_Test)
    plt.PlotPredictions(data_test, house_dates[18:], prediction, y_Test, "teste", look_back=50)

    built_model = md.ModelConstruct([4, 256, 512, 1024, 1])
    md.ModelFit(built_model, x_Train, x_Test, y_Train, y_Test)
