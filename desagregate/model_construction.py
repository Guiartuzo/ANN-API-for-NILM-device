from data_process import ProcessData
from data_plotter import Plotter

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from keras.regularizers import l2

class Model:

    n_cols=X_train.shape[1]

    def build_fc_model(self, layers):
        fc_model = Sequential()
        for i in range(len(layers)-1):
            fc_model.add( Dense(input_shape=(n_cols,), units = (layers[i+1])))
            fc_model.add( Dropout(0.5) )
            if i < (len(layers) - 2):
                fc_model.add( Activation('relu') )
        fc_model.summary()
        return fc_model


if __name__ == "__main__":

    pd = ProcessData()
    plt = Plotter()

    pd.LoadApplianceValues(1)
    dataset_house = pd.OpenHouseData(1)
    resampled_dataframe = pd.Resampler(dataset_house, '3s', 'refrigerator-5')
    print(resampled_dataframe)
    result = pd.AddDifAverage(resampled_dataframe)
    print(result)
    plt.PlotDataFrame(resampled_dataframe, "teste")

    fc_model_1 = build_fc_model([2, 256, 512, 1024, 1])
