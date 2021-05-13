from keras.models import load_model
from data_process import ProcessData
from keras.models import Sequential
import numpy   
import pandas

class Predictions:
    
    pd = ProcessData()

    def mse_loss(self, y_predict, y):
        return numpy.mean(numpy.square(y_predict - y)) 


    def mae_loss(self, y_predict, y):
        return numpy.mean(numpy.abs(y_predict - y)) 


    def ModelTest(self, model, x_Test, y_Test):

        modelo = load_model(model)
        prediction = modelo.predict(x_Test).reshape(-1)

        model_mse_loss = self.mse_loss(prediction, y_Test)
        model_mae_loss = self.mae_loss(prediction, y_Test)
        
        print('Mean square error on test set: ', model_mse_loss)
        print('Mean absolute error on the test set: ', model_mae_loss)

        return prediction, model_mse_loss, model_mae_loss

    
    # def PredictHouseAppliances(self, house):


    #     self.pd.LoadApplianceValues(house)

    #     dataset_house = self.pd.OpenHouseData(house)

    #     house_dates = self.pd.GetDates(dataset_house)

    #     processed_dataset = self.pd.AddDifAverage(dataset_house)

    #     refrig_test = pandas.read_table('low_freq\\house_1\\channel_5.dat', sep = ' ', names = ['unix_time', 'refrigerator'], dtype = {'unix_time': 'int64', 'refrigerator':'float64'}) 
    #     print(refrig_test)

    #     data_test = processed_dataset.loc[house_dates[0] : house_dates[5]]
    #     # refrig_test = refrig_test.loc[house_dates[0] : house_dates[5]]
    #     print(refrig_test)



    #     x_Test = data_test[['mains-1',  'mains-2', 'mains-dif', 'mains-avg']]
    #     y_Test = refrig_test['refrigerator']

    #     print(y_Test)

    #     # self.ModelTest('4th_model.hdf5', x_Test, y_Test)

    #     # prediction = modelo.predict(x_Test).reshape(-1)

    #     model = Sequential()
    #     model.predict_proba(prediction)


        
            

    #     # selected_appliance = [k for k, v in self.pd.appliances.items() if 'refrigerator' in v]
    #     # for app in selected_appliance: 
        
    #     #     print('refrigerator-{}'.format(app))
            

    #     #     dataset_house = self.pd.OpenHouseData(house)

    #     #     house_dates = self.pd.GetDates(dataset_house)

    #     #     processed_dataset = self.pd.AddDifAverage(dataset_house)
        
    #     #     # print(processed_dataset)

    #     #     data_test = processed_dataset.loc[house_dates[0] : house_dates[5]]

    #     #     x_Test = data_test[['mains-1',  'mains-2', 'mains-dif', 'mains-avg']]
    #     #     y_Test = data_test['refrigerator-{}'.format(app)]



    #     #     self.ModelTest('4th_model.hdf5', x_Test, y_Test)