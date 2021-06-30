from desagregate import data_process
from desagregate import data_plotter
from keras.models import load_model
from sklearn.metrics import precision_score, accuracy_score
import numpy   

class Predictions:
    
    pd = data_process.ProcessData()
    plt = data_plotter.Plotter()

    def mse_loss(self, y_predict, y):
        return numpy.mean(numpy.square(y_predict - y)) 


    def mae_loss(self, y_predict, y):
        return numpy.mean(numpy.abs(y_predict - y)) 

    def mean_percentage_error(self,y_predict, y):
        return numpy.mean(numpy.abs((y - y_predict) / y)) * 100

    def ModelTest(self, model, x_Test, y_Test):

        modelo = load_model(model)
        prediction = modelo.predict(x_Test).reshape(-1)

        model_mse_loss = self.mse_loss(prediction, y_Test)
        model_mae_loss = self.mae_loss(prediction, y_Test)
        model_percentage_error = self.mean_percentage_error(y_Test, prediction)
        
        print('Mean square error on test set: ', model_mse_loss)
        print('Mean absolute error on the test set: ', model_mae_loss)
        print('percentage : ', model_percentage_error)

        return prediction, model_mse_loss, model_mae_loss


    def HousePredict(self, house, model, app_label):
        
        self.pd.LoadApplianceValues(house)
        dataset_house = self.pd.OpenHouseData(house)
        processed_dataset = self.pd.AddDifAverage(dataset_house)
        house_dates = self.pd.GetDates(dataset_house)

        data_test = processed_dataset.loc[house_dates[0] : house_dates[5]]
        
        selected_appliance = [k for k, v in self.pd.appliances.items() if app_label in v]

        for app in selected_appliance:
            
            print('{0}-{1}'.format(app_label,app))
            y_Test = data_test['{0}-{1}'.format(app_label,app)]


        x_Test = data_test[['hours', 'mains-1', 'mains-2']]

        prediction, model_mse_loss, model_mae_loss = self.ModelTest(model, x_Test, y_Test)
        self.plt.PlotPredictions(data_test, house_dates[0:5], prediction, y_Test, "teste", look_back = 50)


        average_diff = self.getApplianceAverage(100, 5000, 30, y_Test,prediction)

        data_test = processed_dataset.loc[house_dates[5] : house_dates[10]]
        
        selected_appliance = [k for k, v in self.pd.appliances.items() if app_label in v]

        for app in selected_appliance:
            
            print('{0}-{1}'.format(app_label,app))
            y_Test = data_test['{0}-{1}'.format(app_label,app)]

        x_Test = data_test[['hours', 'mains-1', 'mains-2']]

        prediction, model_mse_loss, model_mae_loss = self.ModelTest(model, x_Test, y_Test)
        prediction_2 = numpy.multiply(prediction, average_diff)
        self.plt.PlotPredictions(data_test, house_dates[5:10], prediction_2, y_Test, "teste", look_back = 50)

        model_mse_loss = self.mse_loss(prediction_2, y_Test)
        model_mae_loss = self.mae_loss(prediction_2, y_Test)
        model_percentage_error = self.mean_percentage_error(y_Test, prediction)

        print(model_mse_loss)
        print(model_mae_loss)
        print(model_percentage_error)

        self.getApplianceAverage_after(100, 5000, 30, y_Test,prediction,average_diff)




    def getApplianceAverage(self, data_start, data_limit, pred_start, dataset, predict):
        
        data_sum = 0
        predict_sum = 0

        data = dataset.to_numpy()

        data_array = numpy.where(numpy.logical_and(data>=100, data<=300))
        data_values = data[data_array]

        for i in data_values:
            data_sum += i

        pred_array = numpy.where(numpy.logical_and(predict>=50, predict<=300))
        pred_values = predict[pred_array]

        for i in pred_values:
            predict_sum += i

        print(data_sum / len(data_values))
        print(predict_sum / len(pred_values))

        correction_factor = (data_sum / len(data_values)) / (predict_sum / len(pred_values))

        self.getApplianceTotal(data, predict, 1)
        self.getApplianceTotal(data, predict, correction_factor)

        return correction_factor








    def getApplianceAverage_after(self, data_start, data_limit, pred_start, dataset, predict, correction_factor):
        
        data_sum = 0
        predict_sum = 0

        data = dataset.to_numpy()

        data_array = numpy.where(numpy.logical_and(data>=200, data<=5000))
        data_values = data[data_array]

        for i in data_values:
            data_sum += i

        pred_array = numpy.where(numpy.logical_and(predict>=20, predict<=5000))
        pred_values = predict[pred_array]

        for i in pred_values:
            predict_sum += i

        print(data_sum / len(data_values))
        print(predict_sum / len(pred_values))

        # correction_factor = (data_sum / len(data_values)) / (predict_sum / len(pred_values))

        self.getApplianceTotal(data, predict, 1)
        self.getApplianceTotal(data, predict, correction_factor)












    def getApplianceTotal(self, data, predict, correction_factor):

        real_appliance_consumption = 0
        predict_appliance_consumption = 0

        for i in data:
            real_appliance_consumption += i

        for i in predict:
            predict_appliance_consumption += (i * correction_factor)

        print("Consumo real: {}" .format(real_appliance_consumption))
        print("Consumo previst: {}".format(predict_appliance_consumption))

        return real_appliance_consumption, predict_appliance_consumption    































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