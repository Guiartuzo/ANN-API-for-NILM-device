from desagregate import data_process
from desagregate import data_plotter
from keras.models import load_model
import numpy   

class Predictions:
    
    pd = data_process.ProcessData()
    plt = data_plotter.Plotter()

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


    def HousePredict(self, house, model, app_label):
        self.pd.LoadApplianceValues(house)
        dataset_house = self.pd.OpenHouseData(house)
        house_dates = self.pd.GetDates(dataset_house)
        data_test = dataset_house.loc[house_dates[0] : house_dates[5]]
        selected_appliance = [k for k, v in self.pd.appliances.items() if app_label in v]

        for app in selected_appliance:
            print('{0}-{1}'.format(app_label,app))
            y_Test = data_test['{0}-{1}'.format(app_label,app)]

        x_Test = data_test[['hours', 'mains-1', 'mains-2']]
        prediction, model_mse_loss, model_mae_loss = self.ModelTest(model, x_Test, y_Test)
        self.plt.PlotPredictions(data_test, house_dates[0:5], prediction, y_Test)
        average_diff = self.getApplianceAverage(100, 5000, 30, y_Test,prediction)
        data_test = dataset_house.loc[house_dates[5] : house_dates[10]]
        selected_appliance = [k for k, v in self.pd.appliances.items() if app_label in v]

        for app in selected_appliance:
            print('{0}-{1}'.format(app_label,app))
            y_Test = data_test['{0}-{1}'.format(app_label,app)]

        x_Test = data_test[['hours', 'mains-1', 'mains-2']]
        prediction, model_mse_loss, model_mae_loss = self.ModelTest(model, x_Test, y_Test)
        prediction_2 = numpy.multiply(prediction, average_diff)
        self.plt.PlotPredictions(data_test, house_dates[5:10], prediction_2, y_Test)
        model_mse_loss = self.mse_loss(prediction_2, y_Test)
        model_mae_loss = self.mae_loss(prediction_2, y_Test)
        print(model_mse_loss)
        print(model_mae_loss)
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
        data_array = numpy.where(numpy.logical_and(data>=100, data<=300))
        data_values = data[data_array]

        for i in data_values:
            data_sum += i

        pred_array = numpy.where(numpy.logical_and(predict>=100, predict<=5000))
        pred_values = predict[pred_array]

        for i in pred_values:
            predict_sum += i

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
