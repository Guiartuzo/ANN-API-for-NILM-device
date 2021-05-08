from keras.models import load_model
import numpy   

class Predictions:


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