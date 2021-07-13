from desagregate import data_process, model_construction, model_predict
from keras.utils.vis_utils import plot_model
import os

class Main:

    def __init__(self):
        
        self.pd = data_process.ProcessData()
        self.mc = model_construction.Model()
        self.mp = model_predict.Predictions()
        
        self.appliances = [
            "refrigerator",
            "dishwaser",
            "microwave"
        ]

        if not os.path.exists("trained_models"):
            os.mkdir("trained_models")


if __name__ == "__main__":

    main = Main()

    for house in range(1,2):

        for app in main.appliances:
            name = "trained_models\\house-{0}_{1}.hdf5".format(house,app)
            print("casa {0} / app {1}".format(house, app))

            x, y, dates = main.pd.LoadMultipleHouses(house, app)
            y.columns = [app]

            x_Train = x
            y_Train = y

            print(x_Train)
            print(y_Train)

            # x_Train, x_Test, y_Train, y_Test = main.pd.Spliter(x_Train, y_Train)
            built_model = main.mc.ModelConstruct([3, 64, 128, 256, 1]) 
            # plot_model(built_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
            # main.mc.train(name,built_model, x_Train,y_Train)
            main.mp.HousePredict(house, name, app)