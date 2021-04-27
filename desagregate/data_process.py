import os
import datetime
import pandas

# import numpy as np

# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

# from keras.models import Sequential
# from keras.layers import Dense
# house_consumo = pandas.read_table('channel_1.dat', sep = ' ', names = ['unix_time', 'labels'], dtype = {'unix_time': 'int64', 'labels':'float64'}) 

class ProcessData:
   
    def __init__(self):

        self.app = {'unix_time': [],
                        'labels': []
        }

        self.house  = {'unix_time': [],
                        'labels': []
        }

        self.dataset_app = pandas.DataFrame(self.app, columns = ['unix_time', 'labels'])
        self.dataset_app = pandas.DataFrame(self.house, columns = ['unix_time', 'labels'])

    def load_appliance_values(self,house, app_label):
        path_to_app = 'low_freq\\{0}\\channel_{1}.dat'.format(house, app_label)
        house_app = pandas.read_table(path_to_app, sep = ' ', names = ['unix_time', 'labels'], dtype = {'unix_time': 'int64', 'labels':'float64'})
        self.dataset_app = self.dataset_app.append(house_app)

        print(house_app)
        print(self.dataset_app)

    
    def load_house_values(self,house):
        pass
        # path_to_house = ''
        # house_values = ''


if __name__ == "__main__":

    pd = ProcessData()

    for house in os.listdir('low_freq'):

        print(house)
        df_house_eletrodomesticos = pandas.read_table('low_freq\\' + house + '\\labels.dat', sep = ' ', names = ['index', 'value'], dtype = {'index': 'int32', 'value':'str'}) 

        rows = df_house_eletrodomesticos[df_house_eletrodomesticos['value'].str.contains("refrigerator")]


        if rows['index'].size > 0:
            print(int(rows['index'].values))
            pd.load_appliance_values(house, int(rows['index'].values))