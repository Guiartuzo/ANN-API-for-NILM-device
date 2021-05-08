from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas
import glob
import time

class ProcessData:
    
    def __init__(self):

        self.appliances = {}

        self.dates = {}


    def Scaler(self, X):
        scaler = MinMaxScaler( feature_range=(0, 1))
        return scaler.fit_transform(X)


    def Spliter(self, x_Scaled, Y):
        
        x_Train, x_Test, y_Train, y_Test = train_test_split(x_Scaled, Y, test_size=0.33, random_state=42)
        inputShape = x_Train.shape[1]
        
        return x_Train, x_Test, y_Train, y_Test, inputShape


    def Resampler(self, dataFrame, time, appliance):
        return dataFrame.resample(time)['mains-1', 'mains-2', appliance].sum()


    def AddDifAverage(self, dataFrame):
        dataFrame['mains-dif'] = dataFrame['mains-1'] - dataFrame['mains-2']
        dataFrame['mains-avg'] = (dataFrame['mains-1'] + dataFrame['mains-2']) // 2 
        return dataFrame
        

    def LoadApplianceValues(self, house):

        path_to_app = 'low_freq\\house_{}\\labels.dat'.format(house)
        
        with open(path_to_app) as apps:
            for line in apps:
                labels = line.split(' ')
                self.appliances[int(labels[0])] = labels[1].strip() + '-' + labels[0] 
        
        print(self.appliances)


    def OpenHouseData(self, house):

        path_to_house = 'low_freq\\house_{}\\'.format(house)
        file = path_to_house + 'channel_1.dat'

        dataset_house = pandas.read_table(file, sep = ' ', names = ['unix_time', self.appliances[1]], dtype = {'unix_time': 'int64', self.appliances[1]:'float64'}) 

        num_apps = len(glob.glob(path_to_house + 'channel*'))    

        for apps in range(2, num_apps + 1):
            path_to_file = '{0}channel_{1}.dat'.format(path_to_house, apps)
            app_data = pandas.read_table(path_to_file, sep = ' ', names = ['unix_time', self.appliances[apps]], dtype = {'unix_time': 'int64', self.appliances[apps]:'float64'}) 
            dataset_house = pandas.merge(dataset_house, app_data, how = 'inner', on = 'unix_time')

        dataset_house['timestamp'] = dataset_house['unix_time'].astype("datetime64[s]")
        dataset_house = dataset_house.set_index(dataset_house['timestamp'].values)
        dataset_house.drop(['unix_time','timestamp'], axis=1, inplace=True)
        
        return dataset_house

    def GetDates(self, dataFrame):
        self.dates = [str(time)[:10] for time in dataFrame.index.values]
        return sorted(list(set(self.dates)))
    


    # def __init__(self):

    #     self.app = {'unix_time': [],
    #                     'labels': []
    #     }

    #     self.house  = {'unix_time': [],
    #                     'labels': []
    #     }

    #     self.dataset_app = pandas.DataFrame(self.app, columns = ['unix_time', 'labels'])
    #     self.dataset_house = pandas.DataFrame(self.house, columns = ['unix_time', 'labels'])

    # def load_appliance_values(self,house, app_label):
    #     path_to_app = 'low_freq\\{0}\\channel_{1}.dat'.format(house, app_label)
    #     house_app = pandas.read_table(path_to_app, sep = ' ', names = ['unix_time', 'labels'], dtype = {'unix_time': 'int64', 'labels':'float64'})
    #     self.dataset_app = self.dataset_app.append(house_app)

    #     print(house_app)
    #     print(self.dataset_app)

    
    # def load_house_values(self,house):
    #     path_to_house = 'low_freq\\{0}\\channel_'.format(house)
    #     house_mains_1 = pandas.read_table(path_to_house + '1.dat', sep = ' ', names = ['unix_time', 'labels'], dtype = {'unix_time': 'int64', 'labels':'float64'}) 
    #     house_mains_2 = pandas.read_table(path_to_house + '2.dat', sep = ' ', names = ['unix_time', 'labels'], dtype = {'unix_time': 'int64', 'labels':'float64'}) 
    #     self.dataset_house = pandas.merge(house_mains_1, house_mains_2, how = "inner", on = "unix_time")
    #     print(house_mains_1)
    #     print(house_mains_2)
    #     print(self.dataset_house)


    # def get_days(self, house):
    #     path_to_house = 'low_freq\\{0}\\channel_'.format(house)
    #     house_mains_1 = pandas.read_table(path_to_house + '1.dat', sep = ' ', names = ['unix_time', 'labels'], dtype = {'unix_time': 'int64', 'labels':'float64'}) 
        
    #     for index, row in house_mains_1.iterrows():
    #         print(datetime.utcfromtimestamp(row['unix_time']).strftime('%Y-%m-%d %H:%M:%S'))







    # for house in os.listdir('low_freq'):
        
    #     pd.load_house_values(house)

    #     # pd.get_days(house)

    #     print(house)
    #     df_house_eletrodomesticos = pandas.read_table('low_freq\\' + house + '\\labels.dat', sep = ' ', names = ['index', 'value'], dtype = {'index': 'int32', 'value':'str'}) 

    #     rows = df_house_eletrodomesticos[df_house_eletrodomesticos['value'].str.contains("refrigerator")]


    #     if rows['index'].size > 0:
    #         print(int(rows['index'].values))
    #         pd.load_appliance_values(house, int(rows['index'].values))
        
    #     data_set_raw =  pandas.merge(pd.dataset_house, pd.dataset_app, how = "inner", on = "unix_time")

    #     print(data_set_raw)

    #     dataset = data_set_raw.values

    #     X = dataset[:,1:2]
    #     Y = dataset[:,3]

    #     X_train = X[0:432000,:]
    #     Y_train = Y[0:432000]

    #     X_val = X[0:86400,:]
    #     Y_val = Y[0:86400]

    #     print(X_train)
    #     print(Y_train)

    #     print(X_val)
    #     print(Y_val)

    #     # min_max_scaler = preprocessing.MinMaxScaler()
    #     # X_scale = min_max_scaler.fit_transform(X)
    #     # # Y_scale = min_max_scaler.fit_transform(Y)

    #     # X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
    #     # X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
    #     print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)

 
    #     model = Sequential([
    #         Dense(32, activation='relu', 
    #         input_shape=(1,)),
    #         Dense(32, activation='relu'),
    #         Dense(1, activation='sigmoid'),
    #         ])

    #     model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

    #     hist = model.fit(X_train , Y_train , batch_size=32, epochs=100, validation_data=(X_val, Y_val))

    #     model.evaluate(X_val, Y_val)[1]

    #     break