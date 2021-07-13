from sklearn.model_selection import train_test_split
import pandas
import glob

class ProcessData:
    
    def __init__(self):
        self.appliances = {}
        self.dates = {}


    def Spliter(self, x_Train, y_Train):
        x_Train, x_Test, y_Train, y_Test = train_test_split(x_Train, y_Train, test_size=0.25, shuffle=False)
        return x_Train, x_Test, y_Train, y_Test


    def LoadApplianceValues(self, house):
        path_to_app = 'low_freq\\house_{}\\labels.dat'.format(house)

        with open(path_to_app) as apps:
            for line in apps:
                labels = line.split(' ')
                self.appliances[int(labels[0])] = labels[1].strip() + '-' + labels[0] 


    def GetDates(self, dataFrame):
        self.dates = [str(time)[:10] for time in dataFrame.index.values]
        return sorted(list(set(self.dates)))


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
        dataset_house['hours'] = pandas.to_datetime(dataset_house['timestamp'],format="%H").dt.hour
        dataset_house = dataset_house.set_index(dataset_house['timestamp'].values)
        dataset_house.drop(['timestamp','unix_time'], axis=1, inplace=True)
        return dataset_house


    def LoadMultipleHouses(self, house, app_label):
        selected_appliance = []
        self.LoadApplianceValues(house)
        data_set = self.OpenHouseData(house)
        dates = self.GetDates(data_set)
        x_Train = data_set[['hours', 'mains-1', 'mains-2']]
        selected_appliance = [k for k, v in self.appliances.items() if app_label in v]

        for app in selected_appliance:
            print('{0}-{1}'.format(app_label,app))
            y_Train = data_set['{0}-{1}'.format(app_label,app)]

        return x_Train, y_Train, dates