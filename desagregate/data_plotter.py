import matplotlib.pyplot as plt

class Plotter:

    def PlotDataFrame(self, dataFrame, title):
        
        apps = dataFrame.columns.values
        num_apps = len(apps)
        fig, axes = plt.subplots((num_apps+1)//2,2, figsize=(24, num_apps*2) )
        
        for i, key in enumerate(apps):
                axes.flat[i].plot(dataFrame[key], alpha = 0.6)
                axes.flat[i].set_title(key, fontsize = '15')
                plt.suptitle(title, fontsize = '30')
                fig.tight_layout()
                fig.subplots_adjust(top=0.95)


    def PlotPredictions(self, dataFrame, dates, prediction, y_Test, title, look_back = 0):
        num_date = len(dates)
        fig, axes = plt.subplots(num_date,1,figsize=(24, num_date*5) )
        plt.suptitle(title, fontsize = '25')
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)

        for i in range(num_date):
            print(i)
            if i == 0: l = 0
            ind = dataFrame.loc[dates[i]].index[look_back:]
            print(ind.shape)
            print(ind)
            axes.flat[i].plot(ind, y_Test[l:l+len(ind)], color = 'blue', alpha = 0.6, label = 'True value')
            axes.flat[i].plot(ind, prediction[l:l+len(ind)], color = 'red', alpha = 0.6, label = 'Predicted value')
            # axes.flat[i].plot(ind, y_Test, color = 'blue', alpha = 0.6, label = 'True value')
            # axes.flat[i].plot(ind, prediction, color = 'red', alpha = 0.6, label = 'Predicted value')
            
            axes.flat[i].legend()
            l = len(ind)
        
        plt.show()