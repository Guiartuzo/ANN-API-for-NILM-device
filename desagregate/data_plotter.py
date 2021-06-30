import matplotlib.pyplot as plt


class Plotter:

    #refatorar esse método

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
            axes.flat[i].plot(ind, y_Test[l:l+len(ind)], color = 'blue', alpha = 0.6, label = 'Valor real')
            axes.flat[i].plot(ind, prediction[l:l+len(ind)], color = 'red', alpha = 0.6, label = 'Valor previsto pelo modelo')            
            axes.flat[i].legend()
            l = len(ind)
        
        plt.show()