import matplotlib.pyplot as plt


class Plotter:

    def PlotPredictions(self, dataFrame, dates, prediction, y_Test):

        num_date = len(dates)
        fig, axes = plt.subplots(num_date,1,figsize=(24, num_date*5) )
        
        for i in range(num_date):

            if i == 0: l = 0
            ind = dataFrame.loc[dates[i]].index[0:]
            axes.flat[i].plot(ind, y_Test[l:l+len(ind)], color = 'blue', alpha = 0.6, label = 'Valor real')
            axes.flat[i].plot(ind, prediction[l:l+len(ind)], color = 'red', alpha = 0.6, label = 'Valor previsto pelo modelo')            
            axes.flat[i].legend()
            l = len(ind)

        plt.show()