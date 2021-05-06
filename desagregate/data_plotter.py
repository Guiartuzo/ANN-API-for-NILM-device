import matplotlib.pyplot as plt

class Plotter:

    def PlotDataFrame(self, df, title):
        
        apps = df.columns.values
        num_apps = len(apps)
        fig, axes = plt.subplots((num_apps+1)//2,2, figsize=(24, num_apps*2) )
        
        for i, key in enumerate(apps):
                axes.flat[i].plot(df[key], alpha = 0.6)
                axes.flat[i].set_title(key, fontsize = '15')
                plt.suptitle(title, fontsize = '30')
                fig.tight_layout()
                fig.subplots_adjust(top=0.95)
        
        plt.show()