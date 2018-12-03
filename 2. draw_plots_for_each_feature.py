#----------------------------------------------------------------------------------
# Draw line plots and box plots per each feature
#----------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------- functions
#########################################
# draw plots for each variable
# data: the whole dataframe
# folder: the path to save the plots in
# change figsize
#########################################
def draw_lineplot(data, folder):
    for i, variable in enumerate(data.columns):
        print(i)
        plt.figure(figsize=(20, 5))
        plt.plot(data[variable])
        plt.grid()
        plt.savefig("./" + folder + "/" + variable + ".png")
        plt.close()

def draw_boxplot(data, folder):
    for i, variable in enumerate(data.columns):
        print(i)
        plt.boxplot(data[variable])
        # plt.boxplot(data[variable][~np.isnan(data[variable])])    # plot ignoring NA values
        plt.savefig("./" + folder + "/" + variable + ".png")
        plt.close()

def draw_scatterplot(data, y, folder):
    for i, variable in enumerate(data.columns):
        print(i)
        plt.scatter(data[variable],y)
        plt.savefig("./" + folder + "/" + variable + ".png")
        plt.close()

#---------------------------------------------------------------------------- function calls
data_all = pd.read_csv("file_path")
list(data_all)
# drop unnecessary columns
data = data_all.drop(['tag'], axis=1)

draw_lineplot(data, "file_path")
draw_boxplot(data, "file_path")
draw_scatterplot(data, data['y'],"file_path")