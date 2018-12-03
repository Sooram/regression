#----------------------------------------------------------------------------------
# Get basic statistical information(count, mean, sd, min, max, quartile) per feature
#----------------------------------------------------------------------------------
import pandas as pd

data_path = "./data_origin/1.csv"
summary_path = "./eda/summary_statistics.csv"

data = pd.read_csv(data_path)

data.describe().to_csv(summary_path)