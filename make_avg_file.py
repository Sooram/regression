import pandas as pd
import numpy as np
from datetime import datetime

data_1 = pd.read_csv("file_path")
data_1 = data_1.replace("[-11059] No Good Data For Calculation", np.nan)

col_names = list(data_1)
col_names.remove('Unnamed: 0')
col_names.remove('tag')
col_names.remove('y')
col_names.remove('y_2')

def make_avg(start_hour, end_hour, data):
    indices = [i for i, s in enumerate(list(data['tag'])) if ' 9:00' == s[-5:]]

    avg_data = pd.DataFrame(data=data.iloc[indices])
    avg_data = avg_data.reset_index()

    for row, date in enumerate(list(avg_data['tag'])):
        print(row)
        from_time = datetime.strptime(date, '%Y-%m-%d %H:%M').replace(hour=start_hour, minute=0)
        to_time = datetime.strptime(date, '%Y-%m-%d %H:%M').replace(hour=end_hour, minute=0)

        around9_indices = [i for i, s in enumerate(list(data['tag'])) if (
                    (from_time <= datetime.strptime(s, '%Y-%m-%d %H:%M')) & (
                        to_time >= datetime.strptime(s, '%Y-%m-%d %H:%M')))]

        for x in col_names:
            val_list = [data[x].iloc[i] for i in around9_indices]
            avg_data.set_value(row, x, np.nanmean(np.array(val_list).astype(np.float)))
            
    avg_data.to_csv('./data/avg.csv')
    return avg_data

df_data = make_avg(5, 6, data_1)






