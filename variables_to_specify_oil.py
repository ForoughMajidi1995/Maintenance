import pandas as pd
from training_utilities import *






def variables_to_specify_oil():
       from darts.datasets import WeatherDataset
       from datetime import datetime, timedelta

       from darts.datasets import ETTh1Dataset

       series = ETTh1Dataset().load()
       df = series.pd_dataframe().reset_index()
       df = df[:17400]


       columns_to_normalize = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
       date_col_name = 'date'
       df = convert_timestamp(df, date_col_name)

       target_col = 'OT'

       forecast_avg_target_col_name = 'forecast_OT_avg'

       avg_target_col_name = 'OT_avg'

       No_of_datapoints_in_one_day = 24

       start_date = datetime(2016, 7, 1)
       end_date = datetime(2018, 6, 25)
       delta = timedelta(days=1)

       one_month_days = 31

       # rows_to_drop = [30, 61, 92, 123, 154, 185, 216, 247, 278, 309, 340, 346]

       out_columns = ['Training dataset', 'Testing dataset', 'mae', 'mse', 'rmse', 'r2', 'mape', 'training_time',
                      'Testing Error', 'testing_time']

       drop_columnss = [target_col]+['year', 'month', 'day', 'hour', 'minute']+[date_col_name]

       # we have 24 data points in each day (24*1)
       # 5 days: 120
       # 15 days: 360
       # 30 days: 720
       # 45 days: 1080
       # 60 days: 1440
       # 75 days: 1800
       # 90 days: 2160

       windows = [120, 360, 744, 1080, 1440, 1800, 2160]

       index_of_one_month = 2
       one_month_window_size = 744

       return df, columns_to_normalize, target_col, forecast_avg_target_col_name, avg_target_col_name, No_of_datapoints_in_one_day, start_date, end_date, delta, one_month_days, out_columns, drop_columnss, windows, index_of_one_month, one_month_window_size, date_col_name


