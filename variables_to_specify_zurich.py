import pandas as pd
from training_utilities import *

def variables_to_specify_zurich():
       from darts.datasets import WeatherDataset
       from datetime import datetime, timedelta

       df = pd.read_csv(
           '/Users/forough/PycharmProjects/mitigation3/Aiops_data_splitting_paper/code/Multivariate_time_series/dart_datasets/ElectricityConsumptionZurichDataset.csv').drop(
           'Unnamed: 0', axis=1)


       columns_to_normalize = ['Value_NE5', 'Value_NE7', 'Hr [%Hr]', 'RainDur [min]', 'StrGlo [W/m2]', 'T [°C]',
                               'WD [°]', 'WVs [m/s]', 'WVv [m/s]', 'p [hPa]']
       date_col_name = 'Timestamp'
       df = convert_timestamp(df, date_col_name)

       target_col = 'Value_NE5'

       forecast_avg_target_col_name = 'forecast_Value_NE5_avg'

       avg_target_col_name = 'Value_NE5_avg'
       date_col_name = 'Timestamp'

       No_of_datapoints_in_one_day = 96

       start_date = datetime(2015, 1, 1)
       end_date = datetime(2022, 8, 31)
       delta = timedelta(days=1)

       date_col_name = 'Timestamp'
       one_month_days = 31

       # rows_to_drop = [30, 61, 92, 123, 154, 185, 216, 247, 278, 309, 340, 346]

       out_columns = ['Training dataset', 'Testing dataset', 'mae', 'mse', 'rmse', 'r2', 'mape', 'training_time',
                      'Testing Error', 'testing_time']

       drop_columnss = [target_col]+['year', 'month', 'day', 'hour', 'minute']+['Timestamp']

       # we have 96 data points in each day (24*4(we have 4 data points in each hour)=144)
       # 5 days: 480
       # 15 days: 1440
       # 30 days: 2880
       # 45 days: 4320
       # 60 days: 5760
       # 75 days: 7200
       # 90 days: 8640

       windows = [480, 1440, 2976, 4320, 5760, 7200, 8640]

       index_of_one_month = 2
       one_month_window_size = 2976

       return df, columns_to_normalize, target_col, forecast_avg_target_col_name, avg_target_col_name, No_of_datapoints_in_one_day, start_date, end_date, delta, one_month_days, out_columns, drop_columnss, windows, index_of_one_month, one_month_window_size, date_col_name


