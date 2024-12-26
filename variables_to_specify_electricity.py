import pandas as pd
from training_utilities import *
from darts.datasets import WeatherDataset
from datetime import datetime, timedelta
import arff
from scipy.io import arff

def convert_day(day):
  weekday_no = str(day).replace("'","")[-1]
  return weekday_no

def convert_class(classs):
  classs = str(classs).replace("b","")
  classs = str(classs).replace("'", "")
  return classs

def variables_to_specify_electricity():
       # Load ARFF file
       with open('/Users/forough/PycharmProjects/mitigation3/Aiops_data_splitting_paper/dataset/electricity/elecNormNew.arff',
       'r') as f: data, meta = arff.loadarff(f)
       df = pd.DataFrame(data)
       df = df.drop(range(1200)).reset_index(drop=True)
       df['day'] = df['day'].apply(convert_day)
       df['class'] = df['class'].apply(convert_class)
       df['class'] = df['class'].apply(lambda x: 1 if x == 'UP' else 0)

       columns_to_normalize = []
       date_col_name = 'date'
       df = convert_timestamp(df, date_col_name)

       target_col = 'nswprice'

       forecast_avg_target_col_name = 'forecast_nswprice_avg'

       avg_target_col_name = 'nswprice_avg'

       No_of_datapoints_in_one_day = 48

       start_date = datetime(1996, 6, 1)
       end_date = datetime(1998, 12, 5)
       delta = timedelta(days=1)

       one_month_days = 31

       # rows_to_drop = [30, 61, 92, 123, 154, 185, 216, 247, 278, 309, 340, 346]

       out_columns = ['Training dataset', 'Testing dataset', 'mae', 'mse', 'rmse', 'r2', 'mape', 'training_time',
                      'Testing Error', 'testing_time']

       drop_columnss = [target_col]+['day','period','class', 'year', 'month', 'day', 'hour', 'minute']+[date_col_name]

       # 1440 is one month (30 days * 48 rows)
       # 5 days: (5*48) = 240
       # 15 days: 720
       # 30 days: 1440
       # 45 days: 2160
       # 60 days: 2880
       # 75 days: 3600
       # 90 days: 4320

       windows = [240, 720, 1488, 2160, 2880, 3600, 4320]

       index_of_one_month = 2
       one_month_window_size = 1488

       return df, columns_to_normalize, target_col, forecast_avg_target_col_name, avg_target_col_name, No_of_datapoints_in_one_day, start_date, end_date, delta, one_month_days, out_columns, drop_columnss, windows, index_of_one_month, one_month_window_size, date_col_name


