import pandas as pd
from training_utilities import *

def variables_to_specify_twitter():
       df = pd.read_csv(
              '/Aiops_data_splitting_paper/code/Multivariate_time_series/twittter1/Twitterdatainsheets_preproccessed.csv')
       columns_to_normalize = [' IsReshare', ' Reach', ' RetweetCount', ' Likes',' Klout', ' Sentiment']
       target_col = ' Klout'
       forecast_avg_target_col_name = 'forecast_ Klout_avg'
       avg_target_col_name = 'Value_ Klout_avg'
       one_month_days = 31
       out_columns = ['Training dataset', 'Testing dataset', 'mae', 'mse', 'rmse', 'r2', 'mape', 'training_time',
                      'Testing Error', 'testing_time']
       drop_columnss = [target_col]+['year', 'month', 'day', 'hour', 'minute']+['Timestamp']
       return df, columns_to_normalize, target_col, forecast_avg_target_col_name, avg_target_col_name, one_month_days, out_columns, drop_columnss


