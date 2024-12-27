def variables_to_specify_SimReuse():

       from datetime import datetime, timedelta

        #load dataset
       from darts.datasets import WeatherDataset
       series = WeatherDataset().load()
       df = series.pd_dataframe().reset_index()


        #identify your feature columns (features should be normalized)
       columns_to_normalize = ['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)',
                               'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
                               'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
                               'wd (deg)', 'rain (mm)', 'raining (s)', 'SWDR (W/m²)',
                               'PAR (µmol/m²/s)', 'max. PAR (µmol/m²/s)', 'Tlog (degC)', 'CO2 (ppm)']

        #identify the target column
       target_col = 'T (degC)'

        # identify the forecast average column name
       forecast_avg_target_col_name = 'forecast_daily_T (degC)_avg'

       # identify the average target column name
       avg_target_col_name = 'daily_T (degC)_avg'

        # identify the number of data points in one day
       No_of_datapoints_in_one_day = 144

        # identify the start and end date
       start_date = datetime(2020, 1, 1)
       end_date = datetime(2020, 12, 31)
       delta = timedelta(days=1)

        # identify the number of days in a month
       one_month_days = 31


       # out_columns = ['Training dataset', 'Testing dataset', 'mae', 'mse', 'rmse', 'r2', 'mape', 'training_time',
       #                'Testing Error', 'testing_time']

        # columns that you want to drop in the rest of the code
       drop_columnss = ['T (degC)', 'year', 'month', 'day', 'hour', 'minute', 'Date Time']

        # identify the segment lengths that you want to exmine in the periodical model retraining approach
       # we have 144 data points in each day (24*6(we have 6 data points in each hour)=144)
       # 5 days: (5*144) = 720
       # 15 days: 2160
       # 30 days:4320
       # 45 days:6480
       # 60 days: 8640
       # 75 days:10800
       # 90 days:12960
       windows = [720, 2160, 4464, 6480, 8640, 10800, 12960]

        # identify the segment length of one month
       one_month_window_size = 4464

       return df, columns_to_normalize, target_col, forecast_avg_target_col_name, avg_target_col_name, No_of_datapoints_in_one_day, start_date, end_date, delta, one_month_days, drop_columnss, windows, one_month_window_size


