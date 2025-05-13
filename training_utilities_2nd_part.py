import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_squared_error
import statistics
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import re
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import timeit
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import keras_tuner as kt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import statistics
import timeit
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import joblib
import numpy as np
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import statsmodels.api as sm
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import tensorflow as tf
import joblib
import os
import random
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
import math
warnings.filterwarnings("ignore")


def find_all_seasonality_periods_peaks(daily_df_avg):
    dff = daily_df_avg.iloc[:, 0]
    if isinstance(dff, pd.Series):
        signal = dff.dropna().values
    else:
        signal = dff.iloc[:, 0].dropna().values  # Select the first column if it's a DataFrame

    acf_values = sm.tsa.acf(signal, nlags=len(signal) // 2, fft=True)

    peaks, _ = find_peaks(acf_values, height=0.1)

    seasonality_periods_acf = peaks

    print(f"Detected seasonality periods (ACF): {seasonality_periods_acf}")
    return seasonality_periods_acf

############################################################################################
def time_series_decomposition(daily_df_avg, start_date , seasonality_periods_acf):
    dff = daily_df_avg.iloc[:, 0]
    signal = dff.values

    ts = pd.Series(signal,
                   index=pd.date_range(start=start_date, periods=len(dff), freq='D'))  # Adjust start date if needed

    decomposition = seasonal_decompose(ts, model='additive', period=seasonality_periods_acf)  # Adjust period based on expected seasonality
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid


######################################################################
def segment_daily_df_avg(daily_df_avg, avg_target_col_name, acf_segment_length):
    target_values = daily_df_avg[avg_target_col_name].values
    num_segments = len(target_values) // acf_segment_length
    truncated_values = target_values[:num_segments * acf_segment_length]
    segmented_data = truncated_values.reshape(num_segments, acf_segment_length).T
    segmented_target_daily_df_avg = pd.DataFrame(segmented_data, columns=[f"Segment_{i}" for i in range(num_segments)])
    return segmented_target_daily_df_avg

######################################################################
def map_all_chains(similarity_dict):
    result = {}
    visited = set()
    for key in similarity_dict:
        if key in visited:
            continue

        first_key = key
        current_value = similarity_dict[key]

        while current_value in similarity_dict:
            visited.add(current_value)  # Mark keys as visited
            current_value = similarity_dict[current_value]

        result[first_key] = current_value  # Store the first key to last value
    return result

######################################################################
def find_most_similar_columns_wass(df):
    most_similar_dict_wass = {}
    for j in range(1, df.shape[1]):
        current_col = df.iloc[:, j].dropna().values
        best_match = None
        min_distance = float('inf')
        for k in range(j):
            prev_col = df.iloc[:, k].dropna().values
            min_length = min(len(current_col), len(prev_col))
            if min_length == 0:
                continue
            distance = wasserstein_distance(prev_col[:min_length], current_col[:min_length])
            if distance < min_distance:
                min_distance = distance
                best_match = k
        most_similar_dict_wass[j] = best_match
    filtered_most_similar_dict_wass = {key: value for key, value in most_similar_dict_wass.items() if value != key - 1}
    return filtered_most_similar_dict_wass

######################################################################
def find_most_similar_columns_tvd(df):
    def total_variation_distance(p, q):
        return 0.5 * np.sum(np.abs(p - q))
    most_similar_dict_tvd = {}
    for j in range(1, df.shape[1]):  # Start from the second column
        current_col = df.iloc[:, j].dropna().values  # Remove NaN values
        best_match = None
        min_distance = float('inf')
        for k in range(j):  # Compare with all previous columns
            prev_col = df.iloc[:, k].dropna().values  # Remove NaN values

            min_length = min(len(current_col), len(prev_col))
            if min_length == 0:
                continue  # Skip empty columns

            # Normalize the distributions to sum to 1
            p = prev_col[:min_length] / np.sum(prev_col[:min_length])
            q = current_col[:min_length] / np.sum(current_col[:min_length])

            distance = total_variation_distance(p, q)

            if distance < min_distance:
                min_distance = distance
                best_match = k
        most_similar_dict_tvd[j] = best_match
    filtered_most_similar_dict_tvd = {key: value for key, value in most_similar_dict_tvd.items() if
                                       value != key - 1}
    return filtered_most_similar_dict_tvd

####################################################################################
def train_model_with_hyperparametertuning(X_train, y_train):
   param_dist = {
       'n_estimators': [100, 200, 500],  # Number of trees in the forest
       'max_features': ['sqrt', 'log2', None],  # Number of features to consider
       'max_depth': [10, 20, 30, None],  # Max depth of each tree
       'min_samples_split': [2, 5, 10],  # Min samples to split a node
       'min_samples_leaf': [1, 2, 4],  # Min samples at a leaf node
       'bootstrap': [True, False]  # Whether to use bootstrapping
   }
   rf_regressor = RandomForestRegressor(random_state=42, verbose=False)
   random_search = RandomizedSearchCV(
       estimator=rf_regressor,
       param_distributions=param_dist,
       n_iter=10,  # Number of parameter settings to try
       cv=5,  # 5-fold cross-validation
       verbose=2,  # Output progress
       random_state=42,
       n_jobs=-1  # Use all available cores
   )

   start_train_time = timeit.default_timer()
   random_search.fit(X_train, y_train)
   train_time = timeit.default_timer() - start_train_time

   return random_search.best_estimator_, train_time

#################################################################################
def preprocess_feature_names(df):
    pattern = re.compile(r'[\[\],<>]')  # Define invalid characters to replace
    if isinstance(df, pd.DataFrame):
        df.columns = [pattern.sub('_', str(col)) for col in df.columns]
        return df
    elif isinstance(df, pd.Series):
        return df

################################################################################
def train_model2_with_hyperparametertuning(X_train, y_train):
    # Define the hyperparameter grid to search
    param_dist = {
        'n_estimators': [100, 200, 300],             # Number of boosting rounds
        'learning_rate': [0.01, 0.1, 0.3],           # Step size shrinkage
        'max_depth': [3, 5, 7, 10],                  # Maximum depth of a tree
        'min_child_weight': [1, 3, 5],               # Minimum sum of instance weight needed in a child
        'subsample': [0.6, 0.8, 1.0],                # Subsample ratio of the training instances
        'colsample_bytree': [0.6, 0.8, 1.0],         # Subsample ratio of columns when constructing each tree
        'gamma': [0, 0.1, 0.3],                      # Minimum loss reduction to make a further partition
        'reg_alpha': [0, 0.1, 1],                    # L1 regularization term
        'reg_lambda': [0.1, 1, 10]                   # L2 regularization term
    }
    xgb_regressor = XGBRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=xgb_regressor,
        param_distributions=param_dist,
        n_iter=10,                  # Number of parameter settings to try
        cv=5,                       # 5-fold cross-validation
        verbose=2,                  # Output progress
        random_state=42,
        n_jobs=-1                   # Use all available cores
    )
    start_train_time = timeit.default_timer()
    random_search.fit(X_train, y_train)
    train_time = timeit.default_timer() - start_train_time
    return random_search.best_estimator_, train_time

################################################################################
def new_es_forecasting(daily_df_avg, segment_size, avg_target_col_name, forecast_avg_target_col_name):
    set_seed(42)
    n_segments = daily_df_avg.shape[0] // segment_size
    forecast_results = []
    for i in range(n_segments - 1):  # -1 to avoid predicting beyond dataset
        train_data = daily_df_avg.iloc[: (i + 1) * segment_size]  # Use data up to the current segment
        model = ExponentialSmoothing(train_data[avg_target_col_name], trend='add', seasonal=None, damped_trend=True)
        fit = model.fit()
        forecast = fit.forecast(steps=segment_size)
        forecast_dates = daily_df_avg.index[(i + 1) * segment_size: (i + 2) * segment_size]  # Corresponding dates
        forecast_results.append(pd.DataFrame({forecast_avg_target_col_name: forecast.values}, index=forecast_dates))
    forecast_daily_df_avg = pd.concat(forecast_results)
    initial_actual = daily_df_avg.iloc[:segment_size][[avg_target_col_name]].rename(columns={avg_target_col_name: forecast_avg_target_col_name})
    forecast_daily_df_avg = pd.concat([initial_actual, forecast_daily_df_avg])
    if daily_df_avg.shape[1] > 1:
        additional_columns = daily_df_avg.iloc[:, 1:]
        forecast_daily_df_avg = forecast_daily_df_avg.join(additional_columns, how='left')

    return forecast_daily_df_avg

################################################################################
def prepare_data(df, target_column, time_steps, drop_columnss):
    df = df.select_dtypes(include=[np.number])

    X, y = [], []
    feature_columns = [col for col in df.columns if col != target_column and col not in drop_columnss]
    df_values = df.to_numpy()

    for i in range(len(df_values) - time_steps):
        X.append(df_values[i:i + time_steps, df.columns.get_indexer(feature_columns)])
        y.append(df_values[i + time_steps, df.columns.get_loc(target_column)])  # Target

    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    return X, y
################################################################################
################################################################################
### training GRU

def train_model3_with_hyperparametertuning(X_train, y_train):
    #GRU code
    epochs = 10
    batch_size = 32
    timesteps = X_train.shape[1]
    features = X_train.shape[2]

    model = Sequential([
        GRU(units=64, return_sequences=False, input_shape=(timesteps, features)),
        Dense(units=32, activation='relu'),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    train_time = 0
    return model, train_time

## training lstm
def train_model3_with_hyperparametertuning(X_train, y_train):
    epochs = 10
    batch_size = 32
    timesteps = X_train.shape[1]
    features = X_train.shape[2]

    # Define the LSTM model
    model = Sequential([
        LSTM(units=64, return_sequences=False, input_shape=(timesteps, features)),
        Dense(units=32, activation='relu'),
        Dense(units=1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    train_time = 0

    return model, train_time
################################################################################
def new_copied_lstm_reuse_with_hptuning_no_while_loop_with_drift(similarity_dict, stationary_model, len_of_training_data_of_stationary_model,
                                        df, forecast_approach, target_col, drop_columnss, time_steps, seasonality_periods_acf, No_of_datapoints_in_one_day, drifted_segments_ls):
    set_seed(42)

    out_columns = ['mse', 'mae', 'training_time',
                   'Testing Error', 'testing_time', 'stationary_model_testing_error', 'stationary_model_testing_time',
                   'stationary_model_mse', 'stationary_model_mae',
                   'reused_model_testing_error', 'reused_model_testing_time',
                   'reused_model_mse', 'reused_model_mae']
    periodical_total_time = 0
    Informed_total_time = 0
    ls_eval_df_monthly = []
    mean_ls_monthly = []
    window = seasonality_periods_acf*No_of_datapoints_in_one_day
    models_ls = []
    # indices_ls = []
    eval_df_monthly = pd.DataFrame(columns=out_columns)
    next_month_of_similar_months_ind = []

    mse_ls = []
    mae_ls = []
    train_time_ls = []

    drift_latest_model = stationary_model
    test_error4_ls = []
    test_error_mae4_ls = []
    test_time4_ls = []

    total_reuse_time = 0
    reuse_trained_and_stored_models_count = 0
    ml_storage_ls = []
    for i in range(window, len(df), window):
        start_periodical = timeit.default_timer()
        start_train11 = timeit.default_timer()

        print("window is: ", i)
        train = df[i - window:i]
        test = df[i: i + window]
        if len(test) < window-1:
            break

        train = train.reset_index().drop(columns=['index'])
        test = test.reset_index().drop(columns=['index'])
        train = preprocess_feature_names(train)
        test = preprocess_feature_names(test)
        train, labels = prepare_data(train, target_col, time_steps, drop_columnss)
        X_test, y_test = prepare_data(test, target_col, time_steps, drop_columnss)
        model, train_time = train_model3_with_hyperparametertuning(train, labels)
        train_time_ls = timeit.default_timer() - start_train11

        start_test_time = timeit.default_timer()
        y_pred = model.predict(X_test, verbose=0)
        y_pred = y_pred[:, 0]

        test_error = mean_squared_error(y_test, y_pred)
        test_error_mae = mean_absolute_error(y_test, y_pred)
        result = [test_error, test_error_mae]
        test_time = timeit.default_timer() - start_test_time
        periodical_total_time += (timeit.default_timer() - start_periodical)
        if forecast_approach == "ES": # forecasting_approach == "ES"
            month_index = math.floor(i / window) - 1
            drift_ls_index = month_index
        elif forecast_approach == "SA": # forecasting_approach == "SA"
            month_index = math.floor(i / window)
            drift_ls_index =month_index
            print("i/window is : ", i/window)

        if month_index in similarity_dict:
            sub_reuse_time_start = timeit.default_timer()
            similar_month_index = similarity_dict[month_index]
            print("similar_month_index is : ", similar_month_index)

            if similar_month_index in similarity_dict:
                if forecast_approach == "ES":  # forecasting_approach == "ES"
                    previous_model_i = (month_index+1)*window
                elif forecast_approach == "SA":  # forecasting_approach == "SA"
                    previous_model_i = (month_index) * window
                print("previous_model_i is : ", previous_model_i)
                train3 = df[previous_model_i - window:previous_model_i]
                test3 = df[previous_model_i: previous_model_i + window]
                train3 = train3.reset_index().drop(columns=['index'])
                test3 = test3.reset_index().drop(columns=['index'])
                train3 = preprocess_feature_names(train3)
                test3 = preprocess_feature_names(test3)
                train3, labels3 = prepare_data(train3, target_col, time_steps, drop_columnss)
                X_test3, y_test3 = prepare_data(test3, target_col, time_steps, drop_columnss)
                # print("train columns:", train[:10])
                print("math.floor(previous_model_i/window) is: ", math.floor(previous_model_i/window))
                print("len(models_ls) is:", len(models_ls))
                model3, train_time3 = train_model3_with_hyperparametertuning(X_test3, y_test3)
                reuse_trained_and_stored_models_count+=1
                ml_storage_ls.append(compute_model_storage(model3,3))
                models_ls[similar_month_index] = model3
                models_ls.append(model3)

                start_test_time2 = timeit.default_timer()
                y_pred2 = model3.predict(X_test, verbose=0)
                y_pred2 = y_pred2[:, 0]

                test_error2 = mean_squared_error(y_test, y_pred2)
                test_error_mae2 = mean_absolute_error(y_test, y_pred2)
                result2 = [test_error2, test_error_mae2]
                test_time2 = timeit.default_timer() - start_test_time2
            else:
                next_month_of_similar_months_ind.append(similar_month_index)
                print("month_index: ", month_index)
                print('\n')
                model2 = models_ls[similar_month_index]
                models_ls.append(model2)
                start_test_time2 = timeit.default_timer()
                y_pred2 = model2.predict(X_test, verbose=0)
                y_pred2 = y_pred2[:, 0]

                test_error2 = mean_squared_error(y_test, y_pred2)
                test_error_mae2 = mean_absolute_error(y_test, y_pred2)
                result2 = [test_error2, test_error_mae2]
                test_time2 = timeit.default_timer() - start_test_time2

            sub_reuse_time = timeit.default_timer() - sub_reuse_time_start

            # drift

            if (drift_ls_index > 0) and (drift_ls_index in drifted_segments_ls) and (forecast_approach == "ES"):
                result4 = result
                test_error4 = test_error
                test_error_mae4 = test_error_mae
                test_time4 = test_time
                drift_latest_model, drfit_model_train_time = train_model3_with_hyperparametertuning(X_test, y_test)
                test_error4_ls.append(test_error4)
                test_error_mae4_ls.append(test_error_mae4)
                test_time4_ls.append(test_time4)
            elif (drift_ls_index > 1) and (drift_ls_index in drifted_segments_ls) and (forecast_approach == "SA"):
                result4 = result
                test_error4 = test_error
                test_error_mae4 = test_error_mae
                test_time4 = test_time
                drift_latest_model, drfit_model_train_time = train_model3_with_hyperparametertuning(X_test, y_test)
                test_error4_ls.append(test_error4)
                test_error_mae4_ls.append(test_error_mae4)
                test_time4_ls.append(test_time4)
            elif drift_ls_index not in drifted_segments_ls:
                start_test_time4 = timeit.default_timer()
                y_pred4 = drift_latest_model.predict(X_test, verbose=0)
                y_pred4 = y_pred4[:, 0]
                test_error4 = mean_squared_error(y_test, y_pred4)
                test_error_mae4 = mean_absolute_error(y_test, y_pred4)
                result4 = [test_error4, test_error_mae4]
                test_time4 = timeit.default_timer() - start_test_time4

                test_error4_ls.append(test_error4)
                test_error_mae4_ls.append(test_error_mae4)
                test_time4_ls.append(test_time4)
        else:
            sub_reuse_time_start = timeit.default_timer()
            model2, train_time = train_model3_with_hyperparametertuning(train, labels)
            ml_storage_ls.append(compute_model_storage(model2,3))
            reuse_trained_and_stored_models_count+=1
            models_ls.append(model2)
            start_test_time2 = timeit.default_timer()
            y_pred2 = model2.predict(X_test, verbose = 0)
            y_pred2 = y_pred2[:, 0]
            test_error2 = mean_squared_error(y_test, y_pred2)
            test_error_mae2 = mean_absolute_error(y_test, y_pred2)
            result2 = [test_error2, test_error_mae2]
            test_time2 = timeit.default_timer() - start_test_time2
            sub_reuse_time = timeit.default_timer() - sub_reuse_time_start
        total_reuse_time += sub_reuse_time
        if len(result2) == 0:
            result2 = ["no result", "no result"]

        # stationary_model1 results
        if i < len_of_training_data_of_stationary_model:
            test_error3 = "no result"
            test_time3 = "no result"
            result3 = ["no result", "no result"]
        elif i >= len_of_training_data_of_stationary_model:
            test_error3 = 0
            test_time3 = 0
            result3 = []
            start_test_time3 = timeit.default_timer()
            y_pred3 = stationary_model.predict(X_test, verbose = 0)
            y_pred3 = y_pred3[:, 0]
            test_error3 = mean_squared_error(y_test, y_pred3)
            test_error_mae3 = mean_absolute_error(y_test, y_pred3)
            result3 = [test_error3, test_error_mae3]
            test_time3 = timeit.default_timer() - start_test_time3

        # random baseline
        X_train = train
        y_train = labels

        # random baseline approach 1
        min_val = y_train.min()
        max_val = y_train.max()
        n_predictions = len(X_test)  # Number of samples in the test set
        random_baseline_predictions = np.random.uniform(min_val, max_val, n_predictions)

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, random_baseline_predictions)
        mae = mean_absolute_error(y_test, random_baseline_predictions)
        # mse_ls.append(mse)

        if len(result2) != 0:
            mse_ls.append(mse)
            mae_ls.append(mae)
        else:
            pass

        eval_df_monthly.loc[len(eval_df_monthly)] =result + [train_time,
                                                                                                  test_error,
                                                                                                  test_time] + [
                                                        test_error3, test_time3] + result3 + [test_error2,
                                                                                              test_time2] + result2
        print("\n")
    print("\n")
    print("total_reuse_time is : ", total_reuse_time)
    print("reuse_trained_and_stored_models_count is : ", reuse_trained_and_stored_models_count)
    avg_ml_storage = sum(ml_storage_ls) / len(ml_storage_ls)
    print("avg_ml_storage: ", avg_ml_storage)
    print("\n")

    eval_df_monthly2 = eval_df_monthly[eval_df_monthly['stationary_model_testing_error'] != 'no result']
    print('\n')

    filtered_eval_df = eval_df_monthly2[eval_df_monthly2['reused_model_testing_error'] != 0]

    mean_stationary_model_testing_error_of_months_in_similar_dict = filtered_eval_df[
        'stationary_model_mse'].mean()
    mean_periodical_testing_error_of_months_in_similar_dict = filtered_eval_df['mse'].mean()
    mean_reused_model_testing_error_of_months_in_similar_dict = filtered_eval_df['reused_model_mse'].mean()
    mean_drift_model_testing_error_of_months_in_similar_dict = statistics.mean(test_error4_ls)

    print('mean_stationary_model_testing_error_of_months_in_similar_dict is: ',
          mean_stationary_model_testing_error_of_months_in_similar_dict)
    print('mean_periodical_testing_error_of_months_in_similar_dict is: ',
          mean_periodical_testing_error_of_months_in_similar_dict)
    print('mean_reused_model_testing_error_of_months_in_similar_dict is: ',
          mean_reused_model_testing_error_of_months_in_similar_dict)
    print('mean_drift_model_testing_error_of_months_in_similar_dict is: ',
          mean_drift_model_testing_error_of_months_in_similar_dict)

    print("\n")
    print("MAE results are: ")
    mean_stationary_model_testing_error_of_months_in_similar_dict = filtered_eval_df[
        'stationary_model_mae'].mean()
    mean_periodical_testing_error_of_months_in_similar_dict = filtered_eval_df['mae'].mean()
    mean_reused_model_testing_error_of_months_in_similar_dict = filtered_eval_df['reused_model_mae'].mean()
    mean_drift_model_testing_error_of_months_in_similar_dict = statistics.mean(test_error_mae4_ls)

    print('mae_mean_stationary_model_testing_error_mae_of_months_in_similar_dict is: ',
          mean_stationary_model_testing_error_of_months_in_similar_dict)
    print('mae_mean_periodical_testing_error_mae_of_months_in_similar_dict is: ',
          mean_periodical_testing_error_of_months_in_similar_dict)
    print('mae_mean_reused_model_testing_error_mae_of_months_in_similar_dict is: ',
          mean_reused_model_testing_error_of_months_in_similar_dict)
    print('mae_mean_drift_model_testing_error_mae_of_months_in_similar_dict is: ',
          mean_drift_model_testing_error_of_months_in_similar_dict)

    average = sum(mse_ls) / len(mse_ls)
    average_mae = sum(mae_ls) / len(mae_ls)
    print("Average mse in random model is: ", average)
    print("average mae in random model is: ", average_mae)
    return eval_df_monthly2, eval_df_monthly, avg_ml_storage

################################################################################
def periodical_lstm_training(df, target_column, time_steps, windows, drop_columnss):
    periodical_total_time_start = timeit.default_timer()
    set_seed(42)
    mean_mse_per_window = []
    mean_mae_per_window = []
    sub_times = []
    storage_mb_ls = []
    for window in windows:
        sub_time_val = 0
        start_time1 = timeit.default_timer()
        mse_ls = []
        mae_ls = []
        print("window size is : ", window)
        for i in range(window, len(df), window):
            train = df[i - window:i]
            test = df[i: i + window]
            if len(test) < window - 1:
                break
            train = train.reset_index().drop(columns=['index'])
            test = test.reset_index().drop(columns=['index'])
            train = preprocess_feature_names(train)
            test = preprocess_feature_names(test)
            train, labels = prepare_data(train, target_column, time_steps, drop_columnss)
            X_test, y_test = prepare_data(test, target_column, time_steps, drop_columnss)
            model, train_time = train_model3_with_hyperparametertuning(train, labels)
            storage_mb_ls.append(compute_model_storage(model, 3))

            start_test_time = timeit.default_timer()
            y_pred = model.predict(X_test, verbose=0)
            y_pred = y_pred[:, 0]

            test_error = mean_squared_error(y_test, y_pred)
            test_error_mae = mean_absolute_error(y_test, y_pred)
            result = [test_error, test_error_mae]
            test_time = timeit.default_timer() - start_test_time
            total_reuse_time_start = timeit.default_timer()
            mse_ls.append(test_error)
            mae_ls.append(test_error_mae)
        end_time1 = timeit.default_timer() - start_time1
        sub_times.append(end_time1)
        if len(mse_ls) > 2:
            mean_mse_per_window.append(statistics.mean(mse_ls))
            mean_mae_per_window.append(statistics.mean(mae_ls))
        elif len(mse_ls) == 1:
            mean_mse_per_window.append(mse_ls[0])
            mean_mae_per_window.append(mae_ls[0])
        elif len(mse_ls) == 0:
            pass

    min_value = min(mean_mse_per_window)
    min_index = mean_mse_per_window.index(min_value)
    print("mean mse per window is : ", mean_mse_per_window)
    print(f"Min mse value: {min_value}, Index of optimal segment length: {min_index}, optimal segment length is: {windows[min_index]}")
    print("optimal MSE seg len time is:", sub_times[min_index])

    print("\n")
    min_mae_value = min(mean_mae_per_window)
    min_mae_index = mean_mae_per_window.index(min_mae_value)
    print("mean mae per window is : ", mean_mae_per_window)
    print(f"Min mae value: {min_mae_value}, Index: {min_mae_index}, optimal segment length is: {windows[min_index]}")
    print("\n")
    print("train time is : ", train_time)
    print("test time is : ", test_time)
    print("optimal MAE seg len time is:", sub_times[min_mae_index])

    avg_storage_mb = sum(storage_mb_ls)/len(storage_mb_ls)
    print("average model storage :", avg_storage_mb)
    periodical_total_time = timeit.default_timer()-periodical_total_time_start
    print("periodical total time is :", periodical_total_time)
    return mean_mse_per_window, min_index, mean_mae_per_window

################################################################################
def new_copied_lstm_statinary(df, optimal_window_len, target_col, drop_columnss, time_steps):
    set_seed(42)
    total_time_start = timeit.default_timer()
    X, y = prepare_data(df, target_col, time_steps, drop_columnss)
    X_train, y_train = X[0:optimal_window_len], y[0:optimal_window_len]
    X_test, y_test = X[optimal_window_len:], y[optimal_window_len:]
    stationary_model, train_time = train_model3_with_hyperparametertuning(X_train, y_train)
    total_y_pred = stationary_model.predict(X_test, verbose=0)
    total_y_pred = total_y_pred[:, 0]

    import numpy as np


    total_test_error = mean_squared_error(y_test,total_y_pred)
    total_test_error_mae = mean_absolute_error(y_test,total_y_pred)
    print("total_test_error is :", total_test_error)
    print("total_test_error_mae is: ", total_test_error_mae)
    total_time = timeit.default_timer() - total_time_start
    print("total_time is: ", total_time)
    print("train time is : ", train_time)
    storage_mb = compute_model_storage(stationary_model,3)
    print("model storage is :", storage_mb)
    return stationary_model


################################################################################
def lstm_informed_update(stationary_model,df, target_col, drop_columnss,time_steps, seasonality_periods_acf,No_of_datapoints_in_one_day, drifted_segments_ls):
    set_seed(42)
    Informed_total_time_start = timeit.default_timer()
    window = seasonality_periods_acf * No_of_datapoints_in_one_day
    models_ls = []
    drift_latest_model = stationary_model
    test_error4_ls = []
    test_error_mae4_ls = []
    ml_storage_ls = []
    for i in range(window, len(df), window):
        print("window is: ", i)
        train = df[i - window:i]
        test = df[i: i + window]
        if len(test) < window - 1:
            break
        train = train.reset_index().drop(columns=['index'])
        test = test.reset_index().drop(columns=['index'])
        train = preprocess_feature_names(train)
        test = preprocess_feature_names(test)
        train, labels = prepare_data(train, target_col, time_steps, drop_columnss)
        X_test, y_test = prepare_data(test, target_col, time_steps, drop_columnss)
        drift_ls_index = math.floor(i/window)
        if drift_ls_index == 1:
            drift_latest_model, train_time = train_model3_with_hyperparametertuning(train, labels)
            ml_storage_ls.append(compute_model_storage(drift_latest_model,3))
            y_pred4 = drift_latest_model.predict(X_test, verbose=0)
            test_error4 = mean_squared_error(y_test, y_pred4)
            test_error_mae4 = mean_absolute_error(y_test, y_pred4)
            test_error4_ls.append(test_error4)
            test_error_mae4_ls.append(test_error_mae4)
        elif (drift_ls_index > 1) and (drift_ls_index in drifted_segments_ls):
            y_pred4 = drift_latest_model.predict(X_test, verbose=0)
            test_error4 = mean_squared_error(y_test, y_pred4)
            test_error_mae4 = mean_absolute_error(y_test, y_pred4)
            test_error4_ls.append(test_error4)
            test_error_mae4_ls.append(test_error_mae4)
            drift_latest_model, drfit_model_train_time = train_model3_with_hyperparametertuning(X_test, y_test)
            ml_storage_ls.append(compute_model_storage(drift_latest_model,3))
        elif drift_ls_index not in drifted_segments_ls:
            y_pred4 = drift_latest_model.predict(X_test, verbose=0)
            y_pred4 = y_pred4[:, 0]
            test_error4 = mean_squared_error(y_test, y_pred4)
            test_error_mae4 = mean_absolute_error(y_test, y_pred4)
            test_error4_ls.append(test_error4)
            test_error_mae4_ls.append(test_error_mae4)
    Informed_total_time = timeit.default_timer() - Informed_total_time_start
    print("Informed total_time: ", Informed_total_time)
    avg_ml_storage = sum(ml_storage_ls) / len(ml_storage_ls)
    print("avg_ml_storage: ", avg_ml_storage)

############################################################
def convert_time(df, date_col_name):
    df['year'] = df[date_col_name].dt.year
    df['month'] = df[date_col_name].dt.month
    df['day'] = df[date_col_name].dt.day
    df['hour'] = df[date_col_name].dt.hour
    df['minute'] = df[date_col_name].dt.minute
    return df
############################################################
import io

def compute_model_storage(model, model_version="default"):

    model_type = type(model).__name__

    if (model_type in ["Sequential", "Model"]) or (model_version==3):  # For Keras LSTM models
        total_params = sum(tf.size(p).numpy() for p in model.trainable_variables)  # Corrected parameter count
        dtype_size = tf.keras.backend.floatx()  # Check default dtype ('float32' or 'float64')
        bytes_per_param = 4 if dtype_size == 'float32' else 8  # Float32 = 4 bytes, Float64 = 8 bytes

        storage_bytes = total_params * bytes_per_param  # Total storage in bytes

    elif (model_type in ["RandomForestClassifier", "RandomForestRegressor"]) or (model_version==1):

        with io.BytesIO() as buffer:
            joblib.dump(model, buffer)
            storage_bytes = buffer.tell()  # The size in bytes
        
    elif (model_type in ["XGBClassifier", "XGBRegressor"])or (model_version==2):  # For XGBoost models
        temp_file = f"xgb_model_{model_version}.model"
        model.save_model(temp_file)  # Save the model temporarily
        storage_bytes = os.path.getsize(temp_file)  # Get file size in bytes
        os.remove(temp_file)  # Remove the temporary file

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    storage_mb = storage_bytes / (1024 ** 2)  # Convert to MB

    print(f"Model Type: {model_type}")
    print(f"Storage Required: {storage_mb:.2f} MB")

    return storage_mb

############################################################
# data drift detection

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)))
############################################################
def two_proportion_ztest(p1, p2, n1, n2):
    p = (p1 * n1 + p2 * n2) / (n1 + n2)

    if p <= 0 or p >= 1:
        return 0.0, 1.0

    z_num = (p2 - p1)
    z_den = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if z_den == 0:
        return 0.0, 1.0

    z_value = z_num / z_den
    p_value = 2.0 * (1.0 - norm.cdf(abs(z_value)))
    return z_value, p_value

#####################################################################
def get_elect_daily_avg(df, No_of_datapoints_in_one_day, target_col, avg_target_col_name):
    daily_avg = []
    for i in range(0, len(df), No_of_datapoints_in_one_day):
        sub_df = df[i:i + No_of_datapoints_in_one_day]
        mean = sub_df[target_col].mean()
        daily_avg.append(mean)
    daily_df_avg = pd.DataFrame(daily_avg, columns=[avg_target_col_name])
    daily_df_avg

    return daily_df_avg

#############################################################
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

##############################################################################################################################
def time_series_cv_error_rate(series, order=(1, 0, 0), folds=5):
    n = len(series)
    if n < folds:
        folds = max(1, n // 2)

    fold_size = max(2, n // folds)
    errors = []

    for i in range(folds):
        end_train = (i + 1) * fold_size if (i < folds - 1) else n
        train_data = np.array(series[:end_train]).flatten()

        if i < folds - 1:
            test_data = np.array(series[end_train: end_train + fold_size]).flatten()
        else:
            test_data = np.array([])

        if len(train_data) <= order[0] or len(test_data) == 0:
            print(f"Skipping fold {i}: Insufficient training or test data.")
            continue

        print(f"Fold {i}: Train size={len(train_data)}, Test size={len(test_data)}")

        try:
            model = ARIMA(train_data, order=order)
            fitted = model.fit()

            forecast = fitted.forecast(steps=len(test_data))

            fold_error = mean_absolute_percentage_error(test_data, forecast)
            errors.append(fold_error)
        except Exception as e:
            print(f"Error in fold {i}: {e}")
            continue
    return np.mean(errors) if errors else 0.0
#############################################################################

def detect_drift_univariate(df, target_col, window_lengths, arima_order=(1, 0, 0)):
    set_seed(42)
    series = df[target_col].values
    n = len(series)
    results_list = []
    for w in window_lengths:
        i = 0
        while (i + 2 * w) <= n:
            train_data = series[i: i + w]
            test_data = series[i + w: i + 2 * w]

            n1 = len(train_data)
            n2 = len(test_data)
            p1 = time_series_cv_error_rate(train_data, order=arima_order, folds=5)
            model = ARIMA(train_data, order=arima_order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=n2)
            p2 = mean_absolute_percentage_error(test_data, forecast)
            z_stat, p_val = two_proportion_ztest(p1, p2, n1, n2)
            drift_detected = (p_val < 0.05)

            results_list.append({
                'window_length': w,
                'train_period_start': i,
                'train_period_end': i + w,
                'test_period_start': i + w,
                'test_period_end': i + 2 * w,
                'p1_training_error': p1,
                'p2_testing_error': p2,
                'z_stat': z_stat,
                'p_value': p_val,
                'drift_detected': drift_detected,
                'relative_diff': (p2 - p1) / (p1 + 1e-8)
            })
            i += w  # slide the window forward by w
    drift_df = pd.DataFrame(results_list)
    return drift_df

#############################################################################
def get_seasonality_segments_and_similarities(daily_df_avg, avg_target_col_name, forecast_avg_target_col_name, seasonality_periods_acf):
    seasonality_periods_acf_ls = find_all_seasonality_periods_peaks(daily_df_avg)

    import statistics
    median_value = round(statistics.median(seasonality_periods_acf_ls))
    print("median_value is: ", median_value)
    segmented_daily_df_avg = segment_daily_df_avg(daily_df_avg, avg_target_col_name, seasonality_periods_acf)
    filtered_most_similar_dict_wass = find_most_similar_columns_wass(segmented_daily_df_avg)
    filtered_most_similar_dict_tvd = find_most_similar_columns_tvd(segmented_daily_df_avg)
    forecast_daily_df_avg = new_es_forecasting(daily_df_avg, seasonality_periods_acf, avg_target_col_name,
                                               forecast_avg_target_col_name)
    segmented_forecast_daily_df_avg = segment_daily_df_avg(forecast_daily_df_avg, forecast_avg_target_col_name,
                                                           seasonality_periods_acf)

    filtered_forecasted_most_similar_dict_wass = find_most_similar_columns_wass(segmented_forecast_daily_df_avg)

    filtered_forecasted_most_similar_dict_tvd = find_most_similar_columns_tvd(segmented_forecast_daily_df_avg)

    return seasonality_periods_acf_ls, seasonality_periods_acf, segmented_daily_df_avg, filtered_most_similar_dict_wass, filtered_most_similar_dict_tvd, forecast_daily_df_avg, segmented_forecast_daily_df_avg, filtered_forecasted_most_similar_dict_wass, filtered_forecasted_most_similar_dict_tvd

#############################################################################
def total_reduced_count_of_retrainings(similarity_dict):
    keys = list(similarity_dict.keys())
    values_set = list(set(list(similarity_dict.values())))
    trained_count = 0
    for value in values_set:
        if value in similarity_dict.keys():
            trained_count += 1

    total_reduced_training_count = len(keys) - trained_count
    return total_reduced_training_count


