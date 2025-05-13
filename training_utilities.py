from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from itertools import zip_longest
import pandas as pd
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from itertools import combinations
from itertools import combinations
import timeit
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import re
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_absolute_error
import math
import statistics
import random
from training_utilities_2nd_part import *
####################################################################################################
def count_reduced_retrainings(similarity_dict, df, drop_columnss, target_col, window ):
    model_dict = {i: "no model" for i in range(round(len(df) / window))}
    count_of_reduced_training = 0
    for i in range(window, len(df), window):
        test = df[i: i + window]
        if len(test) < window:
            break
        month_index = round(i / window) - 1
        if month_index in similarity_dict:
            count_of_reduced_training += 1
            similar_month_index = similarity_dict[month_index]
            while similar_month_index in similarity_dict:
                similar_month_index = similarity_dict[similar_month_index]

    return count_of_reduced_training
####################################################################################################
def convert_time(df, date_col_name):
    df['year'] = df[date_col_name].dt.year
    df['month'] = df[date_col_name].dt.month
    df['day'] = df[date_col_name].dt.day
    df['hour'] = df[date_col_name].dt.hour
    df['minute'] = df[date_col_name].dt.minute

    return df
####################################################################################################
def make_daily_data_for_each_month(daily_df_avg, avg_target_col, one_month_days):
    daily_col_ls = []
    for i in range(len(daily_df_avg)):
        if daily_df_avg.iloc[i, 3] == 1:
            daily_df_avg3 = daily_df_avg[i:i + one_month_days]  # we considered each month as 30 days
            daily_df_avg4 = daily_df_avg3[[avg_target_col]].copy()
            if len(daily_df_avg4) >= one_month_days:
                daily_col_ls.append(daily_df_avg4[avg_target_col].tolist())
    daily_data_for_each_month = pd.DataFrame()
    for i, lst in enumerate(daily_col_ls):
        column_name = f'month_{i}'  # Set the name for each column
        daily_data_for_each_month[column_name] = lst

    return daily_data_for_each_month
####################################################################################################
def wasserstein_dist(df2):
    most_similar_columns_wasserstein = []
    wess_similarity_dictionary = {}
    start_wess_similarity = timeit.default_timer()
    ind_count = 0
    for column in df2.columns:
        print("column is : ", column)
        min_distance = float('inf')
        most_similar_column = None
        if ind_count ==0:
            pass
        else:
            for other_column in df2.columns[0:ind_count]:
                if other_column != column:
                    emd = wasserstein_distance(df2[column], df2[other_column])
                    if emd < min_distance:
                        min_distance = emd
                        most_similar_column = other_column
            print('\n')

            most_similar_columns_wasserstein.append(most_similar_column)
            wess_similarity_dictionary[int(column.split('_')[1])] = int(most_similar_column.split('_')[1])
        ind_count+=1
    end_wess_similarity = timeit.default_timer()
    wess_similarity_time = (end_wess_similarity - start_wess_similarity)
    filtered_wess_similarity_dictionary = {k: v for k, v in wess_similarity_dictionary.items() if k >= v}
    filtered_wess_similarity_dictionary = {key: value for key, value in filtered_wess_similarity_dictionary.items() if
                                           value != key - 1}

    print(filtered_wess_similarity_dictionary)
    print("wess_similarity_time is: ", wess_similarity_time)
    return filtered_wess_similarity_dictionary
####################################################################################################
def wasserstein_dist_forecasted(df2, daily_data_for_each_month):
    most_similar_columns_wasserstein = []
    wess_similarity_dictionary = {}

    start_wess_similarity = timeit.default_timer()
    ind_count = 0
    for column in df2.columns:
        min_distance = float('inf')
        most_similar_column = None
        if ind_count ==0:
            pass
        else:
            for other_column in daily_data_for_each_month.columns[0:ind_count]:
                if other_column != column:
                    emd = wasserstein_distance(df2[column], daily_data_for_each_month[other_column])
                    if emd < min_distance:
                        min_distance = emd
                        most_similar_column = other_column

            most_similar_columns_wasserstein.append(most_similar_column)
            wess_similarity_dictionary[int(column.split('_')[1])] = int(most_similar_column.split('_')[1])
        ind_count+=1
    end_wess_similarity = timeit.default_timer()
    wess_similarity_time = (end_wess_similarity - start_wess_similarity)
    filtered_wess_similarity_dictionary = {k: v for k, v in wess_similarity_dictionary.items() if k >= v}
    filtered_wess_similarity_dictionary = {key: value for key, value in filtered_wess_similarity_dictionary.items() if
                                           value != key - 1}

    print(filtered_wess_similarity_dictionary)
    print("wess_similarity_time is: ", wess_similarity_time)
    return filtered_wess_similarity_dictionary
####################################################################################################
def total_variation_distance(p, q):
    return 0.5 * np.sum(np.abs(p - q))
####################################################################################################
def tvd(df2):
    most_similar_columns_tvd = []
    tvd_similarity_dictionary = {}
    ind_count = 0
    start_tvd_similarity = timeit.default_timer()
    for column in df2.columns:
        min_distance = float('inf')
        most_similar_column = None
        if ind_count ==0:
            pass
        else:
            for other_column in df2.columns[0:ind_count]:
                if other_column != column:
                    tvd = total_variation_distance(df2[column].values, df2[other_column].values)
                    if tvd < min_distance:
                        min_distance = tvd
                        most_similar_column = other_column
            most_similar_columns_tvd.append(most_similar_column)
            tvd_similarity_dictionary[int(column.split('_')[1])] = int(most_similar_column.split('_')[1])
        ind_count+=1
    end_tvd_similarity = timeit.default_timer()
    tvd_similarity_time = (end_tvd_similarity - start_tvd_similarity)

    filtered_tvd_similarity_dictionary = {k: v for k, v in tvd_similarity_dictionary.items() if k >= v}
    filtered_tvd_similarity_dictionary = {key: value for key, value in filtered_tvd_similarity_dictionary.items() if
                                          value != key - 1}
    print("tvd_similarity_time is :", tvd_similarity_time)
    print(filtered_tvd_similarity_dictionary)
    return filtered_tvd_similarity_dictionary
####################################################################################################
def tvd_forecasted(df2, daily_data_for_each_month):
    most_similar_columns_tvd = []
    tvd_similarity_dictionary = {}
    ind_count = 0
    start_tvd_similarity = timeit.default_timer()
    for column in df2.columns:
        min_distance = float('inf')
        most_similar_column = None
        if ind_count ==0:
            pass
        else:
            for other_column in daily_data_for_each_month.columns[0:ind_count]:
                if other_column != column:
                    tvd = total_variation_distance(df2[column].values, daily_data_for_each_month[other_column].values)
                    if tvd < min_distance:
                        min_distance = tvd
                        most_similar_column = other_column
            most_similar_columns_tvd.append(most_similar_column)
            tvd_similarity_dictionary[int(column.split('_')[1])] = int(most_similar_column.split('_')[1])
        ind_count+=1
    end_tvd_similarity = timeit.default_timer()
    tvd_similarity_time = (end_tvd_similarity - start_tvd_similarity)
    filtered_tvd_similarity_dictionary = {k: v for k, v in tvd_similarity_dictionary.items() if k >= v}
    filtered_tvd_similarity_dictionary = {key: value for key, value in filtered_tvd_similarity_dictionary.items() if
                                          value != key - 1}
    print("tvd_similarity_time is :", tvd_similarity_time)
    print(filtered_tvd_similarity_dictionary)
    return filtered_tvd_similarity_dictionary
####################################################################################################
def reg_eval(y_test,y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    return [mae, mse, rmse, r2, mape]
####################################################################################################
def train_model_with_hyperparametertuning(X_train, y_train):
    rf_regressor = RandomForestRegressor(random_state=42, verbose=False)
    start_train_time = timeit.default_timer()
    rf_regressor.fit(X_train, y_train)
    # Calculate training time
    train_time = timeit.default_timer() - start_train_time
    # Return the trained model and training time
    return rf_regressor, train_time
####################################################################################################
def train_model2_with_hyperparametertuning(X_train, y_train):
    # Initialize the XGBRegressor with default parameters
    xgb_regressor = XGBRegressor(random_state=42)
    # Start timing the training process
    start_train_time = timeit.default_timer()
    # Train the model
    xgb_regressor.fit(X_train, y_train)
    # Calculate training time
    train_time = timeit.default_timer() - start_train_time
    # Return the trained model and training time
    return xgb_regressor, train_time
####################################################################################################
def periodical_retraining_with_hptuning(model_version, df, windows, out_columns, target_col, drop_columnss):
    ls_eval_df_periodic = []
    mean_ls = []
    mean_ls_mae = []
    total_time_start = timeit.default_timer()
    start_periodical_train_time = timeit.default_timer()
    sub_times = []
    storage_mb_ls = []
    for window in windows:
        print("window is :", window)
        start_time1 = timeit.default_timer()
        print("window size is : ", window)
        eval_df_periodic = pd.DataFrame(columns=out_columns)
        for i in range(window, len(df) + window, window):
            train = df[i - window:i]
            test = df[i:i + window]
            if len(test) < window:
                break
            train = train.reset_index()
            test = test.reset_index()

            labels = train[target_col]
            train = train.drop(columns=drop_columnss)

            y_test = test[target_col]
            X_test = test.drop(columns=drop_columnss)

            if model_version == 1:
                model, train_time = train_model_with_hyperparametertuning(train, labels)
                storage_mb_ls.append(compute_model_storage(model,model_version))
            elif model_version == 2:
                model, train_time = train_model2_with_hyperparametertuning(train, labels)
                storage_mb_ls.append(compute_model_storage(model,model_version))

            start_test_time = timeit.default_timer()
            y_pred = model.predict(X_test)
            test_time = timeit.default_timer() - start_test_time

            test_error = mean_squared_error(y_test, y_pred)
            # test_error_mae = mean_absolute_error(y_test, y_pred)

            result = reg_eval(y_test, y_pred)
            eval_df_periodic.loc[len(eval_df_periodic)] = ["trained on window i-1", 'tested on window i'] + result + [
                train_time, test_error, test_time]

        end_time1 = timeit.default_timer() - start_time1

        sub_times.append(end_time1)
        ls_eval_df_periodic.append(eval_df_periodic)
        mean_ls.append(eval_df_periodic['Testing Error'].mean()) # Testin error value is test-error
        mean_ls_mae.append(eval_df_periodic['mae'].mean())

    periodical_train_time = timeit.default_timer() - start_periodical_train_time
    avg_storage_mb = sum(storage_mb_ls)/len(storage_mb_ls)
    print("average model storage :", avg_storage_mb)
# MSE
    mean_testing_error_ls = mean_ls
    min_testing_error = np.min(mean_testing_error_ls)
    index_of_min_testing_error = mean_testing_error_ls.index(min_testing_error)
    optimal_segment_number = windows[index_of_min_testing_error]
    index_of_optimal_window = index_of_min_testing_error
    print("optimal MSE seg len time is:", sub_times[index_of_min_testing_error])
    print("mean_ls is: ", mean_ls)
    print('np.min(mean_ls) is : ', np.min(mean_ls))
    print('optimal_segment_number is :', optimal_segment_number)



#######################
#MAE
    mean_testing_error_ls_mae = mean_ls_mae
    min_testing_error_mae = np.min(mean_testing_error_ls_mae)
    index_of_min_testing_error_mae = mean_testing_error_ls_mae.index(min_testing_error_mae)
    optimal_segment_number_mae = windows[index_of_min_testing_error_mae]
    index_of_optimal_window_mae = index_of_min_testing_error_mae
    print("\n")
    print("\n")
    print("mean_ls_mae is: ", mean_ls_mae)
    print('np.min(mean_ls_mae) is : ', np.min(mean_ls_mae))
    # print('np.median(mean_ls) is: ', np.median(mean_ls))
    print('optimal_segment_number_mae is :', optimal_segment_number_mae)
    print("optimal MAE seg len time is:", sub_times[index_of_min_testing_error_mae])
   #######################
    total_time = timeit.default_timer() - total_time_start
    print("\n")
    print("total_time is: ", total_time)
    return ls_eval_df_periodic, optimal_segment_number, index_of_min_testing_error, optimal_segment_number_mae, index_of_min_testing_error_mae
####################################################################################################
def stationary_model_with_hptuning(train, test, one_month_window_size, model_version , out_columns, target_col, drop_columnss):
    total_time_start = timeit.default_timer()
    eval_df = pd.DataFrame(columns=out_columns)

    labels = train[target_col]
    train = train.drop(columns=drop_columnss)

    if model_version == 1:
        stationary_model, train_time = train_model_with_hyperparametertuning(train, labels)
        storage_mb = compute_model_storage(stationary_model, model_version)
    elif model_version == 2:
        stationary_model, train_time = train_model2_with_hyperparametertuning(train, labels)
        storage_mb = compute_model_storage(stationary_model, model_version)
    window = one_month_window_size
    for i in range(window, len(test) + window, window):
        sub_test = test[i:i + window]
        if len(sub_test) < window:
            break
        y_test = sub_test[target_col]
        X_test = sub_test.drop(columns=drop_columnss)

        start_test_time = timeit.default_timer()
        y_pred = stationary_model.predict(X_test)
        test_time = timeit.default_timer() - start_test_time
        test_error = mean_squared_error(y_test, y_pred)
        result = reg_eval(y_test, y_pred)
        eval_df.loc[len(eval_df)] = ["trained on first window size",
                                     'tested on each next months seperately'] + result + [train_time, test_error,
                                                                                          test_time]
    print("model storage is :", storage_mb)
    total_time = timeit.default_timer() - total_time_start
    print("\n")
    print("total_time is: ", total_time)
    return eval_df, stationary_model
####################################################################################################
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
####################################################################################################
def new_copied_reuse_with_hptuning_no_while_loop_with_drift(similarity_dict, stationary_model, len_of_training_data_of_stationary_model,
                                        df, forecast_approach, target_col, drop_columnss, time_steps, seasonality_periods_acf, No_of_datapoints_in_one_day, drifted_segments_ls, model_version):
    set_seed(42)
    out_columns = ['mse', 'mae', 'training_time',
                   'Testing Error', 'testing_time', 'stationary_model_testing_error', 'stationary_model_testing_time',
                   'stationary_model_mse', 'stationary_model_mae',
                   'reused_model_testing_error', 'reused_model_testing_time',
                   'reused_model_mse', 'reused_model_mae']

    ls_eval_df_monthly = []
    mean_ls_monthly = []
    window = seasonality_periods_acf*No_of_datapoints_in_one_day
    models_ls = []
    eval_df_monthly = pd.DataFrame(columns=out_columns)
    next_month_of_similar_months_ind = []
    total_reuse_time = 0
    mse_ls = []
    mae_ls = []

    drift_latest_model = stationary_model
    test_error4_ls = []
    test_error_mae4_ls = []
    test_time4_ls = []
    ml_storage_ls = []
    reuse_trained_and_stored_models_count = 0
    for i in range(window, len(df), window):

        print("window is: ", i)
        train = df[i - window:i]
        test = df[i: i + window]
        if len(test) < window-1:
            break

        train = train.reset_index().drop(columns=['index'])
        test = test.reset_index().drop(columns=['index'])
        y_test = test[target_col]
        X_test = test.drop(columns=drop_columnss)

        labels = train[target_col]
        train = train.drop(columns=drop_columnss)
        if model_version == 1:
            model, train_time = train_model_with_hyperparametertuning(train, labels)
        elif model_version == 2:
            model, train_time = train_model2_with_hyperparametertuning(train, labels)

        start_test_time = timeit.default_timer()
        y_pred = model.predict(X_test)

        test_error = mean_squared_error(y_test, y_pred)
        test_error_mae = mean_absolute_error(y_test, y_pred)
        result = [test_error, test_error_mae]
        test_time = timeit.default_timer() - start_test_time

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

                labels3 = train3[target_col]
                train3 = train3.drop(columns=drop_columnss)

                y_test3 = test[target_col]
                X_test3 = test.drop(columns=drop_columnss)

                print("math.floor(previous_model_i/window) is: ", math.floor(previous_model_i/window))
                print("len(models_ls) is:", len(models_ls))
                if model_version == 1:
                    model3, train_time3 = train_model_with_hyperparametertuning(X_test3, y_test3)
                elif model_version == 2:
                    model3, train_time3 = train_model2_with_hyperparametertuning(X_test3, y_test3)
                ml_storage_ls.append(compute_model_storage(model3,model_version))
                reuse_trained_and_stored_models_count+=1
                models_ls[similar_month_index] = model3
                models_ls.append(model3)

                start_test_time2 = timeit.default_timer()
                y_pred2 = model3.predict(X_test)

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
                y_pred2 = model2.predict(X_test)

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
                drift_latest_model = model
                test_error4_ls.append(test_error4)
                test_error_mae4_ls.append(test_error_mae4)
                test_time4_ls.append(test_time4)


            elif (drift_ls_index > 1) and (drift_ls_index in drifted_segments_ls) and (forecast_approach == "SA"):
                result4 = result
                test_error4 = test_error
                test_error_mae4 = test_error_mae
                test_time4 = test_time
                drift_latest_model = model
                test_error4_ls.append(test_error4)
                test_error_mae4_ls.append(test_error_mae4)
                test_time4_ls.append(test_time4)
            elif drift_ls_index not in drifted_segments_ls:
                start_test_time4 = timeit.default_timer()
                y_pred4 = drift_latest_model.predict(X_test)
                test_error4 = mean_squared_error(y_test, y_pred4)
                test_error_mae4 = mean_absolute_error(y_test, y_pred4)
                result4 = [test_error4, test_error_mae4]
                test_time4 = timeit.default_timer() - start_test_time4

                test_error4_ls.append(test_error4)
                test_error_mae4_ls.append(test_error_mae4)
                test_time4_ls.append(test_time4)
        else:
            sub_reuse_time_start = timeit.default_timer()
            if model_version == 1:
                model2, train_time = train_model_with_hyperparametertuning(train, labels)
            elif model_version == 2:
                model2, train_time = train_model2_with_hyperparametertuning(train, labels)
            ml_storage_ls.append(compute_model_storage(model2, model_version))
            reuse_trained_and_stored_models_count+=1
            models_ls.append(model2)
            start_test_time2 = timeit.default_timer()
            y_pred2 = model2.predict(X_test)
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
            y_pred3 = stationary_model.predict(X_test)

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
    print("\n")

    eval_df_monthly2 = eval_df_monthly[eval_df_monthly['stationary_model_testing_error'] != 'no result']
    print('\n')

    filtered_eval_df = eval_df_monthly2[eval_df_monthly2['reused_model_testing_error'] != 0]

    mean_stationary_model_testing_error_of_months_in_similar_dict = filtered_eval_df[
        'stationary_model_mse'].mean()
    mean_periodical_testing_error_of_months_in_similar_dict = filtered_eval_df['mse'].mean()
    mean_reused_model_testing_error_of_months_in_similar_dict = filtered_eval_df['reused_model_mse'].mean()
    # mean_drift_model_testing_error_of_months_in_similar_dict = filtered_eval_df['drift_model_mse'].mean()
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
    # mean_drift_model_testing_error_of_months_in_similar_dict = filtered_eval_df['drift_model_mae'].mean()
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
    return eval_df_monthly2, avg_ml_storage
################################################################################
def informed_update(stationary_model,df, target_col, drop_columnss,time_steps, seasonality_periods_acf,No_of_datapoints_in_one_day, drifted_segments_ls, model_version):
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
        labels = train[target_col]
        train = train.drop(columns=drop_columnss)
        y_test = test[target_col]
        X_test = test.drop(columns=drop_columnss)
        drift_ls_index = math.floor(i/window)
        if drift_ls_index == 1:
            if model_version == 1:
                drift_latest_model, train_time = train_model_with_hyperparametertuning(train, labels)
            elif model_version == 2:
                drift_latest_model, train_time = train_model2_with_hyperparametertuning(train, labels)
            ml_storage_ls.append(compute_model_storage(drift_latest_model,model_version))
            y_pred4 = drift_latest_model.predict(X_test)
            test_error4 = mean_squared_error(y_test, y_pred4)
            test_error_mae4 = mean_absolute_error(y_test, y_pred4)
            test_error4_ls.append(test_error4)
            test_error_mae4_ls.append(test_error_mae4)
        elif (drift_ls_index > 1) and (drift_ls_index in drifted_segments_ls):
            y_pred4 = drift_latest_model.predict(X_test)
            test_error4 = mean_squared_error(y_test, y_pred4)
            test_error_mae4 = mean_absolute_error(y_test, y_pred4)
            test_error4_ls.append(test_error4)
            test_error_mae4_ls.append(test_error_mae4)
            if model_version == 1:
                drift_latest_model, train_time = train_model_with_hyperparametertuning(train, labels)
            elif model_version == 2:
                drift_latest_model, train_time = train_model2_with_hyperparametertuning(train, labels)
            ml_storage_ls.append(compute_model_storage(drift_latest_model,model_version))
        elif drift_ls_index not in drifted_segments_ls:
            y_pred4 = drift_latest_model.predict(X_test)
            y_pred4 = y_pred4
            test_error4 = mean_squared_error(y_test, y_pred4)
            test_error_mae4 = mean_absolute_error(y_test, y_pred4)
            test_error4_ls.append(test_error4)
            test_error_mae4_ls.append(test_error_mae4)
    Informed_total_time = timeit.default_timer() - Informed_total_time_start
    print("Informed total_time: ", Informed_total_time)
    avg_ml_storage = sum(ml_storage_ls) / len(ml_storage_ls)
    print("avg_ml_storage: ", avg_ml_storage)

############################################################
def preprocess_feature_names(df):
    feature_names = df.columns
    pattern = re.compile(r'[\[\],<>]')
    feature_names = [pattern.sub('_', name) for name in feature_names]
    df.columns = [str(name) for name in feature_names]
    return df
####################################################################################################
def cut_data(df):
    index = (df['year'] == 2020) & (df['month'] == 1) & (df['day'] == 1)
    row_index = index[index].index[0]
    index_fin = (df['year'] == 2021) & (df['month'] == 1) & (df['day'] == 1)
    row_index_fin = index_fin[index_fin].index[0]
    df = df[row_index:row_index_fin]
    return df
####################################################################################################
def convert_timestamp(df, date_col_name):
    df[date_col_name] = pd.to_datetime(df[date_col_name])
    return df
####################################################################################################
def random_baseline(train, test, one_month_window_size, target_col, drop_columnss):
    total_time_start = timeit.default_timer()
    labels = train[target_col]
    train = train.drop(columns=drop_columnss)
    train_time = 0
    mse_ls = []
    window = one_month_window_size
    for i in range(window, len(test) + window, window):
        sub_test = test[i:i + window]
        if len(sub_test) < window:
            break
        y_test = sub_test[target_col]
        X_test = sub_test.drop(columns=drop_columnss)
        X_train = train
        y_train = labels

        min_val = y_train.min()
        max_val = y_train.max()
        n_predictions = len(X_test)  # Number of samples in the test set
        random_baseline_predictions = np.random.uniform(min_val, max_val, n_predictions)

        mse = mean_squared_error(y_test, random_baseline_predictions)
        mse_ls.append(mse)


    average = sum(mse_ls) / len(mse_ls)
    print("Average mse is: ", average)
####################################################################################################
def convert_sim_to_month(dictionary_, dataset_ind):
    new_key = []
    new_val = []
    dictionary_1 = {key + 1: value + 1 for key, value in dictionary_.items()}
    print("dictionary_1 is: ", dictionary_1)
    print("\n")
    keys_ls = list(dictionary_1.keys())
    vals_ls = list(dictionary_1.values())
    # Assuming dictionary_ is already defined
    if dataset_ind ==1:
        for key in keys_ls:
            if key >= 1 and key <= 7:
                year = "1996"
                month = "M" + str(key + 5)
                new_key.append(month + ", " + year)

            elif key >= 8 and key <= 19:
                year = "1997"
                month = "M" + str(key - 7)
                new_key.append(month + ", " + year)

            elif key >= 20:
                year = "1998"
                month = "M" + str(key - 19)
                new_key.append(month + ", " + year)
        for value in vals_ls:
            if value >= 1 and value <= 7:
                year = "1996"
                month = "M" + str(value + 5)
                new_val.append(month + ", " + year)

            elif value >= 8 and value <= 19:
                year = "1997"
                month = "M" + str(value - 7)
                new_val.append(month + ", " + year)

            elif value >= 20:
                year = "1998"
                month = "M" + str(value - 19)
                new_val.append(month + ", " + year)

        dictionary_C = dict(zip(new_key, new_val))
        print("final_similarities are: ", dictionary_C)
        return dictionary_C


    elif dataset_ind ==2:
        for key in keys_ls:
            year = "2020"
            month = "M" + str(key)
            new_key.append(month + ", " + year)
        for value in vals_ls:
            year = "2020"
            month = "M" + str(value)
            new_val.append(month + ", " + year)
        dictionary_C = dict(zip(new_key, new_val))
        print("final_similarities are: ", dictionary_C)
        return dictionary_C



    elif dataset_ind ==3:
        for key in keys_ls:
            if key >= 1 and key <= 6:
                year = "2016"
                month = "M" + str(key + 6)
                new_key.append(month + ", " + year)

            elif key >= 7 and key <= 18:
                year = "2017"
                month = "M" + str(key - 6)
                new_key.append(month + ", " + year)

            elif key >= 19:
                year = "2018"
                month = "M" + str(key - 18)
                new_key.append(month + ", " + year)

        for value in vals_ls:
            if value >= 1 and value <= 6:
                year = "2016"
                month = "M" + str(value + 6)
                new_val.append(month + ", " + year)

            elif value >= 7 and value <= 18:
                year = "2017"
                month = "M" + str(value - 6)
                new_val.append(month + ", " + year)

            elif value >= 19:
                year = "2018"
                month = "M" + str(value - 18)
                new_val.append(month + ", " + year)

        dictionary_C = dict(zip(new_key, new_val))
        print("final_similarities are: ", dictionary_C)
        return dictionary_C

    elif dataset_ind ==4:
        for key in keys_ls:
            year = "2020"
            month = "M" + str(key)
            new_key.append(month + ", " + year)
        for value in vals_ls:
            year = "2020"
            month = "M" + str(value)
            new_val.append(month + ", " + year)
        dictionary_C = dict(zip(new_key, new_val))
        print("final_similarities are: ", dictionary_C)
        return dictionary_C
####################################################################################################
def twitter_stationary_model_with_hptuning(train, test, model_version , out_columns, target_col, drop_columnss):
    total_time_start = timeit.default_timer()
    eval_df = pd.DataFrame(columns=out_columns)

    labels = train[target_col]
    train = train.drop(columns=drop_columnss)

    if model_version == 1:
        stationary_model, train_time = train_model_with_hyperparametertuning(train, labels)
    elif model_version == 2:
        stationary_model, train_time = train_model2_with_hyperparametertuning(train, labels)
    print("model storage is: ", compute_model_storage(stationary_model, model_version))

    y_test = test[target_col]
    X_test = test.drop(columns=drop_columnss)

    start_test_time = timeit.default_timer()
    y_pred = stationary_model.predict(X_test)
    test_time = timeit.default_timer() - start_test_time
    test_error = mean_squared_error(y_test, y_pred)
    result = reg_eval(y_test, y_pred)
    eval_df.loc[len(eval_df)] = ["trained on first window size",
                                 'tested on each next months seperately'] + result + [train_time, test_error,
                                                                                      test_time]
    total_time = timeit.default_timer() - total_time_start
    print("\n")
    print("total_time is: ", total_time)
    return eval_df, stationary_model