from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from itertools import zip_longest

def count_reduced_retrainings(similarity_dict, df, drop_columnss, target_col, window ):
    model_dict = {i: "no model" for i in range(round(len(df) / window))}
    # print(model_dict)
    count_of_reduced_training = 0
    for i in range(window, len(df), window):
        # train = df[i - window:i]
        test = df[i: i + window]
        if len(test) < window:
            break
        # train = train.reset_index().drop(columns=['index'])
        # test = test.reset_index().drop(columns=['index'])
        # labels = train[target_col]
        # train = train.drop(columns=drop_columnss)
        month_index = round(i / window) - 1
        if month_index in similarity_dict:
            count_of_reduced_training += 1
            similar_month_index = similarity_dict[month_index]
            while similar_month_index in similarity_dict:
                similar_month_index = similarity_dict[similar_month_index]
            # model = model_dict[similar_month_index]
            # print("month index is : ", month_index, ', similar month index is: ', similar_month_index)
        # else:
        #     model, train_time = train_model(train, labels)
        #     model_dict[month_index] = model

    return count_of_reduced_training


def convert_time(df, date_col_name):
    df['year'] = df[date_col_name].dt.year
    df['month'] = df[date_col_name].dt.month
    df['day'] = df[date_col_name].dt.day
    df['hour'] = df[date_col_name].dt.hour
    df['minute'] = df[date_col_name].dt.minute

    return df




import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
def forecast_es(train, test, forecast_daily_avg, forecast_avg_target_col_name):
    start_forecast_es = timeit.default_timer()
    model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=12)
    fit = model.fit()

    # Forecast the values
    forecast = fit.forecast(steps=len(test))
    end_forecast_es = timeit.default_timer()
    forecast_time = end_forecast_es - start_forecast_es
    # print("Forecast time: ", forecast_time)
    # Compute the Mean Squared Error
    mse = mean_squared_error(test, forecast)

    results = pd.DataFrame(forecast, columns=[forecast_avg_target_col_name])

    # Output the MSE and the results dataframe
    # print(f'Mean Squared Error: {mse}')
    forecast_daily_avg = pd.concat([forecast_daily_avg, results], ignore_index=True)

    return forecast_daily_avg, forecast_time




from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
# def forecast_sa(train, test, forecast_daily_avg, forecast_avg_target_col_name):
#     # Automatically determine the best p, d, q, P, D, Q, m values using auto_arima
#     model = auto_arima(train, seasonal=True, m=12, trace=True,
#                        error_action='ignore', suppress_warnings=True, stepwise=True)
#
#     # Print the best parameters
#     print(model.summary())
#
#     # Make predictions
#     predictions = model.predict(n_periods=len(test))
#
#
#     # Calculate error
#     error = mean_squared_error(test, predictions)
#     print(f'Test MSE: {error}')
#     # print('type(prediction is: ', type(predictions))
#     # print('prediction.shape is: ', predictions.shape)
#     # print('test.shape is: ', test.shape)
#     # print('prediction is: ', predictions)
#     print("\n")
#     results = pd.DataFrame(predictions, columns=[forecast_avg_target_col_name])
#     forecast_daily_avg = pd.concat([forecast_daily_avg, results], ignore_index=True)
#
#
#     # forecast_daily_avg
#
#     return forecast_daily_avg



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# old plotting version, to be deleted
# def plot_months_patterns(avg, one_month_days, forecast_or_regular_avg_target_col_name, index):
#     # Create custom colormap
#     num_segments = len(avg) // one_month_days + 1
#     # colors = plt.cm.viridis(np.linspace(0, 1, num_segments))
#     colors = plt.cm.tab20c(np.linspace(0, 1, num_segments))
#
#     cmap = mcolors.ListedColormap(colors)
#
#     # Plot the data
#     fig, ax = plt.subplots()
#     for i in range(0, len(avg), one_month_days):
#         # if index== 0:
#         #     ax.plot(avg.index[i:i + one_month_days], avg[forecast_or_regular_avg_target_col_name][i:i + one_month_days],
#         #     color=colors[i // one_month_days], label=f'month {(i // one_month_days)+7}')
#         # elif index ==1:
#         #     ax.plot(avg.index[i:i + one_month_days]-184, avg[forecast_or_regular_avg_target_col_name][i:i + one_month_days],
#         #     color=colors[i // one_month_days], label=f'month {(i // one_month_days)}')
#         # else:
#         #     ax.plot(avg.index[i:i + one_month_days]-549, avg[forecast_or_regular_avg_target_col_name][i:i + one_month_days],
#         #     color=colors[i // one_month_days], label=f'month {(i // one_month_days)}')
#
#     #######################
#         ax.plot(avg.index[i:i + one_month_days], avg[forecast_or_regular_avg_target_col_name][i:i + one_month_days],
#         color=colors[i // one_month_days], label=f'month {(i // one_month_days)}')
#
#     # Customize the plot
#     ax.set_xlabel('Days')
#     ax.set_ylabel('Daily average of target value')
#     ax.set_title('')
#     ax.legend()
#
#     plt.show()



import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def plot_months_patterns(avg, one_month_days, forecast_or_regular_avg_target_col_name, index):
    num_segments = len(avg) // one_month_days + 1
    colors = plt.cm.tab20c(np.linspace(0, 1, num_segments))

    cmap = mcolors.ListedColormap(colors)

    fig, ax = plt.subplots()
    for i in range(0, len(avg), one_month_days):
        ax.plot(avg.index[i:i + one_month_days], avg[forecast_or_regular_avg_target_col_name][i:i + one_month_days],
                color=colors[i // one_month_days], label=f'Month {(i // one_month_days) + 1}')
    ax.set_xlabel('Days')
    ax.set_ylabel('Daily average of target value')
    ax.set_title('')

    # Place the legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def plot_months_patterns_box(avg, one_month_days, forecast_or_regular_avg_target_col_name, index):
    num_segments = len(avg) // one_month_days + 1
    colors = plt.cm.tab20c(np.linspace(0, 1, num_segments))

    fig, ax = plt.subplots()

    # Prepare data for boxplot
    data_for_boxplot = []
    for i in range(0, len(avg), one_month_days):
        data_for_boxplot.append(avg[forecast_or_regular_avg_target_col_name][i:i + one_month_days].values)

    # Create boxplot
    ax.boxplot(data_for_boxplot, patch_artist=True)

    # Set colors for each box
    for patch, color in zip(ax.artists, colors):
        patch.set_facecolor(color)

    ax.set_xlabel('Months')
    ax.set_ylabel('Daily average of target value')
    ax.set_title('Monthly box plots')

    # Set x-ticks to be the month numbers
    ax.set_xticks(np.arange(1, num_segments + 1))
    ax.set_xticklabels([f'M {i + 1}' for i in range(num_segments)])

    plt.show()




def make_daily_data_for_each_month(daily_df_avg, avg_target_col, one_month_days):
    daily_col_ls = []
    for i in range(len(daily_df_avg)):
        if daily_df_avg.iloc[i, 3] == 1:
            daily_df_avg3 = daily_df_avg[i:i + one_month_days]  # we considered each month as 30 days
            daily_df_avg4 = daily_df_avg3[[avg_target_col]].copy()
            if len(daily_df_avg4) >= one_month_days:
                daily_col_ls.append(daily_df_avg4[avg_target_col].tolist())

    # Create an empty DataFrame
    daily_data_for_each_month = pd.DataFrame()

    # Iterate through the list of lists
    for i, lst in enumerate(daily_col_ls):
        column_name = f'month_{i}'  # Set the name for each column
        daily_data_for_each_month[column_name] = lst

    return daily_data_for_each_month


# compute wasserstein distance
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance

from itertools import combinations


def wasserstein_dist(df2):
    # Create a list to store the most similar column for each column
    most_similar_columns_wasserstein = []
    wess_similarity_dictionary = {}
    # Iterate through each column
    start_wess_similarity = timeit.default_timer()
    ind_count = 0
    for column in df2.columns:
        print("column is : ", column)
        min_distance = float('inf')
        most_similar_column = None
        # Compute EMD between the current column and all other columns
        if ind_count ==0:
            pass
        else:
            for other_column in df2.columns[0:ind_count]:
                if other_column != column:
                    # print("other column is : ", other_column)
                    # Calculate Earth Mover's Distance (EMD)
                    emd = wasserstein_distance(df2[column], df2[other_column])

                    # Update most similar column if the current EMD is smaller
                    if emd < min_distance:
                        # print("min dist other column is : ", other_column)
                        min_distance = emd
                        most_similar_column = other_column
            print('\n')

            # Append the name of the most similar column to the list
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


def wasserstein_dist_forecasted(df2, daily_data_for_each_month):
    # Create a list to store the most similar column for each column
    most_similar_columns_wasserstein = []
    wess_similarity_dictionary = {}

    # Iterate through each column
    start_wess_similarity = timeit.default_timer()
    ind_count = 0
    for column in df2.columns:
        min_distance = float('inf')
        most_similar_column = None
        if ind_count ==0:
            pass
        else:
            # Compute EMD between the current column and all other columns
            for other_column in daily_data_for_each_month.columns[0:ind_count]:
                if other_column != column:
                    # Calculate Earth Mover's Distance (EMD)
                    emd = wasserstein_distance(df2[column], daily_data_for_each_month[other_column])

                    # Update most similar column if the current EMD is smaller
                    if emd < min_distance:
                        min_distance = emd
                        most_similar_column = other_column
                        # print("min distance is : ", min_distance)
                        # print("\n")

            # Append the name of the most similar column to the list
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


# Total Variation Distance

import pandas as pd
import numpy as np
from itertools import combinations


def total_variation_distance(p, q):
    return 0.5 * np.sum(np.abs(p - q))


def tvd(df2):
    # Create a list to store the most similar column for each column
    most_similar_columns_tvd = []
    tvd_similarity_dictionary = {}
    ind_count = 0
    start_tvd_similarity = timeit.default_timer()
    # Iterate through each column
    for column in df2.columns:
        min_distance = float('inf')
        most_similar_column = None
        if ind_count ==0:
            pass
        else:
            # Compute TVD between the current column and all other columns
            for other_column in df2.columns[0:ind_count]:
                if other_column != column:
                    # Calculate Total Variation Distance (TVD)
                    tvd = total_variation_distance(df2[column].values, df2[other_column].values)

                    # Update most similar column if the current TVD is smaller
                    if tvd < min_distance:
                        min_distance = tvd
                        most_similar_column = other_column
                        # print("min distance is : ", min_distance)
                        # print("\n")

            # Append the name of the most similar column to the list
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



def tvd_forecasted(df2, daily_data_for_each_month):
    # Create a list to store the most similar column for each column
    most_similar_columns_tvd = []
    tvd_similarity_dictionary = {}
    ind_count = 0
    start_tvd_similarity = timeit.default_timer()
    # Iterate through each column
    for column in df2.columns:
        min_distance = float('inf')
        most_similar_column = None
        if ind_count ==0:
            pass
        else:
            # Compute TVD between the current column and all other columns
            for other_column in daily_data_for_each_month.columns[0:ind_count]:
                if other_column != column:
                    # Calculate Total Variation Distance (TVD)
                    tvd = total_variation_distance(df2[column].values, daily_data_for_each_month[other_column].values)

                    # Update most similar column if the current TVD is smaller
                    if tvd < min_distance:
                        min_distance = tvd
                        most_similar_column = other_column
                        # print("min distance is : ", min_distance)
                        # print("\n")

            # Append the name of the most similar column to the list
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




from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def reg_eval(y_test,y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    return [mae, mse, rmse, r2, mape]


import timeit
from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor
def train_model(X_train, y_train):
    train_time = 0

    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    start_train_time = timeit.default_timer()
    rf_regressor.fit(X_train, y_train)
    train_time = timeit.default_timer() - start_train_time

    return rf_regressor, train_time



from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
def train_model_with_hyperparametertuning(X_train, y_train):
   # Define the hyperparameter grid to search
   param_dist = {
       'n_estimators': [100, 200, 500],  # Number of trees in the forest
       'max_features': ['sqrt', 'log2', None],  # Number of features to consider
       'max_depth': [10, 20, 30, None],  # Max depth of each tree
       'min_samples_split': [2, 5, 10],  # Min samples to split a node
       'min_samples_leaf': [1, 2, 4],  # Min samples at a leaf node
       'bootstrap': [True, False]  # Whether to use bootstrapping
   }


   # Initialize the RandomForestRegressor
   rf_regressor = RandomForestRegressor(random_state=42, verbose=False)


   # Initialize the RandomizedSearchCV object
   random_search = RandomizedSearchCV(
       estimator=rf_regressor,
       param_distributions=param_dist,
       n_iter=10,  # Number of parameter settings to try
       cv=5,  # 5-fold cross-validation
       verbose=2,  # Output progress
       random_state=42,
       n_jobs=-1  # Use all available cores
   )


   # Start timing the training process
   start_train_time = timeit.default_timer()


   # Perform Randomized Search
   random_search.fit(X_train, y_train)


   # Calculate training time
   train_time = timeit.default_timer() - start_train_time
   # Extract MSE for each fold


   # mse_scores = (-random_search.cv_results_['split0_test_score'],
   #               -random_search.cv_results_['split1_test_score'],
   #               -random_search.cv_results_['split2_test_score'],
   #               -random_search.cv_results_['split3_test_score'],
   #               -random_search.cv_results_['split4_test_score'])
   #
   #
   # # Print MSE for each fold and compute the average
   # for i, fold_scores in enumerate(mse_scores):
   #     avg_mse = np.mean(fold_scores)
   #     print(f"Fold {i + 1} MSE scores: {fold_scores}")
   #     print(f"Fold {i + 1} average MSE: {avg_mse}\n")
   #
   #
   # print("################################################")


   # Return the best estimator and training time
   return random_search.best_estimator_, train_time



def train_model2(X_train, y_train):
    train_time = 0

    xgb_regressor = XGBRegressor(n_estimators=100, random_state=42)

    start_train_time = timeit.default_timer()
    xgb_regressor.fit(X_train, y_train)
    train_time = timeit.default_timer() - start_train_time

    return xgb_regressor, train_time

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
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

    # Initialize the XGBRegressor
    xgb_regressor = XGBRegressor(random_state=42)

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=xgb_regressor,
        param_distributions=param_dist,
        n_iter=10,                  # Number of parameter settings to try
        cv=5,                       # 5-fold cross-validation
        verbose=2,                  # Output progress
        random_state=42,
        n_jobs=-1                   # Use all available cores
    )

    # Start timing the training process
    start_train_time = timeit.default_timer()

    # Perform Randomized Search
    random_search.fit(X_train, y_train)

    # Calculate training time
    train_time = timeit.default_timer() - start_train_time

    # Return the best estimator and training time
    return random_search.best_estimator_, train_time

def periodical_retraining(model_version, df, windows, out_columns, target_col, drop_columnss):
    ls_eval_df_periodic = []
    mean_ls = []
    mean_ls_mae = []
    total_time_start = timeit.default_timer()
    start_periodical_train_time = timeit.default_timer()
    for window in windows:
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

            if model_version == 1:
                model, train_time = train_model(train, labels)
            elif model_version == 2:
                model, train_time = train_model2(train, labels)

            y_test = test[target_col]
            X_test = test.drop(columns=drop_columnss)

            start_test_time = timeit.default_timer()
            y_pred = model.predict(X_test)
            test_time = timeit.default_timer() - start_test_time

            test_error = mean_squared_error(y_test, y_pred)

            result = reg_eval(y_test, y_pred)
            # print("result is: ",result)
            eval_df_periodic.loc[len(eval_df_periodic)] = ["trained on window i-1", 'tested on window i'] + result + [
                train_time, test_error, test_time]
        ls_eval_df_periodic.append(eval_df_periodic)
        mean_ls.append(eval_df_periodic['Testing Error'].mean()) # Testin error value is test-error
        mean_ls_mae.append(eval_df_periodic['mae'].mean())

    periodical_train_time = timeit.default_timer() - start_periodical_train_time
    # print("periodical_training time: " + str(periodical_train_time))
# MSE
    mean_testing_error_ls = mean_ls
    min_testing_error = np.min(mean_testing_error_ls)
    index_of_min_testing_error = mean_testing_error_ls.index(min_testing_error)
    optimal_segment_number = windows[index_of_min_testing_error]
    index_of_optimal_window = index_of_min_testing_error

    print("mean_ls is: ", mean_ls)
    print('np.min(mean_ls) is : ', np.min(mean_ls))
    # print('np.median(mean_ls) is: ', np.median(mean_ls))
    print('optimal_segment_number is :', optimal_segment_number)
    total_time = timeit.default_timer() - total_time_start
    print("\n")
    print("total_time is: ", total_time)


#MAE
    mean_testing_error_ls_mae = mean_ls_mae
    min_testing_error_mae = np.min(mean_testing_error_ls_mae)
    index_of_min_testing_error_mae = mean_testing_error_ls_mae.index(min_testing_error_mae)
    optimal_segment_number_mae = windows[index_of_min_testing_error_mae]
    index_of_optimal_window_mae = index_of_min_testing_error_mae
    # print("\n")
    # print("\n")
    # print("mean_ls_mae is: ", mean_ls_mae)
    # print('np.min(mean_ls_mae) is : ', np.min(mean_ls_mae))
    # # print('np.median(mean_ls) is: ', np.median(mean_ls))
    # print('optimal_segment_number_mae is :', optimal_segment_number_mae)

    return ls_eval_df_periodic, optimal_segment_number, index_of_min_testing_error, optimal_segment_number_mae, index_of_min_testing_error_mae







def periodical_retraining_with_hptuning(model_version, df, windows, out_columns, target_col, drop_columnss):
    ls_eval_df_periodic = []
    mean_ls = []
    mean_ls_mae = []
    total_time_start = timeit.default_timer()
    start_periodical_train_time = timeit.default_timer()
    for window in windows:
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

            if model_version == 1:
                model, train_time = train_model_with_hyperparametertuning(train, labels)
            elif model_version == 2:
                model, train_time = train_model2_with_hyperparametertuning(train, labels)

            y_test = test[target_col]
            X_test = test.drop(columns=drop_columnss)

            start_test_time = timeit.default_timer()
            y_pred = model.predict(X_test)
            test_time = timeit.default_timer() - start_test_time

            test_error = mean_squared_error(y_test, y_pred)

            result = reg_eval(y_test, y_pred)
            # print("result is: ",result)
            eval_df_periodic.loc[len(eval_df_periodic)] = ["trained on window i-1", 'tested on window i'] + result + [
                train_time, test_error, test_time]
        ls_eval_df_periodic.append(eval_df_periodic)
        mean_ls.append(eval_df_periodic['Testing Error'].mean()) # Testin error value is test-error
        mean_ls_mae.append(eval_df_periodic['mae'].mean())

    periodical_train_time = timeit.default_timer() - start_periodical_train_time
    # print("periodical_training time: " + str(periodical_train_time))
# MSE
    mean_testing_error_ls = mean_ls
    min_testing_error = np.min(mean_testing_error_ls)
    index_of_min_testing_error = mean_testing_error_ls.index(min_testing_error)
    optimal_segment_number = windows[index_of_min_testing_error]
    index_of_optimal_window = index_of_min_testing_error

    print("mean_ls is: ", mean_ls)
    print('np.min(mean_ls) is : ', np.min(mean_ls))
    # print('np.median(mean_ls) is: ', np.median(mean_ls))
    print('optimal_segment_number is :', optimal_segment_number)
    total_time = timeit.default_timer() - total_time_start
    print("\n")
    print("total_time is: ", total_time)




#####################
    # Get the minimum, second minimum, and third minimum
    min_values = np.partition(mean_testing_error_ls, 2)[:3]  # Get three smallest values in any order
    min_values_sorted = sorted(min_values)  # Sort to ensure order of minimums

    # Assign values
    min_testing_error = min_values_sorted[0]
    second_min_testing_error = min_values_sorted[1]
    third_min_testing_error = min_values_sorted[2]

    # Get indices of these minimums
    index_of_min_testing_error = mean_testing_error_ls.index(min_testing_error)
    index_of_second_min_testing_error = mean_testing_error_ls.index(second_min_testing_error)
    index_of_third_min_testing_error = mean_testing_error_ls.index(third_min_testing_error)

    # If you have a corresponding windows array:
    optimal_segment_number = windows[index_of_min_testing_error]
    second_optimal_segment_number = windows[index_of_second_min_testing_error]
    third_optimal_segment_number = windows[index_of_third_min_testing_error]

    # Print results
    print("Minimum Testing Error:", min_testing_error)
    print("Second Minimum Testing Error:", second_min_testing_error)
    print("Third Minimum Testing Error:", third_min_testing_error)

    print("Index of Minimum Testing Error:", index_of_min_testing_error)
    print("Index of Second Minimum Testing Error:", index_of_second_min_testing_error)
    print("Index of Third Minimum Testing Error:", index_of_third_min_testing_error)

    print("Optimal Segment Number:", optimal_segment_number)
    print("Second Optimal Segment Number:", second_optimal_segment_number)
    print("Third Optimal Segment Number:", third_optimal_segment_number)



#######################
#MAE
    mean_testing_error_ls_mae = mean_ls_mae
    min_testing_error_mae = np.min(mean_testing_error_ls_mae)
    index_of_min_testing_error_mae = mean_testing_error_ls_mae.index(min_testing_error_mae)
    optimal_segment_number_mae = windows[index_of_min_testing_error_mae]
    index_of_optimal_window_mae = index_of_min_testing_error_mae
    # print("\n")
    # print("\n")
    # print("mean_ls_mae is: ", mean_ls_mae)
    # print('np.min(mean_ls_mae) is : ', np.min(mean_ls_mae))
    # # print('np.median(mean_ls) is: ', np.median(mean_ls))
    # print('optimal_segment_number_mae is :', optimal_segment_number_mae)

    return ls_eval_df_periodic, optimal_segment_number, index_of_min_testing_error, optimal_segment_number_mae, index_of_min_testing_error_mae




def periodical_retraining_with_hptuning_with_scoring(model_version, df, windows, out_columns, target_col, drop_columnss):
    ls_eval_df_periodic = []
    mean_ls = []
    mean_ls_mae = []
    total_time_start = timeit.default_timer()
    start_periodical_train_time = timeit.default_timer()
    both_mse_lss = []
    for window in windows:
        mse_ls = []
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

            if model_version == 1:
                model, train_time = train_model_with_hyperparametertuning(train, labels)
            elif model_version == 2:
                model, train_time = train_model2_with_hyperparametertuning(train, labels)

            y_test = test[target_col]
            X_test = test.drop(columns=drop_columnss)

            start_test_time = timeit.default_timer()
            y_pred = model.predict(X_test)
            test_time = timeit.default_timer() - start_test_time

            test_error = mean_squared_error(y_test, y_pred)

            result = reg_eval(y_test, y_pred)
            # print("result is: ",result)
            eval_df_periodic.loc[len(eval_df_periodic)] = ["trained on window i-1",
                                                           'tested on window i'] + result + [
                                                              train_time, test_error, test_time]
        ls_eval_df_periodic.append(eval_df_periodic)
        mean_ls.append(eval_df_periodic['Testing Error'].mean())  # Testin error value is test-error
        mean_ls_mae.append(eval_df_periodic['mae'].mean())

    periodical_train_time = timeit.default_timer() - start_periodical_train_time
    # print("periodical_training time: " + str(periodical_train_time))
    # MSE
    mean_testing_error_ls = mean_ls
    min_testing_error = np.min(mean_testing_error_ls)
    index_of_min_testing_error = mean_testing_error_ls.index(min_testing_error)
    optimal_segment_number = windows[index_of_min_testing_error]
    index_of_optimal_window = index_of_min_testing_error

    print("mean_ls is: ", mean_ls)
    print('np.min(mean_ls) is : ', np.min(mean_ls))
    # print('np.median(mean_ls) is: ', np.median(mean_ls))
    print('optimal_segment_number is :', optimal_segment_number)
    total_time = timeit.default_timer() - total_time_start
    print("\n")
    print("total_time is: ", total_time)

    # MAE
    mean_testing_error_ls_mae = mean_ls_mae
    min_testing_error_mae = np.min(mean_testing_error_ls_mae)
    index_of_min_testing_error_mae = mean_testing_error_ls_mae.index(min_testing_error_mae)
    optimal_segment_number_mae = windows[index_of_min_testing_error_mae]
    index_of_optimal_window_mae = index_of_min_testing_error_mae
    # print("\n")
    # print("\n")
    # print("mean_ls_mae is: ", mean_ls_mae)
    # print('np.min(mean_ls_mae) is : ', np.min(mean_ls_mae))
    # # print('np.median(mean_ls) is: ', np.median(mean_ls))
    # print('optimal_segment_number_mae is :', optimal_segment_number_mae)

    return ls_eval_df_periodic, optimal_segment_number, index_of_min_testing_error, optimal_segment_number_mae, index_of_min_testing_error_mae





def stationary_model(train, test, one_month_window_size, model_version , out_columns, target_col, drop_columnss):
    total_time_start = timeit.default_timer()
    eval_df = pd.DataFrame(columns=out_columns)

    labels = train[target_col]
    train = train.drop(columns=drop_columnss)

    if model_version == 1:
        stationary_model, train_time = train_model(train, labels)
    elif model_version == 2:
        stationary_model, train_time = train_model2(train, labels)

    # 30 days* 144 data points per day = 4320
    window = one_month_window_size
    for i in range(window, len(test) + window, window):
        # print(i)
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
        # print("result is: ",result)
        eval_df.loc[len(eval_df)] = ["trained on first window size",
                                     'tested on each next months seperately'] + result + [train_time, test_error,
                                                                                          test_time]

    total_time = timeit.default_timer() - total_time_start
    print("\n")
    print("total_time is: ", total_time)
    return eval_df, stationary_model


def stationary_model_with_hptuning(train, test, one_month_window_size, model_version , out_columns, target_col, drop_columnss):
    total_time_start = timeit.default_timer()
    eval_df = pd.DataFrame(columns=out_columns)

    labels = train[target_col]
    train = train.drop(columns=drop_columnss)

    if model_version == 1:
        stationary_model, train_time = train_model_with_hyperparametertuning(train, labels)
    elif model_version == 2:
        stationary_model, train_time = train_model2_with_hyperparametertuning(train, labels)

    # 30 days* 144 data points per day = 4320
    window = one_month_window_size
    for i in range(window, len(test) + window, window):
        # print(i)
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
        # print("result is: ",result)
        eval_df.loc[len(eval_df)] = ["trained on first window size",
                                     'tested on each next months seperately'] + result + [train_time, test_error,
                                                                                          test_time]

    total_time = timeit.default_timer() - total_time_start
    print("\n")
    print("total_time is: ", total_time)
    return eval_df, stationary_model


def model_reuse(similarity_dict, stationary_model, number_of_window_in_stationary_training_data, one_month_window_size,
                df, model_version, forecast_var_bool,target_col, drop_columnss):
    out_columns = ['Training dataset', 'Testing dataset', 'mae', 'mse', 'rmse', 'r2', 'mape', 'training_time',
                   'Testing Error', 'testing_time', 'stationary_model_testing_error', 'stationary_model_testing_time',
                   'stationary_model_mae', 'stationary_model_mse', 'stationary_model_rmse', 'stationary_model_r2',
                   'stationary_model_mape', 'reused_model_testing_error', 'reused_model_testing_time',
                   'reused_model_mae', 'reused_model_mse', 'reused_model_rmse', 'reused_model_r2', 'reused_model_mape']

    ls_eval_df_monthly = []
    mean_ls_monthly = []
    window = one_month_window_size
    models_ls = []

    eval_df_monthly = pd.DataFrame(columns=out_columns)
    next_month_of_similar_months_ind = []
    total_reuse_time = 0
    mse_ls = []
    for i in range(window, len(df), window):
        train = df[i - window:i]
        test = df[i: i + window]

        if len(test) < window:
            break

        train = train.reset_index().drop(columns=['index'])
        test = test.reset_index().drop(columns=['index'])

        labels = train[target_col]
        train = train.drop(columns=drop_columnss)

        # print("train columns:", train[:10])
        if model_version == 1:
            model, train_time = train_model(train, labels)
        elif model_version == 2:
            model, train_time = train_model2(train, labels)

        # models_ls.append(model)

        y_test = test[target_col]
        X_test = test.drop(columns=drop_columnss)

        start_test_time = timeit.default_timer()
        y_pred = model.predict(X_test)
        test_time = timeit.default_timer() - start_test_time
        test_error = mean_squared_error(y_test, y_pred)
        result = reg_eval(y_test, y_pred)

        # reuse model results
        total_reuse_time_start = timeit.default_timer()
        if forecast_var_bool == False:
            month_index = round(i / window) - 1
            # print("fist if")
        elif forecast_var_bool == True:
            month_index = round(i / window)
            # print("second if")
        if month_index in similarity_dict:
            similar_month_index = similarity_dict[month_index]
            while similar_month_index in similarity_dict:
                similar_month_index = similarity_dict[similar_month_index]
            next_month_of_similar_months_ind.append(similar_month_index)
            model2 = models_ls[similar_month_index]
            # print("month index is : ", month_index, ', similar month index is: ', similar_month_index)
            models_ls.append(model2)
            start_test_time2 = timeit.default_timer()
            y_pred2 = model2.predict(X_test)
            test_time2 = timeit.default_timer() - start_test_time2
            test_error2 = mean_squared_error(y_test, y_pred2)
            result2 = reg_eval(y_test, y_pred2)
        else:
            if model_version == 1:
                model2, train_time = train_model(train, labels)
            elif model_version == 2:
                model2, train_time = train_model2(train, labels)
            models_ls.append(model2)
            start_test_time2 = timeit.default_timer()
            y_pred2 = model2.predict(X_test)
            test_time2 = timeit.default_timer() - start_test_time2
            test_error2 = mean_squared_error(y_test, y_pred2)
            result2 = reg_eval(y_test, y_pred2)

        total_reuse_time1 = timeit.default_timer() - total_reuse_time_start
        total_reuse_time = total_reuse_time + total_reuse_time1
        if len(result2) == 0:
            result2 = ["no result", "no result", 'no result', 'no result', 'no result']

        # stationary_model1 results
        if i < number_of_window_in_stationary_training_data * window:
            test_error3 = "no result"
            test_time3 = "no result"
            result3 = ["no result", "no result", 'no result', 'no result', 'no result']
        elif i >= number_of_window_in_stationary_training_data * window:
            test_error3 = 0
            test_time3 = 0
            result3 = []
            start_test_time3 = timeit.default_timer()
            y_pred3 = stationary_model.predict(X_test)
            test_time3 = timeit.default_timer() - start_test_time3
            test_error3 = mean_squared_error(y_test, y_pred3)
            result3 = reg_eval(y_test, y_pred3)

# random baseline
        X_train = train
        y_train = labels


        # random baseline approach 1
        # Define the range based on the min and max stock prices from the training data
        min_val = y_train.min()
        max_val = y_train.max()
        # Generate random predictions for the test set
        n_predictions = len(X_test)  # Number of samples in the test set
        random_baseline_predictions = np.random.uniform(min_val, max_val, n_predictions)

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, random_baseline_predictions)
        # mse_ls.append(mse)


        # random baseline approach 2
        # 1. Mean Baseline Model
        # mean_baseline_pred = np.mean(y_train)  # Constant prediction using the mean of the training target
        #
        # # Predict the mean for all test samples
        # mean_predictions = np.full_like(y_test, mean_baseline_pred)
        #
        # # Evaluate the mean baseline model
        # mse = mean_squared_error(y_test, mean_predictions)
        if len(result2) != 0:
            mse_ls.append(mse)
        else:
            pass


        eval_df_monthly.loc[len(eval_df_monthly)] = [f"trained on month {(i / window) - 1}",
                                                     f'tested on month {i / window}'] + result + [train_time,
                                                                                                  test_error,
                                                                                                  test_time] + [
                                                        test_error3, test_time3] + result3 + [test_error2,
                                                                                              test_time2] + result2
    print("\n")
    print("total_reuse_time is : ", total_reuse_time)
    print("\n")

    sum_msr_on_next_month = 0

    for r in range(0, len(next_month_of_similar_months_ind)):
        next_month_row_no = next_month_of_similar_months_ind[r]
        # print("next_month_row_no is", next_month_row_no)
        sum_msr_on_next_month += eval_df_monthly.iloc[next_month_row_no, 8]
    print('sum_msr_on_next_month is : ', sum_msr_on_next_month)

    eval_df_monthly2 = eval_df_monthly[eval_df_monthly['stationary_model_testing_error'] != 'no result']

    eval_df_monthly2[
        'ideal:True _is_testing_error_of_model_trained_on_previous_month_greater_than_reused_model_testing_error'] = \
    eval_df_monthly2['reused_model_testing_error'] < eval_df_monthly2['Testing Error']

    eval_df_monthly2['Ideal:True_ is_stationary_testing_error_greater_than_reused_model_testing_error'] = \
    eval_df_monthly2['reused_model_testing_error'] < eval_df_monthly2['stationary_model_testing_error']

    eval_df_monthly2[
        'ideal: True_ is_stationary_testing_error_greater_than_testing_error_of_model_trained_on_previous_month'] = \
    eval_df_monthly2['Testing Error'] < eval_df_monthly2['stationary_model_testing_error']

    eval_df_monthly2['merge of reuse error and test error'] = eval_df_monthly2.apply(
        lambda row: row['reused_model_testing_error'] if row['reused_model_testing_error'] != 0 else row[
            'Testing Error'], axis=1)

    print('\n')

    filtered_eval_df = eval_df_monthly2[eval_df_monthly2['reused_model_testing_error'] != 0]

    mean_stationary_model_testing_error_of_months_in_similar_dict = filtered_eval_df[
        'stationary_model_testing_error'].mean()
    mean_periodical_testing_error_of_months_in_similar_dict = filtered_eval_df['Testing Error'].mean()
    mean_reused_model_testing_error_of_months_in_similar_dict = filtered_eval_df['reused_model_testing_error'].mean()

    print('mean_stationary_model_testing_error_of_months_in_similar_dict is: ',
          mean_stationary_model_testing_error_of_months_in_similar_dict)
    print('mean_periodical_testing_error_of_months_in_similar_dict is: ',
          mean_periodical_testing_error_of_months_in_similar_dict)
    print('mean_reused_model_testing_error_of_months_in_similar_dict is: ',
          mean_reused_model_testing_error_of_months_in_similar_dict)

    average = sum(mse_ls) / len(mse_ls)
    print("Average mse in random model is: ", average)
    print("filtered_eval_df is:", len(filtered_eval_df))
    return eval_df_monthly2







def model_reuse_with_hptuning(similarity_dict, stationary_model, number_of_window_in_stationary_training_data, one_month_window_size,
                df, model_version, forecast_var_bool,target_col, drop_columnss):
    out_columns = ['Training dataset', 'Testing dataset', 'mae', 'mse', 'rmse', 'r2', 'mape', 'training_time',
                   'Testing Error', 'testing_time', 'stationary_model_testing_error', 'stationary_model_testing_time',
                   'stationary_model_mae', 'stationary_model_mse', 'stationary_model_rmse', 'stationary_model_r2',
                   'stationary_model_mape', 'reused_model_testing_error', 'reused_model_testing_time',
                   'reused_model_mae', 'reused_model_mse', 'reused_model_rmse', 'reused_model_r2', 'reused_model_mape']

    ls_eval_df_monthly = []
    mean_ls_monthly = []
    window = one_month_window_size
    models_ls = []

    eval_df_monthly = pd.DataFrame(columns=out_columns)
    next_month_of_similar_months_ind = []
    total_reuse_time = 0
    mse_ls = []
    for i in range(window, len(df), window):
        train = df[i - window:i]
        test = df[i: i + window]
        print("round(i / window) is : ", round(i / window))
        if len(test) < window:
            break

        train = train.reset_index().drop(columns=['index'])
        test = test.reset_index().drop(columns=['index'])

        labels = train[target_col]
        train = train.drop(columns=drop_columnss)

        # print("train columns:", train[:10])
        if model_version == 1:
            model, train_time = train_model_with_hyperparametertuning(train, labels)
        elif model_version == 2:
            model, train_time = train_model2_with_hyperparametertuning(train, labels)

        # models_ls.append(model)

        y_test = test[target_col]
        X_test = test.drop(columns=drop_columnss)

        start_test_time = timeit.default_timer()
        y_pred = model.predict(X_test)
        test_time = timeit.default_timer() - start_test_time
        test_error = mean_squared_error(y_test, y_pred)
        result = reg_eval(y_test, y_pred)

        # reuse model results
        total_reuse_time_start = timeit.default_timer()
        if forecast_var_bool == False:
            month_index = round(i / window) - 1
            # print("fist if")
        elif forecast_var_bool == True:
            month_index = round(i / window)
            # print("second if")
        if month_index in similarity_dict:
            similar_month_index = similarity_dict[month_index]
            while similar_month_index in similarity_dict:
                similar_month_index = similarity_dict[similar_month_index]
            next_month_of_similar_months_ind.append(similar_month_index)
            print("month_index: ", month_index)
            print('\n')

            model2 = models_ls[similar_month_index]
            # print("month index is : ", month_index, ', similar month index is: ', similar_month_index)
            models_ls.append(model2)
            start_test_time2 = timeit.default_timer()
            y_pred2 = model2.predict(X_test)
            test_time2 = timeit.default_timer() - start_test_time2
            test_error2 = mean_squared_error(y_test, y_pred2)
            result2 = reg_eval(y_test, y_pred2)
        else:
            if model_version == 1:
                model2, train_time = train_model_with_hyperparametertuning(train, labels)
            elif model_version == 2:
                model2, train_time = train_model2_with_hyperparametertuning(train, labels)
            models_ls.append(model2)
            start_test_time2 = timeit.default_timer()
            y_pred2 = model2.predict(X_test)
            test_time2 = timeit.default_timer() - start_test_time2
            test_error2 = mean_squared_error(y_test, y_pred2)
            result2 = reg_eval(y_test, y_pred2)

        total_reuse_time1 = timeit.default_timer() - total_reuse_time_start
        total_reuse_time = total_reuse_time + total_reuse_time1
        if len(result2) == 0:
            result2 = ["no result", "no result", 'no result', 'no result', 'no result']

        # stationary_model1 results
        if i < number_of_window_in_stationary_training_data * window:
            test_error3 = "no result"
            test_time3 = "no result"
            result3 = ["no result", "no result", 'no result', 'no result', 'no result']
        elif i >= number_of_window_in_stationary_training_data * window:
            test_error3 = 0
            test_time3 = 0
            result3 = []
            start_test_time3 = timeit.default_timer()
            y_pred3 = stationary_model.predict(X_test)
            test_time3 = timeit.default_timer() - start_test_time3
            test_error3 = mean_squared_error(y_test, y_pred3)
            result3 = reg_eval(y_test, y_pred3)

# random baseline
        X_train = train
        y_train = labels


        # random baseline approach 1
        # Define the range based on the min and max stock prices from the training data
        min_val = y_train.min()
        max_val = y_train.max()
        # Generate random predictions for the test set
        n_predictions = len(X_test)  # Number of samples in the test set
        random_baseline_predictions = np.random.uniform(min_val, max_val, n_predictions)

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, random_baseline_predictions)
        # mse_ls.append(mse)


        # random baseline approach 2
        # 1. Mean Baseline Model
        # mean_baseline_pred = np.mean(y_train)  # Constant prediction using the mean of the training target
        #
        # # Predict the mean for all test samples
        # mean_predictions = np.full_like(y_test, mean_baseline_pred)
        #
        # # Evaluate the mean baseline model
        # mse = mean_squared_error(y_test, mean_predictions)
        if len(result2) != 0:
            mse_ls.append(mse)
        else:
            pass


        eval_df_monthly.loc[len(eval_df_monthly)] = [f"trained on month {(i / window) - 1}",
                                                     f'tested on month {i / window}'] + result + [train_time,
                                                                                                  test_error,
                                                                                                  test_time] + [
                                                        test_error3, test_time3] + result3 + [test_error2,
                                                                                              test_time2] + result2
        print("\n")
    print("\n")
    print("total_reuse_time is : ", total_reuse_time)
    print("\n")

    sum_msr_on_next_month = 0
    # next_month_ls = list(similarity_dict.values())
    # for nex_month in next_month_ls:
    #     sum_msr_on_next_month += eval_df_monthly.iloc[nex_month, 8]
    #     print("eval_df_monthly.iloc[nex_month, 8] is: ", eval_df_monthly.iloc[nex_month, 8])
    # print("avg_msr_on_next_month is: ",sum_msr_on_next_month/len(next_month_ls))
    # print("eval_df_monthly.iloc[0, 8] is: ", eval_df_monthly.iloc[0, 8])

    # print("similarity_dict", similarity_dict)
    # print("next_month_of_similar_months_ind is", next_month_of_similar_months_ind)
    for r in range(0, len(next_month_of_similar_months_ind)):
        next_month_row_no = next_month_of_similar_months_ind[r]
        print("next_month_row_no is", next_month_row_no)
        sum_msr_on_next_month += eval_df_monthly.iloc[next_month_row_no, 8]
        print('eval_df_monthly.iloc[next_month_row_no, 8] is:  ', eval_df_monthly.iloc[next_month_row_no, 8])
    print('sum_msr_on_next_month is : ', sum_msr_on_next_month)



    eval_df_monthly2 = eval_df_monthly[eval_df_monthly['stationary_model_testing_error'] != 'no result']

    eval_df_monthly2[
        'ideal:True _is_testing_error_of_model_trained_on_previous_month_greater_than_reused_model_testing_error'] = \
    eval_df_monthly2['reused_model_testing_error'] < eval_df_monthly2['Testing Error']

    eval_df_monthly2['Ideal:True_ is_stationary_testing_error_greater_than_reused_model_testing_error'] = \
    eval_df_monthly2['reused_model_testing_error'] < eval_df_monthly2['stationary_model_testing_error']

    eval_df_monthly2[
        'ideal: True_ is_stationary_testing_error_greater_than_testing_error_of_model_trained_on_previous_month'] = \
    eval_df_monthly2['Testing Error'] < eval_df_monthly2['stationary_model_testing_error']

    eval_df_monthly2['merge of reuse error and test error'] = eval_df_monthly2.apply(
        lambda row: row['reused_model_testing_error'] if row['reused_model_testing_error'] != 0 else row[
            'Testing Error'], axis=1)

    # print('testing error mean is: ', eval_df_monthly2['Testing Error'].mean())
    # print('stationary_model_testing_error mean is: ', eval_df_monthly2['stationary_model_testing_error'].mean())
    # print('reused_model_testing_error mean is: ', eval_df_monthly2['reused_model_testing_error'].mean())
    # print('merge of reuse error and test error mean is: ',
    #       eval_df_monthly2['merge of reuse error and test error'].mean())
    print('\n')

    filtered_eval_df = eval_df_monthly2[eval_df_monthly2['reused_model_testing_error'] != 0]

    mean_stationary_model_testing_error_of_months_in_similar_dict = filtered_eval_df[
        'stationary_model_testing_error'].mean()
    mean_periodical_testing_error_of_months_in_similar_dict = filtered_eval_df['Testing Error'].mean()
    mean_reused_model_testing_error_of_months_in_similar_dict = filtered_eval_df['reused_model_testing_error'].mean()

    print('mean_stationary_model_testing_error_of_months_in_similar_dict is: ',
          mean_stationary_model_testing_error_of_months_in_similar_dict)
    print('mean_periodical_testing_error_of_months_in_similar_dict is: ',
          mean_periodical_testing_error_of_months_in_similar_dict)
    print('mean_reused_model_testing_error_of_months_in_similar_dict is: ',
          mean_reused_model_testing_error_of_months_in_similar_dict)

    average = sum(mse_ls) / len(mse_ls)
    print("Average mse in random model is: ", average)
    return eval_df_monthly2





import re
def preprocess_feature_names(df):
    feature_names = df.columns
    # Define a regex pattern to replace invalid characters with underscores
    pattern = re.compile(r'[\[\],<>]')
    # Replace invalid characters with underscores
    feature_names = [pattern.sub('_', name) for name in feature_names]
    df.columns = [str(name) for name in feature_names]
    return df




def cut_data(df):
    # index = (df['year'] == 2016) & (df['month'] == 1) & (df['day'] == 1)
    # row_index = index[index].index[0]
    # df = df[0: row_index]

    index = (df['year'] == 2020) & (df['month'] == 1) & (df['day'] == 1)
    row_index = index[index].index[0]

    index_fin = (df['year'] == 2021) & (df['month'] == 1) & (df['day'] == 1)
    row_index_fin = index_fin[index_fin].index[0]

    # df = df[0 : row_index]
    df = df[row_index:row_index_fin]

    return df




def convert_timestamp(df, date_col_name):
    df[date_col_name] = pd.to_datetime(df[date_col_name])
    return df



def random_baseline(train, test, one_month_window_size, target_col, drop_columnss):
    total_time_start = timeit.default_timer()

    labels = train[target_col]
    train = train.drop(columns=drop_columnss)

    # if model_version == 1:
    #     stationary_model, train_time = train_model(train, labels)
    # elif model_version == 2:
    #     stationary_model, train_time = train_model2(train, labels)
    train_time = 0
    mse_ls = []
    # 30 days* 144 data points per day = 4320
    window = one_month_window_size
    for i in range(window, len(test) + window, window):
        # print(i)
        sub_test = test[i:i + window]
        if len(sub_test) < window:
            break
        y_test = sub_test[target_col]
        X_test = sub_test.drop(columns=drop_columnss)
        X_train = train
        y_train = labels

        # random baseline approach 1
        # Define the range based on the min and max stock prices from the training data
        min_val = y_train.min()
        max_val = y_train.max()
        # Generate random predictions for the test set
        n_predictions = len(X_test)  # Number of samples in the test set
        random_baseline_predictions = np.random.uniform(min_val, max_val, n_predictions)

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, random_baseline_predictions)
        mse_ls.append(mse)
        #
        #
        # # random baseline approach 2
        # # 1. Mean Baseline Model
        # mean_baseline_pred = np.mean(y_train)  # Constant prediction using the mean of the training target
        #
        # # Predict the mean for all test samples
        # mean_predictions = np.full_like(y_test, mean_baseline_pred)
        #
        # # Evaluate the mean baseline model
        # mse = mean_squared_error(y_test, mean_predictions)
        # mse_ls.append(mse)

    average = sum(mse_ls) / len(mse_ls)
    print("Average mse is: ", average)
    #
    #
    #
    #     start_test_time = timeit.default_timer()
    #     y_pred = stationary_model.predict(X_test)
    #     test_time = timeit.default_timer() - start_test_time
    #     test_error = mean_squared_error(y_test, y_pred)
    #     result = reg_eval(y_test, y_pred)
    #     # print("result is: ",result)
    #     eval_df.loc[len(eval_df)] = ["trained on first window size",
    #                                  'tested on each next months seperately'] + result + [train_time, test_error,
    #                                                                                       test_time]
    #
    # total_time = timeit.default_timer() - total_time_start
    # print("\n")
    # print("total_time is: ", total_time)
    # return eval_df, stationary_model


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
