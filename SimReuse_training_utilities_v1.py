from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from itertools import zip_longest
from darts.models import ExponentialSmoothing
import pandas as pd
from darts.models import ExponentialSmoothing
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
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
from scipy.stats import wasserstein_distance
from itertools import combinations
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import timeit
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import re


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

def convert_time(df, date_col_name):
    df['year'] = df[date_col_name].dt.year
    df['month'] = df[date_col_name].dt.month
    df['day'] = df[date_col_name].dt.day
    df['hour'] = df[date_col_name].dt.hour
    df['minute'] = df[date_col_name].dt.minute
    return df

def forecast_es(train, test, forecast_daily_avg, forecast_avg_target_col_name):
    start_forecast_es = timeit.default_timer()
    model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=12)
    fit = model.fit()
    forecast = fit.forecast(steps=len(test))
    end_forecast_es = timeit.default_timer()
    forecast_time = end_forecast_es - start_forecast_es
    mse = mean_squared_error(test, forecast)
    results = pd.DataFrame(forecast, columns=[forecast_avg_target_col_name])
    forecast_daily_avg = pd.concat([forecast_daily_avg, results], ignore_index=True)
    return forecast_daily_avg, forecast_time

def plot_months_patterns(avg, one_month_days, forecast_or_regular_avg_target_col_name):
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
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def plot_months_patterns_box(avg, one_month_days, forecast_or_regular_avg_target_col_name):
    num_segments = len(avg) // one_month_days + 1
    colors = plt.cm.tab20c(np.linspace(0, 1, num_segments))
    fig, ax = plt.subplots()
    data_for_boxplot = []
    for i in range(0, len(avg), one_month_days):
        data_for_boxplot.append(avg[forecast_or_regular_avg_target_col_name][i:i + one_month_days].values)
    ax.boxplot(data_for_boxplot, patch_artist=True)
    for patch, color in zip(ax.artists, colors):
        patch.set_facecolor(color)
    ax.set_xlabel('Months')
    ax.set_ylabel('Daily average of target value')
    ax.set_title('Monthly box plots')
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
    daily_data_for_each_month = pd.DataFrame()
    for i, lst in enumerate(daily_col_ls):
        column_name = f'month_{i}'  # Set the name for each column
        daily_data_for_each_month[column_name] = lst
    return daily_data_for_each_month


# compute wasserstein distance
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

# Total Variation Distance
def total_variation_distance(p, q):
    return 0.5 * np.sum(np.abs(p - q))

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


def reg_eval(y_test,y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    return [mae, mse, rmse, r2, mape]

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


def train_model2_with_hyperparametertuning(X_train, y_train):
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

def model_reuse_with_hptuning(similarity_dict, one_month_window_size, df, model_version, forecast_var_bool,target_col, drop_columnss):
    out_columns = ['Training dataset', 'Testing dataset', 'model reuse status', 'reused_model_mae', 'reused_model_mse', 'reused_model_rmse', 'reused_model_r2', 'reused_model_mape']
    window = one_month_window_size
    models_ls = []
    eval_df_monthly = pd.DataFrame(columns=out_columns)
    next_month_of_similar_months_ind = []
    for i in range(window, len(df), window):
        train = df[i - window:i]
        test = df[i: i + window]
        if len(test) < window:
            break

        train = train.reset_index().drop(columns=['index'])
        test = test.reset_index().drop(columns=['index'])

        labels = train[target_col]
        train = train.drop(columns=drop_columnss)


        y_test = test[target_col]
        X_test = test.drop(columns=drop_columnss)

        if forecast_var_bool == False:
            month_index = round(i / window) - 1
        elif forecast_var_bool == True:
            month_index = round(i / window)
        if month_index in similarity_dict:
            similar_month_index = similarity_dict[month_index]
            while similar_month_index in similarity_dict:
                similar_month_index = similarity_dict[similar_month_index]
            next_month_of_similar_months_ind.append(similar_month_index)
            model2 = models_ls[similar_month_index]
            models_ls.append(model2)
            y_pred2 = model2.predict(X_test)
            result2 = [np.float64(1)]+reg_eval(y_test, y_pred2)
        else:
            if model_version == 1:
                model2, train_time = train_model_with_hyperparametertuning(train, labels)
            elif model_version == 2:
                model2, train_time = train_model2_with_hyperparametertuning(train, labels)
            models_ls.append(model2)
            y_pred2 = model2.predict(X_test)
            result2 = [np.float64(0)]+reg_eval(y_test, y_pred2)
        eval_df_monthly.loc[len(eval_df_monthly)] = [f"month {(i / window) - 1}",
                                                     f'month {i / window}']+ result2
    return eval_df_monthly

def preprocess_feature_names(df):
    feature_names = df.columns
    pattern = re.compile(r'[\[\],<>]')
    feature_names = [pattern.sub('_', name) for name in feature_names]
    df.columns = [str(name) for name in feature_names]
    return df

def cut_data(df):
    index = (df['year'] == 2020) & (df['month'] == 1) & (df['day'] == 1)
    row_index = index[index].index[0]
    index_fin = (df['year'] == 2021) & (df['month'] == 1) & (df['day'] == 1)
    row_index_fin = index_fin[index_fin].index[0]
    df = df[row_index:row_index_fin]
    return df

def convert_timestamp(df, date_col_name):
    df[date_col_name] = pd.to_datetime(df[date_col_name])
    return df



def random_baseline(train, test, one_month_window_size, target_col, drop_columnss):
    total_time_start = timeit.default_timer()

    labels = train[target_col]
    train = train.drop(columns=drop_columnss)

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

# modify the following function based on your dataset
def convert_sim_to_month(dictionary_, dataset_ind):
    new_key = []
    new_val = []
    dictionary_1 = {key + 1: value + 1 for key, value in dictionary_.items()}
    print("dictionary_1 is: ", dictionary_1)
    print("\n")
    keys_ls = list(dictionary_1.keys())
    vals_ls = list(dictionary_1.values())
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
