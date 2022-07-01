from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
import chardet
from pathlib import Path
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

filename = "Delaydataset.csv"
#########################################################################
### UTILS
#########################################################################
def get_encoding_type(fname):
    detected = chardet.detect(Path(fname).read_bytes())
    encoding = detected.get("encoding")
    assert encoding, "Unable to detect encoding, is it a binary file?"
    return encoding

def read_csv_file_(fname , date_column_name , auto_infer_date_bool):
    my_df = pd.read_csv(fname,parse_dates =[date_column_name] , infer_datetime_format = auto_infer_date_bool , encoding = get_encoding_type(fname))
    return my_df

def plot_df_contents(dataframe_to_plot , column_list_to_drop):
    dataframe_to_plot.drop(column_list_to_drop, axis = 1).plot()
    plt.show()

################ Get Master dataset ######################################
master_dataset_df = read_csv_file_(filename,'Date/Year',True)
print(list(master_dataset_df.columns))
print(master_dataset_df.head(30))
#plot_df_contents(master_dataset_df , ['Date/Year'])
#########################################################################

#########################################################################
### UTILS
#########################################################################
def encode_date_as_day_of_week(dataframe_to_encode , datetime_col_name):
    dataframe_to_encode['Day of Week'] = dataframe_to_encode[datetime_col_name].dt.dayofweek

def encode_dayofweek_to_number(dataframe_to_encode , column_name_to_encode):
    days = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 7}
    dataframe_to_encode[column_name_to_encode] = dataframe_to_encode[column_name_to_encode].map(days)

def encode_date_as_day_of_year(dataframe_to_encode , datetime_col_name):
    dataframe_to_encode['Day of Year'] = dataframe_to_encode[datetime_col_name].dt.dayofyear

def drop_pandas_column(df_to_process , list_of_cols):
    trimmed_df = df_to_process.drop(list_of_cols, axis = 1)
    return trimmed_df

#########################################################################

########################################################################
### Prediction for Cancellation forecast
########################################################################

cancellation_forecast_cleaned_df = drop_pandas_column(master_dataset_df, ['Flights_on_time_percentage' , 'Flights_delayed_percentage' , 'Flights_cancelled_percentage'])

encode_dayofweek_to_number(cancellation_forecast_cleaned_df, 'Day of Week')

print('cancellation_forecast_cleaned_df : \n ', cancellation_forecast_cleaned_df.head(10))
print('cancellation_forecast_cleaned_df columns : \n ', list(cancellation_forecast_cleaned_df.columns))

encode_date_as_day_of_year(cancellation_forecast_cleaned_df, 'Date/Year')

cancellation_forecast_df_final = cancellation_forecast_cleaned_df.set_index('Date/Year')

print('cancellation_forecast_df_final : \n ', cancellation_forecast_df_final.head(10))
print('cancellation_forecast_df_final columns : \n ', list(cancellation_forecast_df_final.columns))

# #########################################################################
# ####Test Train data preparation ZONE for Cancellation forecast
# #########################################################################
x1 , x2 , x3 , x4 , y = cancellation_forecast_df_final['Day of Week'] , \
                        cancellation_forecast_df_final['Day of Year'] ,\
                        cancellation_forecast_df_final['Flights_on_time'] , \
                        cancellation_forecast_df_final['Flights_delayed'] , \
                        cancellation_forecast_df_final['Flights_cancelled']

x1 , x2 , x3 , x4 , y = np.array(x1) , np.array(x2) , np.array(x3) , np.array(x4) , np.array(y)

x1 , x2 , x3 , x4 , y = x1.reshape(-1,1) , x2.reshape(-1,1) , x3.reshape(-1,1) , x4.reshape(-1,1) , y.reshape(-1,1)

X_final = np.concatenate((x1, x2, x3, x4), axis=1)
print('final_training_dataset : \n ', X_final)

X_train , X_test , y_train , y_test = train_test_split( X_final ,y ,test_size=0.20)
# #########################################################################
#
# #########################################################################
# #### Model preparation ZONE for Cancellation forecast
# #########################################################################
linear_model = LinearRegression()
randomForest_model = RandomForestRegressor(n_estimators=200 , max_features=4 , random_state=5)

linear_model.fit(X_train , y_train)
randomForest_model.fit(X_train,y_train)

print('Training done!')

print('Unseen test data on for delay case on which prediction will happen : \n\n', X_test)

linear_regression_pred = linear_model.predict(X_test)
randomforest_regression_pred = randomForest_model.predict(X_test)

plt.rcParams["figure.figsize"] = (12,8)
plt.plot(linear_regression_pred , label = 'LR prediction on Cancellation')
plt.plot(y_test,label= 'Actual Delay Data')
plt.legend(loc="upper left")
plt.show()


plt.rcParams["figure.figsize"] = (12,8)
plt.plot(randomforest_regression_pred , label = 'RF prediction on Cancellation')
plt.plot(y_test,label= 'Actual Delay Data')
plt.legend(loc="upper left")
plt.show()

rmse_random_forest = sqrt(mean_squared_error(randomforest_regression_pred , y_test))
rmse_linear_regression = sqrt(mean_squared_error(linear_regression_pred , y_test))

print('Mean squared error from random forest',rmse_random_forest)
print('Mean squared error from linear regression',rmse_linear_regression)

# #########################################################################

# #######################################################################
# ## preparation and prediction of data from first of july cancellation
# #######################################################################

predictionset_cancellation = pd.read_csv('cancellation_prediction_set.csv')
print(list(predictionset_cancellation.columns))
predictionset_cancellation['Dates'] = pd.to_datetime(predictionset_cancellation['Dates'], format="%d-%m-%Y")

encode_date_as_day_of_week(predictionset_cancellation, 'Dates')

encode_date_as_day_of_year(predictionset_cancellation, 'Dates')

predictionset_cancellation = predictionset_cancellation.set_index('Dates')

print('prediction_set final : \n ', predictionset_cancellation.head(10))
print('prediction_set columns : \n ', list(predictionset_cancellation.columns))

x1,x2,x3,x4 = predictionset_cancellation['Day of Week'] , predictionset_cancellation['Day of Year'] , predictionset_cancellation['ontime_forecast_data'] , predictionset_cancellation['delay_forecast_data']
x1 , x2 , x3 , x4 = np.array(x1) , np.array(x2) , np.array(x3) , np.array(x4)

x1 , x2 , x3 , x4= x1.reshape(-1,1) , x2.reshape(-1,1) , x3.reshape(-1,1) , x4.reshape(-1,1)

X_final_prediction_set_cancel = np.concatenate((x1, x2, x3, x4), axis=1)
print('final_prediction_dataset : \n ', X_final_prediction_set_cancel)

lr_pred_output_cancellation = linear_model.predict(X_final_prediction_set_cancel)
rf_pred_output_cancellation = randomForest_model.predict(X_final_prediction_set_cancel)

predictionset_cancellation['cancellation_prediction_lr_output'] = lr_pred_output_cancellation
predictionset_cancellation['cancellation_prediction_rf_output'] = rf_pred_output_cancellation

predictionset_cancellation.to_csv('prediction_set_cancellation_output.csv')