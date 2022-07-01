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

########################################################################
### Prediction for Delay forecast
########################################################################

delayforecast_cleaned_df = drop_pandas_column(master_dataset_df , ['Flights_on_time', 'Flights_on_time_percentage' , 'Flights_delayed_percentage' , 'Flights_cancelled_percentage'])

encode_dayofweek_to_number(delayforecast_cleaned_df , 'Day of Week')

print('delayforecast_cleaned_df : \n ' ,delayforecast_cleaned_df.head(10))
print('delayforecast_cleaned_df columns : \n ' , list(delayforecast_cleaned_df.columns))

encode_date_as_day_of_year(delayforecast_cleaned_df , 'Date/Year')

delayforecast_df_final = delayforecast_cleaned_df.set_index('Date/Year')

print('delayforecast_df_final : \n ' ,delayforecast_df_final.head(10))
print('delayforecast_df_final columns : \n ' , list(delayforecast_df_final.columns))

# #########################################################################
# ####Test Train data preparation ZONE for delay forecast
# #########################################################################
x1 , x2 , x3 , x4 , y = delayforecast_df_final['Day of Week'] , delayforecast_df_final['Day of Year'] ,delayforecast_df_final['Total Flights'] , delayforecast_df_final['Flights_cancelled'] , delayforecast_df_final['Flights_delayed']

x1 , x2 , x3 , x4 , y = np.array(x1) , np.array(x2) , np.array(x3) , np.array(x4) , np.array(y)

x1 , x2 , x3 , x4 , y = x1.reshape(-1,1) , x2.reshape(-1,1) , x3.reshape(-1,1) , x4.reshape(-1,1) , y.reshape(-1,1)

X_final = np.concatenate((x1, x2, x3, x4), axis=1)
print('final_training_dataset : \n ', X_final)

X_train , X_test , y_train , y_test = train_test_split( X_final ,y ,test_size=0.25)
# #########################################################################
#
# #########################################################################
# #### Model preparation ZONE for delay forecast
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
plt.plot(linear_regression_pred , label = 'LR prediction on Delay Data')
plt.plot(y_test,label= 'Actual Delay Data')
plt.legend(loc="upper left")
plt.show()


plt.rcParams["figure.figsize"] = (12,8)
plt.plot(randomforest_regression_pred , label = 'RF prediction on Delay Data')
plt.plot(y_test,label= 'Actual Delay Data')
plt.legend(loc="upper left")
plt.show()

rmse_random_forest = sqrt(mean_squared_error(randomforest_regression_pred , y_test))
rmse_linear_regression = sqrt(mean_squared_error(linear_regression_pred , y_test))

print('Mean squared error from random forest',rmse_random_forest)
print('Mean squared error from linear regression',rmse_linear_regression)

# ######################################################################
# #######################################################################
# ## preparation of data for prediction on first of july week
# #######################################################################
# datelist =['01-07' , '02-07' , '03-07' , '04-07' , '05-07']
# yearlist = ['2013' , '2014' , '2015' , '2016', '2017','2018','2019','2020','2021']
#
# prediction_set_df = delayforecast_cleaned_df
#
# prediction_set_df["Date/Month"] = prediction_set_df["Date/Year"].dt.strftime('%d-%m')
#
# prediction_set_df.groupby('Date/Month')
#
# x = ''
# date_wise_dataset = pd.DataFrame()
# for x in datelist:
#     dt_df = prediction_set_df.iloc[list(prediction_set_df['Date/Month'] == str(x))]
#     date_wise_dataset = pd.concat([date_wise_dataset , dt_df])
#
#
# date_wise_dataset_final = drop_pandas_column(date_wise_dataset , ['Date/Year','Day of Week','Day of Year','Flights_delayed'])
# print('date_wise_dataset_final: \n\n' , date_wise_dataset_final.head(10))
# print(list(date_wise_dataset_final.columns))
# date_wise_dataset_final.to_csv('date_wise-dataset.csv')
# #######################################################################


# #######################################################################
# ## preparation and prediction of data from first of july delay
# #######################################################################

predictionset = pd.read_csv('predictionset.csv')
predictionset['Date/Year'] = pd.to_datetime(predictionset['Date/Year'] , format= "%d-%m-%Y")
print(list(predictionset.columns))

encode_date_as_day_of_week(predictionset, 'Date/Year')

encode_date_as_day_of_year(predictionset, 'Date/Year')

predictionset = predictionset.set_index('Date/Year')

print('prediction_set final : \n ', predictionset.head(10))
print('prediction_set columns : \n ', list(predictionset.columns))

x1,x2,x3,x4 = predictionset['Day of Week'] , predictionset['Day of Year'] , predictionset['Total Flights'] , predictionset['Flights_cancelled']
x1 , x2 , x3 , x4 = np.array(x1) , np.array(x2) , np.array(x3) , np.array(x4)

x1 , x2 , x3 , x4= x1.reshape(-1,1) , x2.reshape(-1,1) , x3.reshape(-1,1) , x4.reshape(-1,1)

X_final_prediction_set = np.concatenate((x1, x2, x3, x4), axis=1)
print('final_prediction_dataset : \n ', X_final_prediction_set)

lr_pred_output = linear_model.predict(X_final_prediction_set)
rf_pred_output = randomForest_model.predict(X_final_prediction_set)

predictionset['delay_prediction_lr_output'] = lr_pred_output
predictionset['delay_prediction_rf_output'] = rf_pred_output

predictionset.to_csv('prediction_set_delay_output.csv')

#**********************************************************************************************************************
#**********************************************************************************************************************
#**********************************************************************************************************************
#**********************************************************************************************************************

# ########################################################################
# ### Prediction for On_Time forecast
# ########################################################################

print(master_dataset_df)
print(list(master_dataset_df.columns))

ontimeforecast_cleaned_df = drop_pandas_column(master_dataset_df , ['Flights_on_time_percentage', 'Flights_delayed', 'Flights_delayed_percentage','Flights_cancelled_percentage'])

encode_dayofweek_to_number(ontimeforecast_cleaned_df , 'Day of Week')

print('ontimeforecast_cleaned_df : \n ' ,ontimeforecast_cleaned_df.head(10))
print('ontimeforecast_cleaned_df columns : \n ' , list(ontimeforecast_cleaned_df.columns))

encode_date_as_day_of_year(ontimeforecast_cleaned_df , 'Date/Year')

ontimeforecast_df_final = ontimeforecast_cleaned_df.set_index('Date/Year')

print('ontimeforecast_cleaned_df : \n ' ,ontimeforecast_df_final.head(10))
print('ontimeforecast_cleaned_df columns : \n ' , list(ontimeforecast_df_final.columns))
# ########################################################################
#
# #########################################################################
# #### Test Train data preparation ZONE for ontime forecast
# #########################################################################
x1 , x2 , x3 , x4 , y = ontimeforecast_df_final['Day of Week'] , ontimeforecast_df_final['Day of Year'] ,ontimeforecast_df_final['Total Flights'] , ontimeforecast_df_final['Flights_cancelled'] , ontimeforecast_df_final['Flights_on_time']

x1 , x2 , x3 , x4 , y = np.array(x1) , np.array(x2) , np.array(x3) , np.array(x4) , np.array(y)

x1 , x2 , x3 , x4 , y = x1.reshape(-1,1) , x2.reshape(-1,1) , x3.reshape(-1,1) , x4.reshape(-1,1) , y.reshape(-1,1)

X_final = np.concatenate((x1, x2, x3, x4), axis=1)
print('final_training_dataset_for_ontime_prediction : \n ', X_final)

X_train , X_test , y_train , y_test = train_test_split( X_final ,y ,test_size=0.25)
# #########################################################################
#
# #########################################################################
# #### Model preparation ZONE for ontime forecast
# #########################################################################
linear_model_ontime = LinearRegression()
randomForest_model_ontime = RandomForestRegressor(n_estimators=200 , max_features=4 , random_state=5)

linear_model_ontime.fit(X_train , y_train)
randomForest_model_ontime.fit(X_train,y_train)

print('Training done for Ontime prediction models!')

print('Unseen test data on which prediction will happen : \n\n', X_test)

linear_regression_pred_ontime = linear_model_ontime.predict(X_test)
randomforest_regression_pred_ontime = linear_model_ontime.predict(X_test)

plt.rcParams["figure.figsize"] = (12,8)
plt.plot(linear_regression_pred_ontime , label = 'LinearRegression prediction for Ontime Data')
plt.plot(y_test,label= 'Actual Delay Data')
plt.legend(loc="upper left")
plt.show()

plt.rcParams["figure.figsize"] = (12,8)
plt.plot(randomforest_regression_pred_ontime , label = 'RandomForest prediction for Ontime Data')
plt.plot(y_test,label= 'Actual Delay Data')
plt.legend(loc="upper left")
plt.show()

rmse_random_forest = sqrt(mean_squared_error(randomforest_regression_pred_ontime , y_test))
rmse_linear_regression = sqrt(mean_squared_error(linear_regression_pred_ontime , y_test))

print('Mean squared error from random forest for ontime prediction',rmse_random_forest)
print('Mean squared error from linear regression for ontime prediction',rmse_linear_regression)

# #######################################################################
# ## preparation and prediction of data from first of july ontime
# #######################################################################

predictionset_ontime = pd.read_csv('predictionset.csv')
predictionset_ontime['Date/Year'] = pd.to_datetime(predictionset_ontime['Date/Year'] , format= "%d-%m-%Y")
print(list(predictionset_ontime.columns))

encode_date_as_day_of_week(predictionset_ontime, 'Date/Year')

encode_date_as_day_of_year(predictionset_ontime, 'Date/Year')

predictionset_ontime = predictionset_ontime.set_index('Date/Year')

print('prediction_set final : \n ', predictionset_ontime.head(10))
print('prediction_set columns : \n ', list(predictionset_ontime.columns))

x1,x2,x3,x4 = predictionset_ontime['Day of Week'] , predictionset_ontime['Day of Year'] , predictionset_ontime['Total Flights'] , predictionset_ontime['Flights_cancelled']
x1 , x2 , x3 , x4 = np.array(x1) , np.array(x2) , np.array(x3) , np.array(x4)

x1 , x2 , x3 , x4= x1.reshape(-1,1) , x2.reshape(-1,1) , x3.reshape(-1,1) , x4.reshape(-1,1)

X_final_prediction_set_ontime = np.concatenate((x1, x2, x3, x4), axis=1)
print('final_prediction_dataset : \n ', X_final_prediction_set_ontime)

lr_pred_output_ontime = linear_model_ontime.predict(X_final_prediction_set_ontime)
rf_pred_output_ontime = randomForest_model_ontime.predict(X_final_prediction_set_ontime)

predictionset['ontime_prediction_lr_output'] = lr_pred_output_ontime
predictionset['ontime_prediction_rf_output'] = rf_pred_output_ontime

predictionset.to_csv('prediction_set_ontime_output.csv')

