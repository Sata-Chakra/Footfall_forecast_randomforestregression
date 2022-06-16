from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import os
import chardet
from pathlib import Path
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

filename = "Delaydataset.csv"
detected = chardet.detect(Path(filename).read_bytes())
encoding = detected.get("encoding")
assert encoding, "Unable to detect encoding, is it a binary file?"


master_dataset_df = pd.read_csv(filename, parse_dates =['Date/Year'], infer_datetime_format=True, encoding = encoding)
print(list(master_dataset_df.columns))
print(master_dataset_df.head(30))

# delayed_dataset_df.plot(marker ='o' , linestyle = 'dotted')
# plt.show()

#########################################################################
delayforecast_cleaned_df = master_dataset_df.drop(['Flights_on_time', 'Flights_on_time_percentage' , 'Flights_delayed_percentage' , 'Flights_cancelled_percentage'], axis = 1)

days = {'Mon':1 , 'Tue' : 2,'Wed':3, 'Thu':4, 'Fri':5, 'Sat' : 6 , 'Sun' : 7}
delayforecast_cleaned_df['Day of Week'] = delayforecast_cleaned_df['Day of Week'].map(days)

print('delayforecast_cleaned_df : \n ' ,delayforecast_cleaned_df.head(10))
print('delayforecast_cleaned_df columns : \n ' , list(delayforecast_cleaned_df.columns))

delayforecast_cleaned_df['Day of Year'] = delayforecast_cleaned_df['Date/Year'].dt.dayofyear

delayforecast_df_final = delayforecast_cleaned_df.set_index('Date/Year')

print('delayforecast_df_final : \n ' ,delayforecast_df_final.head(10))
print('delayforecast_df_final columns : \n ' , list(delayforecast_df_final.columns))


#########################################################################
####Test Train data preparation ZONE
#########################################################################
x1 , x2 , x3 , x4 , y = delayforecast_df_final['Day of Week'] , delayforecast_df_final['Day of Year'] ,delayforecast_df_final['Total Flights'] , delayforecast_df_final['Flights_cancelled'] , delayforecast_df_final['Flights_delayed']

x1 , x2 , x3 , x4 , y = np.array(x1) , np.array(x2) , np.array(x3) , np.array(x4) , np.array(y)

x1 , x2 , x3 , x4 , y = x1.reshape(-1,1) , x2.reshape(-1,1) , x3.reshape(-1,1) , x4.reshape(-1,1) , y.reshape(-1,1)

X_final = np.concatenate((x1, x2, x3, x4), axis=1)
print('final_training_dataset : \n ', X_final)

X_train , X_test , y_train , y_test = train_test_split( X_final ,y ,test_size=0.25)
#########################################################################

#########################################################################
#### Model preparation ZONE
#########################################################################
linear_model = LinearRegression()
randomForest_model = RandomForestRegressor(n_estimators=100 , max_features=7 , random_state=5)

linear_model.fit(X_train , y_train)
randomForest_model.fit(X_train,y_train)

print('Training done!')

print(X_test)

linear_regression_pred = linear_model.predict(X_test)
randomforest_regression_pred = randomForest_model.predict(X_test)

plt.rcParams["figure.figsize"] = (12,8)
plt.plot(linear_regression_pred , label = 'LR prediction Delay Data')
plt.plot(y_test,label= 'Actual Delay Data')
plt.legend(loc="upper left")
plt.show()


plt.rcParams["figure.figsize"] = (12,8)
plt.plot(linear_regression_pred , label = 'RF prediction Delay Data')
plt.plot(y_test,label= 'Actual Delay Data')
plt.legend(loc="upper left")
plt.show()

rmse_random_forest = sqrt(mean_squared_error(randomforest_regression_pred , y_test))
rmse_linear_regression = sqrt(mean_squared_error(linear_regression_pred , y_test))

print('Mean squared error from random forest',rmse_random_forest)
print('Mean squared error from linear regression',rmse_linear_regression)

#########################################################################