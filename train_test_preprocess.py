import data_preparation
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt


print('Hello..Trainer started!')

footfall_dataset_df = pd.read_csv('footfallNumbers.tsv', sep='\t', index_col ='Date', parse_dates =True)
print(list(footfall_dataset_df.columns))

footfall_dataset_df.plot()
plt.show()

footfall_dataset_df['Footfall_1_day_ago'] = footfall_dataset_df['Footfall_count'].shift(+1)
footfall_dataset_df['Footfall_2_day_ago'] = footfall_dataset_df['Footfall_count'].shift(+2)
footfall_dataset_df['Footfall_3_day_ago'] = footfall_dataset_df['Footfall_count'].shift(+3)
footfall_dataset_df['Footfall_4_day_ago'] = footfall_dataset_df['Footfall_count'].shift(+4)
footfall_dataset_df['Footfall_5_day_ago'] = footfall_dataset_df['Footfall_count'].shift(+5)
footfall_dataset_df['Footfall_6_day_ago'] = footfall_dataset_df['Footfall_count'].shift(+6)
footfall_dataset_df['Footfall_7_day_ago'] = footfall_dataset_df['Footfall_count'].shift(+7)

footfall_dataset_df = footfall_dataset_df.dropna()

linear_model = LinearRegression()

randomForest_model = RandomForestRegressor(n_estimators=100 , max_features=7 , random_state=5)

x1,x2,x3,x4,x5,x6,x7 , y = footfall_dataset_df['Footfall_1_day_ago'] , footfall_dataset_df['Footfall_2_day_ago'], \
                           footfall_dataset_df['Footfall_3_day_ago'], footfall_dataset_df['Footfall_4_day_ago'] , \
                           footfall_dataset_df['Footfall_5_day_ago'], footfall_dataset_df['Footfall_6_day_ago'], \
                           footfall_dataset_df['Footfall_7_day_ago'] , footfall_dataset_df['Footfall_count']

x1,x2,x3,x4,x5,x6,x7 , y = np.array(x1) ,np.array(x2) ,np.array(x3), np.array(x4) ,np.array(x5) ,np.array(x6) ,np.array(x7), np.array(y)
x1,x2,x3,x4,x5,x6,x7 , y = x1.reshape(-1,1) , x2.reshape(-1,1) , x3.reshape(-1,1) , x4.reshape(-1,1), x5.reshape(-1,1) , x6.reshape(-1,1) , x7.reshape(-1,1) , y.reshape(-1,1)

final_x = np.concatenate((x1,x2,x3,x4,x5,x6,x7),axis=1)
print(final_x)

final_x.round()
print(final_x)

X_train , X_test , y_train , y_test = final_x[:-300] , final_x[-300:] , y[:-300] , y[-300:]
linear_model.fit(X_train , y_train)
randomForest_model.fit(X_train,y_train)

print('Training done!')
print(X_test)
linear_regression_pred = linear_model.predict(X_test)
randomforest_regression_pred = randomForest_model.predict(X_test)

plt.rcParams["figure.figsize"] = (12,8)
plt.plot(linear_regression_pred , label = 'Liner Regression Model')
plt.plot(y_test,label= 'Actual footfall')
plt.legend(loc="upper left")
plt.show()

plt.rcParams["figure.figsize"] = (12,8)
plt.plot(randomforest_regression_pred , label = 'Random Forest Prediction Model')
plt.plot(y_test,label= 'Actual footfall')
plt.legend(loc="upper left")
plt.show()

rmse_random_forest = sqrt(mean_squared_error(randomforest_regression_pred , y_test))
rmse_linear_regression = sqrt(mean_squared_error(linear_regression_pred , y_test))

print('Mean squared error from random forest',rmse_random_forest)
print('Mean squared error from linear regression',rmse_linear_regression)

############################################ Forecasting on the Future ######################################

rf_pred_list =[]
lr_pred_list = []

test_rf = [2085927, 2364754, 2371014, 2155747, 2052377, 2279743, 2387196]
test_lr = [2085927, 2364754, 2371014, 2155747, 2052377, 2279743, 2387196]

for i in range(30):
    #my_pred_variable = randomForest_model.predict([[2085927,2364754,2371014,2155747,2052377,2279743,2387196]])
    rf_var = randomForest_model.predict([test_rf])
    lr_var = linear_model.predict([test_lr])

    rf_pred_list.append(round(rf_var[0]))
    lr_pred_list.append(round(lr_var.item(0)))

    del test_rf[-1:]
    test_rf.insert(0, round(rf_var[0]))

    del test_lr[-1:]
    test_lr.insert(0, round(lr_var.item(0)))


print('Prediction list for 30 days from june 11 by random Forest model : \n' , rf_pred_list)
print('\nPrediction list for 30 days from june 11 by linear regression model : \n' , lr_pred_list)

predicted_dataset_rf_df = pd.read_csv('outputs/Prediction_result_random_forest.tsv', sep='\t', index_col ='Future Dates for the Model', parse_dates =True)
predicted_dataset_lr_df = pd.read_csv('outputs/Prediction_result_linear_regression.tsv', sep='\t', index_col ='Future Dates for the Model', parse_dates =True)

predicted_dataset_rf_df.to_csv('rf_output.csv')
predicted_dataset_lr_df.to_csv('lr_output.csv')

original_dataset_df = pd.read_csv('footfallNumbers.tsv', sep='\t', index_col ='Date', parse_dates =True)

plt.rcParams["figure.figsize"] = (12,8)
plt.plot(original_dataset_df , color='green', marker='o', linestyle='solid' , label = 'Original Footfall Line')
plt.plot(predicted_dataset_rf_df,color='red', marker='o', linestyle='solid' , label= 'Predicted Footfall by Random forest')
plt.legend(loc="upper left")
plt.show()

plt.rcParams["figure.figsize"] = (12,8)
plt.plot(original_dataset_df , color='green', marker='o', linestyle='solid' , label = 'Original Footfall Line')
plt.plot(predicted_dataset_lr_df,color='red', marker='o', linestyle='solid' , label= 'Predicted Footfall by Linear Regression')
plt.legend(loc="upper left")
plt.show()

df_comparison_4july = pd.read_csv('comparison_chart_4july.tsv' , sep='\t', index_col ='Dates', parse_dates =True)
df_comparison_4july = df_comparison_4july.sort_values(by='Dates')
print(df_comparison_4july.head(10))
df_comparison_4july.plot()
plt.show()