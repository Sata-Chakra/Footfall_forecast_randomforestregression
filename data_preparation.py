import json
import datetime
import pandas as pd
from ggplot import ggplot
from ggplot import geom_point, aes
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sea

#newData = pd.read_excel('Book1.xlsx')
from sklearn.preprocessing import StandardScaler


def clean():
    newData = pd.read_csv('YearlyFlightStats.tsv', sep='\t')
    print(list(newData.columns))
    newData['Date_formatted'] = pd.to_datetime(newData['Date'])
    newData['Date_formatted'] = newData['Date_formatted'].dt.date

    # print(newData.head(20))
    # print(newData.dtypes)

    newData = newData.fillna(0)
    newData['Date_formatted'] = pd.to_datetime(newData['Date_formatted'])
    newData['Day_of_Year'] = newData['Date_formatted'].dt.strftime('%j')

    newData['Date_formatted'] = pd.to_datetime(newData['Date_formatted']).dt.strftime('%d/%m')

    newData = newData.sort_values(by='Day_of_Year')
    # print(newData.head(20))

    #half_yearly_data = newData.head(180)
    Yearly_final_data = newData
    Yearly_final_data.to_csv('Cleaned_Data.csv')

    # x = half_yearly_data['Date_formatted']
    # y = half_yearly_data['2019']
    #
    # a = half_yearly_data['Date_formatted']
    # b = half_yearly_data['2020']
    #
    # i = half_yearly_data['Date_formatted']
    # j = half_yearly_data['2021']
    #
    # k = half_yearly_data['Date_formatted']
    # l = half_yearly_data['2022']

    # plt.plot(x, y, color='red', marker='o', linestyle='solid',label='2019')
    # plt.plot(a, b, color='blue', marker='o', linestyle='dotted',label='2020')
    # plt.plot(i, j, color='green', marker='o', linestyle='dashdot',label='2021')
    # plt.plot(k, l, color='yellow', marker='o', linestyle='dashed',label='2022')
    # plt.xlabel("Dates on which the footfall was recorded")
    # plt.title("Blue for 2020 , red for 2019 , green for 2021 and yellow for 2022")
    # plt.ylabel("Footfall in millions")
    # plt.show()

    ## Prediction on the flow
    # # To be reviewd
    # X = half_yearly_data.drop(['Date','Date_formatted'],axis=1)
    # y = half_yearly_data['2022']
    #
    # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
    #
    # sc = StandardScaler()
    # normed_train_data = pd.DataFrame(sc.fit_transform(X_train), columns = X.columns)
    # normed_test_data = pd.DataFrame(sc.fit_transform(X_test), columns = X.columns)
    #
    # clf= RandomForestClassifier()
    # clf.fit(X_train, y_train)
    #
    # predictions = clf.predict(X_test)
    #
    # print ('Score on training data labels : ' , clf.score(X_train, y_train))
    # print('Score on testing data labels ', clf.score(X_test, y_test))
    #
    # print('Confuson matrix for the model eval : \n' , metrics.confusion_matrix(y_test, predictions))
    # print('Performance SCORE [LinearRegression] : ',LR.score(X_test,y_test))
    return Yearly_final_data



# Transpose matrix creation
# df_new = half_yearly_data.drop(['Date_formatted','Date'],axis=1)
# df_new_t = df_new.set_index('Day_of_Year').transpose()
# print('Intial feature Transposed Matrix 4 X __ Days data from Jan 1 :\n',df_new_t.head())
# print(df_new_t.columns)
# df_new_t.to_csv('Transposed_data.csv')









