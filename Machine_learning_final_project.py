import os
import json
import csv
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

folder_path = r'C:\Users\Tim Chen\OneDrive\桌面\台大\台大機器學習\機器學習專案\html.2023.final.data\release'
rain_path = 'C:/Users/Tim Chen/OneDrive/桌面/台大/台大機器學習/機器學習專案/extend X feature/weather/If_rain.xlsx'
holiday_path = 'C:/Users/Tim Chen/OneDrive/桌面/台大/台大機器學習/機器學習專案/extend X feature/holiday/If_holiday.xlsx'

def read_feature(feature_path, feature_name):
    df = pd.read_excel(feature_path)
    df_dict = {}
    for index, row in df.iterrows():
        key = int(row['date'])
        value = row[feature_name]
        df_dict[str(key)] = value
    return df_dict
def read_json_files(folder_path, stop_point, rain_data, holiday_data):
    Date = []
    Data = []
    Bike_stops = []
    
    for root, dirs, files in os.walk(folder_path):
        if str(root)[-8:][0] != '2':
            continue
        stop = 0
        for file in files:
            if stop == stop_point:
                break
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
  
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    
                    for time, value in data.items():
                        if value:
                            entry = {
                                'date': str(root)[-8:],
                                'bike_stop': str(file)[:9],
                                'time': time,
                                'total': value["tot"],
                                'current': value["sbi"],
                                'can_park': value["bemp"],
                                'open': value["act"],
                                'rain_hour': rain_data[f'{str(root)[-8:]}'],
                                'holiday': holiday_data[f'{str(root)[-8:]}']
                            }
                            Data.append(entry)
                            
                            if str(file)[:9] not in Bike_stops:
                                Bike_stops.append(str(file)[:9])
            stop += 1
        
        Date.append({str(root)[-8:]})
    
    return Data, Date, Bike_stops

rain_data = read_feature(rain_path, "rain_hour")
holiday_data = read_feature(holiday_path, "holiday")
data, date, bike_stops = read_json_files(folder_path, 1, rain_data, holiday_data)
new_data = {}

for stops in bike_stops:
    new_data[stops] = []

for elem in data:
    bike_stop = elem.get('bike_stop')
    current_bike = elem.get('current')
    if bike_stop is not None:
        new_data[bike_stop].append(elem)
models = []

def time_to_minutes(time_str):
    hour, minute = map(int, time_str.split(':'))
    return hour * 60 + minute
def err(y_test, y_pred, total_stop):
    sum = 0
    for i in range(y_test.size):
        yt = y_test[i]
        yp = y_pred[i]
        sum += 3*(abs(yt-yp)/total_stop)*(abs(yt/total_stop-1/3)+abs(yt/total_stop-2/3))
    sum /= y_test.size
    return sum
stop_point = 0
best_models = {}
for stop in new_data:
    print(stop)
    X = []
    y = []
    total_stop = 0
    for elem in new_data[stop]:
        total_stop = elem.get('total')
        if elem.get('open') == 0:
            continue
        X.append([time_to_minutes(elem.get('time')), elem.get('total'), elem.get('open'), elem.get('rain_hour'), elem.get('holiday')])
        y.append(elem.get('current'))
    X = np.array(X)
    #print(X.shape)
    y = np.array(y)
    X_numeric = X.astype(float)
    smallest_error = float('inf')
    best_model = RandomForestClassifier(n_estimators=100, max_depth = 10, random_state=42) #initialize
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.1, random_state=i*42)
        
        model = RandomForestClassifier(n_estimators=100, max_depth = 10, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        #print(y_test)
        #print(y_pred)  
        #for true_value, predicted_value in zip(y_test, y_pred):
            #print(f"True: {true_value}, Predicted: {predicted_value}")
        error = err(y_test, y_pred, total_stop)
        if error < smallest_error:
            smallest_error = error
            best_model = model
    best_models[str(stop)] = best_model
    print(smallest_error)
           
print("Done")


