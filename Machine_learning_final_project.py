import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

folder_path = r'C:\Users\Tim Chen\OneDrive\桌面\台大\台大機器學習\機器學習專案\html.2023.final.data\release'

def read_json_files(folder_path, stop_point):
    stop = 0
    Date = []
    Data = []
    Bike_stops = []
    for root, dirs, files in os.walk(folder_path):
        #print(str(root)[-8:])
        if(str(root)[-8:][0] != '2'):
            continue
        if stop == stop_point:
                break
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    for time, value in data.items():
                        #print(f'Time: {time}')
                        entry = {}
                        if(value):
                            entry = {'date': str(root)[-8:], 'bike_stop': str(file)[:9], 'time': time, 'total': value["tot"], 'current': value["sbi"], 'can_park' : value["bemp"], 'open' : value["act"]}
                        if str(file)[:9] not in Bike_stops:
                            Bike_stops.append(str(file)[:9])
                        Data.append(entry)
        stop += 1
        Date.append({str(root)[-8:]})                   
    return Data, Date, Bike_stops

data, date, bike_stops = read_json_files(folder_path, 3)
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
stop_point = 0
for stop in new_data:
    if stop_point == 1:
        break
    print(stop)
    X = []
    y = []
    for elem in new_data[stop]:
        X.append([time_to_minutes(elem.get('time')), elem.get('total'), elem.get('open')])
        y.append(elem.get('current'))
    X = np.array(X)
    #print(X.shape)
    y = np.array(y)
    #print(y)
    X_numeric = X.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.1, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(y_test)
    print(y_pred)  
    stop_point += 1    
        
print("Done")


