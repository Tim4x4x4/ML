import os
import json

folder_path = r'C:\Users\Tim Chen\OneDrive\桌面\台大\台大機器學習\機器學習專案\html.2023.final.data\release'

def read_json_files(folder_path, stop_point):
    stop = 0
    Date = []
    for root, dirs, files in os.walk(folder_path):
        Position = []
        print(len(dirs))
        for file in files:
            Time = []
            if stop == stop_point:
                break
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    for time, value in data.items():
                        #print(f'Time: {time}')
                        entry = {'time':time, 'total': "None", 'current' : "None", 'can_park' : "None", 'open' : "None"}
                        if(value):
                            entry = {'time': time, 'total': value["tot"], 'current' : value["sbi"], 'can_park' : value["bemp"], 'open' : value["act"]}
                        Time.append(entry)
            stop += 1
            Position.append({str(file)[:9]: Time})
        Date.append({str(files):Position})    
    return Date

# Call the function and assign the result to my_list
my_list = read_json_files(folder_path, 2)
#print(my_list)
