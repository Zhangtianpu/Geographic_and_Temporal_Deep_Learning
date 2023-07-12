import pandas as pd
import numpy as np
import my_parameter

def Is_exist(data_dict,key):
    key_1=key
    key_list=key.split('_')
    key_2=key_list[1]+"_"+key_list[0]
    if key_1 in data_dict:
        return key_1
    if key_2 in data_dict:
        return key_2
    return False

df=pd.read_csv(my_parameter.ROAD_NETWORK_PATH)

count_dict={}
for idx,data in df.iterrows():
    station_name=data['StationName']
    connect_station=data['connect_station']
    connect_list=connect_station.split('|')
    for conn in connect_list:
        key=station_name+"_"+conn
        re=Is_exist(count_dict,key)
        if re:
            count_dict[re]+=1
        else:
            count_dict[key]=1

for key,data in count_dict.items():
    if data !=2:
        print(key,data)

station_dict={}
for index,row in df.iterrows():
    station_dict[row['StationName']]=row['StationID']

edge_index=[]
for index,row in df.iterrows():
    connect_station_str_list=row['connect_station'].split('|')
    for connect_station_str in connect_station_str_list:
        connect_station_id=station_dict[connect_station_str]
        edge_index.append([row['StationID'],connect_station_id])