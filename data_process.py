from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import torch
import pickle
import pandas as pd
import my_parameter
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
from torch_geometric.data import DataLoader
from sklearn.preprocessing import StandardScaler


class HighWayDataSet(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(HighWayDataSet, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['highway_data.pt']

    def download(self):
        pass

    def process(self):

        data_list = []
        road_network_df = pd.read_csv(my_parameter.ROAD_NETWORK_PATH)
        # './allstation_alldata_3.csv'
        # './allstation_alldata_3_14_scal.csv'
        # './allstation_alldata_4.csv'
        node_feature_df = pd.read_csv(my_parameter.NODE_FEATURE_PROCESSED_PATH)

        station_dict = {}
        for index, row in road_network_df.iterrows():
            station_dict[row['StationName']] = row['StationID']

        # 构造边
        with open(my_parameter.SID_ID_PROJECTION_PATH, 'rb') as f:
            sid_id_projection = pickle.load(f, encoding='bytes')

        edge_index = []
        road_network_df = road_network_df.sort_values('StationID')
        for index, row in road_network_df.iterrows():
            sid_id = sid_id_projection[str(row['StationID'])]
            connect_station_str_list = row['connect_station'].split('|')
            for connect_station_str in connect_station_str_list:
                connect_station_id = sid_id_projection[str(station_dict[connect_station_str])]
                edge_index.append([sid_id, connect_station_id])

        # 构造顶点特征
        node_feature_df = node_feature_df.sort_values(['timestamp', 'sid'])
        # node_feature_df=node_feature_df.drop(columns=['v1','v2','v3'],axis=1)
        timestamp_list = node_feature_df['timestamp'].unique()
        for timestamp in timestamp_list:
            df = node_feature_df[node_feature_df['timestamp'] == timestamp]
            # 2:-1
            features = df.iloc[:, 2:-1].values
            features_tensor = torch.tensor(features, dtype=torch.float)

            # 提取label标签
            label_tensor = torch.tensor(df.iloc[:, -1].values, dtype=torch.float)

            # 构造图
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
            x = features_tensor
            y = label_tensor

            data = Data(x=x, edge_index=edge_index_tensor.t().contiguous(), y=y)

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def add_history_feature(history_window,feature_node_df):
    """
    add history features
    :param history_window: int,
    :param feature_node_df: DataFrame
    :return:
    """
    feature_node_df[['sid', 'timestamp']] = feature_node_df[['sid', 'timestamp']].astype(str)
    drop_before_time = feature_node_df[history_window * 269:]
    node_value_dict = dict(feature_node_df.groupby('sid')['v'].apply(list))
    drop_before_time_sort = drop_before_time.sort_values(['sid', 'timestamp'], ascending=[True, True])
    for i in range(history_window):
        value_list = []
        for key, data in node_value_dict.items():
            value_list.extend(data[i:-(history_window - i)])
        drop_before_time_sort["before_{}".format(history_window - i)] = value_list

    drop_before_time_sort['label'] = drop_before_time_sort['v']
    drop_before_time_sort = drop_before_time_sort.drop('v', axis=1)
    return drop_before_time_sort

def drop_useless_node(feature_df,node_df):
    """
    drop some nodes that don't have features
    :param feature_df:
    :param node_df:
    :return:
    """
    station_id_list = node_df['StationID'].values
    index_boolean = [True if index in station_id_list else False for index in feature_df['sid']]

    feature_df_drop_node = feature_df[index_boolean]
    return feature_df_drop_node

def get_sid_id_projectoin(node_feature):
    """
    :param node_feature: DataFrame
    :return: dict
    """
    sid_list = node_feature['sid'].unique()
    sid_id_projection = {}
    for index, sid in enumerate(sid_list):
        sid_id_projection[sid] = index

    with open(my_parameter.SID_ID_PROJECTION_PATH, 'wb') as f:
        pickle.dump(sid_id_projection, f)
    return sid_id_projection

def get_lstm_data(data):
    """
    :param data:
    :return: list
    """
    x, edge_index = data.x, data.edge_index

    # LSTM
    x_temporal = x[:, 5:]
    num_row, num_column = x_temporal.size()[0], x_temporal.size()[1]

    batch_size = int(num_row / my_parameter.TOTAL_NUM)
    lstm_data = []

    for batch_num in range(batch_size):
        seq_list = []
        for col in range(num_column):
            row_index = batch_num * my_parameter.TOTAL_NUM
            seq_list.append(x[row_index:row_index + my_parameter.TOTAL_NUM, col].numpy())
        lstm_data.append(seq_list)
    return lstm_data

def process_label(label):
    row_num = label.size()[0]
    batch_size = int(row_num / my_parameter.TOTAL_NUM)

    re = []
    for batch in range(batch_size):
        index = batch * my_parameter.TOTAL_NUM
        re.append(label[index:index + my_parameter.TOTAL_NUM].numpy())
    return re

def process_lstm_output(x_temporal):
    row_num = x_temporal.size()[0]
    output_data=torch.tensor([]).to('cuda')
    for row in range(row_num):
            output_data=torch.cat((output_data,x_temporal[row]),0)
    return output_data.reshape(-1,1)

def scal_data(data,path,is_train=True):
    """
    :param data:
    :param is_train:
    :return:
    """
    if is_train:
        scal=StandardScaler()
        x=scal.fit_transform(data)

        with open(path,'wb') as f:
            pickle.dump(scal,f)
    else:
        with open(path,'r') as f:
            scal=pickle.load(f)
            x=scal.transform(data)
    return x

def scal_data_inverse(data,path):
    """
    :param data:
    :return:
    """
    with open(path,'rb') as f:
        scal=pickle.load(f)
        inverse_data=scal.inverse_transform(data)
    return inverse_data


def scal_process(node_feature):
    """
    :param node_feature: DataFrame
    :return:
    """
    col_list = node_feature.columns[4:-1]
    print(col_list)
    for col in col_list:
        node_feature[col + '_scal'] = (node_feature[col] - node_feature[col].min()) / (
                node_feature[col].max() - node_feature[col].min())

    scal_df = node_feature.drop(columns=col_list, axis=1)
    label = scal_df['label']
    scal_df_1 = scal_df.drop(columns=['label'], axis=1)
    scal_df_1['label'] = label
    return scal_df_1

def evaluate(model,device,loader,is_test=True):
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for data in loader:
            lstm_data=get_lstm_data(data)
            lstm_data=torch.tensor(np.array(lstm_data)).to(device)

            data = data.to(device)
            pred = model(data,lstm_data).detach().cpu().numpy()


            label = data.y.detach().cpu().numpy()
            if is_test:
                pred=scal_data_inverse(pred,my_parameter.TEMPORAL_FEATURE_SCAL_PKL)
                label = scal_data_inverse(label, my_parameter.TEMPORAL_FEATURE_SCAL_PKL)

            predictions.extend(pred[:,0])
            labels.extend(label)
    #predictions = np.hstack(predictions)
    #labels = np.hstack(labels)
    return np.sqrt(mean_squared_error(labels, predictions)),r2_score(labels,predictions),predictions

def split_trian_test(dataset,train_prop=-15):
    """
    :param dataset:
    :param train_prop:
    :return:
    """
    # -14
    train_dataset = dataset[:train_prop]
    #train_dataset = train_dataset.shuffle()
    val_dataset = dataset[train_prop:]
    test_dataset = dataset[train_prop:]
    return train_dataset,val_dataset,test_dataset

def get_data_loader(batch_size,train_dataset,val_dataset,test_dataset):
    """
    :param batch_size:
    :param train_dataset:
    :param val_dataset:
    :param test_dataset:
    :return:
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader,val_loader,test_loader

def node_process(feature_df,node_df,history_window=15):
    """
    :param feature_df:DataFrame, node feature
    :param node_df:DataFrame, road network
    :param history_window: int,  the amount of period of looking back
    :return:
    """
    feature_df_drop_node=drop_useless_node(feature_df,node_df)

    scal_data_df=feature_df_drop_node.copy()

    scal_data_df.loc[:,['v1','v2','v3']]=scal_data(data=feature_df_drop_node[['v1','v2','v3']],
                                                     path=my_parameter.OTHER_FEATURE_SCAL_PKL,
                                                     is_train=True)
    scal_data_df['v']=scal_data(data=np.array(feature_df_drop_node['v']).reshape(-1,1),
                                        path=my_parameter.TEMPORAL_FEATURE_SCAL_PKL,
                                        is_train=True)
    scal_data_df=add_history_feature(history_window,scal_data_df)

    sid_id_projection=get_sid_id_projectoin(scal_data_df)

    #scal_df_1=scal_process(drop_before_time_sort)
    #scal_df_1.to_csv(my_parameter.NODE_FEATURE_PROCESSED_PATH, index=False)

    scal_data_df.to_csv(my_parameter.NODE_FEATURE_PROCESSED_PATH,index=False)


def combin_feature_pred(path,pred,pred_name,test_prob):
    """
    :param path:
    :param pred:
    :param pred_name:
    :param test_prob:
    :return:
    """
    process_feature = pd.read_csv(path)
    process_feature = process_feature.sort_values(by=['timestamp', 'sid'])
    test_data = process_feature.iloc[test_prob * my_parameter.TOTAL_NUM:, :].copy()
    test_data['label']=scal_data_inverse(test_data['label'],my_parameter.TEMPORAL_FEATURE_SCAL_PKL)
    for index in range(len(pred)):
        test_data[pred_name[index]]=pred[index]

    test_data.to_csv("./re/result.csv",index=False)
