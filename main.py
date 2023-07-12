import data_process
import pandas as pd
import my_parameter
import GCN_LSTM
import torch
import numpy as np
import lstm_net

DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    #
    feature_df = pd.read_csv(my_parameter.ALLSTATION_ALLDATA_PATH)
    node_df = pd.read_csv(my_parameter.ROAD_NETWORK_PATH)
    data_process.node_process(feature_df,node_df,my_parameter.HISTORY_WINDOW)

    #create dataset for GCN
    dataset_highway = data_process.HighWayDataSet(my_parameter.HIGHWAYDATASET_PATH)

    train_dataset,val_dataset,test_dataset=data_process.split_trian_test(dataset_highway,train_prop=-15)
    print(train_dataset)
    train_loader,val_loader,test_loader=data_process.get_data_loader(my_parameter.DATA_LOADER_BATCH_SIZE,train_dataset,val_dataset,test_dataset)


    #Train HGCN hidden_channels,num_features,input_size,hidden_size,ouput_size,seq_length=15,num_layer=2
    model=GCN_LSTM.MyNet(hidden_channels=64,
                         num_features=18,
                         input_size=my_parameter.TOTAL_NUM,
                         hidden_size=my_parameter.TOTAL_NUM,
                         ouput_size=1,
                         num_layer=2).to(DEVICE)
    #GCN_LSTM.train(model, train_loader, test_loader, DEVICE, len(train_dataset))
    #GCN_LSTM.save_model(model,my_parameter.MODEL_PATH)
    model=GCN_LSTM.load_model(model,my_parameter.MODEL_PATH)
    test_mse,test_r2,pred=GCN_LSTM.predict(model,test_loader,DEVICE)
    print(test_mse,test_r2)

    # data_process.combin_feature_pred(path=my_parameter.NODE_FEATURE_PROCESSED_PATH,
    #                                  pred=[pred],
    #                                  pred_name=["gcn_lstm_pred"],
    #                                  test_prob=-15)


