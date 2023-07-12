import torch

from tqdm import tqdm

import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
import data_process
import my_parameter
import numpy as np


class MyNet(nn.Module):

    def __init__(self, hidden_channels,num_features,input_size,hidden_size,ouput_size,num_layer=2):
        super(MyNet, self).__init__()

        self.input_size=input_size
        self.conv1 = GCNConv(in_channels=num_features, out_channels=hidden_channels)
        self.conv2 = GCNConv(in_channels=hidden_channels, out_channels=hidden_channels)
        self.lstm=nn.LSTM(input_size,hidden_size,num_layer,batch_first=True)
        self.lin1 = nn.Linear(in_features=hidden_channels+3, out_features=64)
        self.lin2 = nn.Linear(in_features=64, out_features=64)
        self.lin3 = nn.Linear(in_features=64, out_features=1)

    def forward(self, data,lstm_data):
        x, edge_index = data.x, data.edge_index
        external_data=x[:,0:2]
        #LSTM

        #x_temporal=x_temporal.reshape(my_parameter.HISTORY_WINDOW,-1,my_parameter.TOTAL_NUM)
        x_temporal,_=self.lstm(lstm_data)
        x_temporal=x_temporal[:,-1,:] #[60,269]
        x_temporal=data_process.process_lstm_output(x_temporal)
        x_temporal=x_temporal.reshape(-1,1)

        #GCN

        x=x[:,2:]
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        #FNN
        x_merge = torch.cat((x, x_temporal,external_data),dim=1)
        x_merge = F.relu(x_merge)
        x_merge = self.lin1(x_merge)
        x_merge = F.relu(x_merge)
        x_merge = self.lin2(x_merge)
        x_merge = F.relu(x_merge)
        x_merge = self.lin3(x_merge)

        return x_merge

def _train(model,train_loader,device,dataset_length):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=my_parameter.LEARNING_RATE, weight_decay=5e-4)
    #optimizer = torch.optim.SGD(model.parameters(), lr=my_parameter.LEARNING_RATE, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()
    loss_all=0

    for data in train_loader:
        lstm_data=data_process.get_lstm_data(data)
        lstm_data=torch.tensor(np.array(lstm_data)).to(device)

        label = data.y
        #label = torch.tensor(data_process.process_label(label))
        label = label.to(device)

        data=data.to(device)

        optimizer.zero_grad()

        output=model(data,lstm_data)


        loss=criterion(output[:,0],label)
        loss_all += data.num_graphs * loss.item()
        loss.backward()
        optimizer.step()
    return loss_all / dataset_length


def train(model,train_loader,test_loader,device,dataset_length):

    for epoch in tqdm(range(1, my_parameter.TRAIN_STEPS)):
        loss = _train(model,train_loader,device,dataset_length)

        train_mse, train_r2 ,pred= data_process.evaluate(model,device,train_loader,is_test=False)
        if (epoch + 1) % (my_parameter.TRAIN_STEPS/5) == 0:
            #val_mse, val_r2 = data_process.evaluate(model,device,test_loader,is_test=True)
            test_mse, test_r2,pred = data_process.evaluate(model,device,test_loader,is_test=True)
            print(
                'Epoch: {:03d}, Loss: {:.5f}, Train RMSE: {:.5f},Train R2: {:.5f}, Test RMSE: {:.5f}, Test R2: {:.5f}'.
                format(epoch, loss, train_mse, train_r2, test_mse, test_r2))

def predict(model,test_loader,device):
    test_mse, test_r2,pred= data_process.evaluate(model, device, test_loader, is_test=True)
    return test_mse,test_r2,pred

def save_model(model,path):
    torch.save(model.state_dict(),path)

def load_model(model,path):
    model.load_state_dict(torch.load(path))
    return model