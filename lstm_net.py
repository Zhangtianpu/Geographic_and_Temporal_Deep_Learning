import torch

from tqdm import tqdm

import torch.nn.functional as F
import torch.nn as nn
import data_process
import my_parameter
import numpy as np

class lstm(nn.Module):
    def __init__(self,input_size=my_parameter.TOTAL_NUM,hidden_size=my_parameter.TOTAL_NUM,output_size=1,num_layer=2,sequence_lenth=10):
        super(lstm,self).__init__()
        self.layer1 = nn.LSTM(input_size,hidden_size,num_layer)
        ##self.layer3=nn.Linear(128,output_size)

    def forward(self,x):
        x=x.reshape(my_parameter.HISTORY_WINDOW,-1,1)
        x,_ = self.layer1(x)
        x=x[-1,:,:]
        x=x.reshape(-1,1)
        #s,b,h = x.size()
        #x = x.reshape(b,-1)
        #x = F.relu(self.layer2(x))
        #x=self.layer3(x)

        return x

def _train(model,X,y,device,dataset_length):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    #optimizer = torch.optim.SGD(model.parameters(), lr=my_parameter.LEARNING_RATE, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()

    X=torch.tensor(X).float().to(device)
    y=torch.tensor(y).float().to(device)
    optimizer.zero_grad()
    output=model(X)
    output=torch.squeeze(output)
    loss=criterion(output,y)
    loss.backward()
    optimizer.step()
    return loss



def train(model,train_X,train_y,device,dataset_length):

    for epoch in tqdm(range(1, my_parameter.TRAIN_STEPS)):
        loss = _train(model,train_X,train_y,device,dataset_length)
        #train_mse, train_r2 = data_process.evaluate(model,device,train_loader)
        if (epoch + 1) % (my_parameter.TRAIN_STEPS/5) == 0:
            print( 'Epoch: {:03d}, Loss: {:.5f}'.format(epoch, loss))
            # val_mse, val_r2 = data_process.evaluate(model,device,test_loader)
            # test_mse, test_r2 = data_process.evaluate(model,device,test_loader)
            # print(
            #     'Epoch: {:03d}, Loss: {:.5f}, Train RMSE: {:.5f},Train R2: {:.5f}, Val RMSE: {:.5f}, Val R2: {:.5f}, Test RMSE: {:.5f}, Test R2: {:.5f}'.
            #     format(epoch, loss, train_mse, train_r2, val_mse, val_r2, test_mse, test_r2))

def test(model,X,y,device):
    model.eval()

    with torch.no_grad():
        X=torch.tensor(X).float().to(device)
        pred = model(X).detach().cpu().numpy()

    return pred