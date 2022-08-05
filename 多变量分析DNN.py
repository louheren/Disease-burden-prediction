from collections import Counter
import torch
from torch import nn
from torch import optim
import math
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from torch.utils.data import random_split
from collections import Counter
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
# from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import random_split
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_y = []
data_x = []
from sklearn.preprocessing import MinMaxScaler
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data=pd.read_csv("趋势分析.csv").values

###for line  in data:
  #temp_line=[]
     #for i in line:
      #   temp_line.append(i/7090)
     #data_min_max.append(temp_line)
#data=data_min_max
#print(data)
data=[line[1:] for line in data]
data_x=[]
data_y=[]
for line in data:
    data_x.append([[i] for i in line[1:]])
    data_y.append(line[0])

print(len(data_x),len(data_y))
train_X,test_X,train_y,test_y =train_test_split(data_x,data_y,test_size=5,train_size=27,shuffle=False)
# print("65",len(test_y))
# print(test_y)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class LSTM(nn.Module):  # 注意Module首字母需要大写
    def __init__(self, ):
        super().__init__()
        input_size = 1
        output_size = 1
        self.lstm = nn.LSTM(input_size, output_size, num_layers=1)  # ,batch_first=True
        self.relu=nn.ReLU()
        self.linear_1 = nn.Linear(4,128)
        self.linear_2 = nn.Linear(128,1)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = x.type(torch.FloatTensor)
        x = x.to(device)
        x = x.unsqueeze(0)
        # # print(x,x.shape,type(x))
        # # 输入 lstm的矩阵形状是：[序列长度，batch_size,每个向量的维度] [序列长度,batch, 64]
        # lstm_out, (h_n, c_n) = self.lstm(x, None)
        # # print(lstm_out.shape)
        # lstm_out=self.relu(lstm_out)
        lstm_out=x.view(1,4)
        # print("lstm_out",lstm_out)
        lstm_out=self.linear_1(lstm_out)
        lstm_out = self.relu(lstm_out)
        prediction = self.linear_2(lstm_out)
        return prediction


#这个函数是测试用来测试x_test y_test 数据 函数
def eval_test(model):  # 返回的是这10个 测试数据的平均loss
    test_epoch_loss = []
    with torch.no_grad():
        optimizer.zero_grad()
        for setp in range(len(test_X)):
            testx = torch.tensor(test_X[setp])
            testy = torch.tensor([test_y[setp]])
            testy = testy.type(torch.FloatTensor)
            # print("120",trainx,trainy)
            y_pred = model(testx)
            # print("121",y_pred,trainy,type(y_pred),type(trainy))
            single_loss = loss_function(y_pred, testy)
            test_epoch_loss.append(single_loss.item())
    return np.mean(test_epoch_loss)
epochs = 2000
# 创建LSTM()类的对象，定义损失函数和优化器
model = LSTM().to(device)
loss_function = torch.nn.MSELoss().to(device)  # 损失函数的计算 交叉熵损失函数计算
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # 建立优化器实例
print(model)

sum_train_epoch_loss = []  # 存储每个epoch 下 训练train数据的loss
sum_test_epoch_loss = []  # 存储每个epoch 下 测试 test数据的loss
best_test_loss = 100000000000000
for epoch in range(epochs):
    epoch_loss = []
    for setp in range(len(train_X)):
        trainx= torch.tensor(train_X[setp])
        trainy=torch.tensor([train_y[setp]])
        trainy = trainy.type(torch.FloatTensor)
        # print("120",trainx,trainy)
        y_pred = model(trainx)
        # print("121",y_pred,trainy,type(y_pred),type(trainy))
        single_loss = loss_function(y_pred,trainy)
        single_loss.backward()  # 调用backward()自动生成梯度
        optimizer.step()  # 使用optimizer.step()执行优化器，把梯度传播回每个网络
        epoch_loss.append(single_loss.item())

    train_epoch_loss = np.mean(epoch_loss)
    test_epoch_loss = eval_test(model)  # 测试数据的平均loss

    if test_epoch_loss < best_test_loss:
        best_test_loss = test_epoch_loss
        print("best_test_loss", best_test_loss)
        best_model = model
    sum_train_epoch_loss.append(train_epoch_loss)
    sum_test_epoch_loss.append(test_epoch_loss)
    print("epoch:" + str(epoch) + "  train_epoch_loss： " + str(train_epoch_loss) + "  test_epoch_loss: " + str(
        test_epoch_loss))
#
torch.save(best_model, 'best_model.pth')
# 模型加载：
model.load_state_dict(torch.load('最好的模型.pth').cpu().state_dict())
model.eval()
test_pred = []
test_true = []
# 直观的进行测试：一共95个学生的信息 76个训练 19个进行训练
with torch.no_grad():
    optimizer.zero_grad()
    with torch.no_grad():
        optimizer.zero_grad()
        for setp in range(len(test_X)):
            testx = torch.tensor(test_X[setp])
            # print(testx)
            testy = torch.tensor([test_y[setp]])
            testy = testy.type(torch.FloatTensor)
            y_pred = model(testx)
            print(y_pred,testy)
            test_pred.append(y_pred[0][0].item())
            test_true.append(testy[0].item())

        # 之前的训练数据拟合值进行输出：
        print("之前的训练数据拟合值进行输出：")
        for setp in range(len(train_X)):
            testx = torch.tensor(train_X[setp])
            # print(testx)
            testy = torch.tensor([train_y[setp]])
            testy = testy.type(torch.FloatTensor)
            y_pred = model(testx)
            print(y_pred,testy)

# 损失函数画图
fig = plt.figure(facecolor='white', figsize=(10, 7))
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(xmax=len(sum_test_epoch_loss), xmin=0)
plt.ylim(ymax=max(sum_test_epoch_loss), ymin=0)

x1 = [i for i in range(0, len(sum_train_epoch_loss ), 1)]  # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的x轴坐标
y1 = sum_test_epoch_loss  # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的y轴坐标
x2 = [i for i in range(0, len(sum_train_epoch_loss ), 1)]
y2 = sum_train_epoch_loss

colors1 = '#00CED4'  # 点的颜色
colors2 = '#DC143C'
area = np.pi * 4 ** 1  # 点面积
plt.scatter(x1, y1, s=area, c=colors1, label='val_loss')
plt.scatter(x2, y2, s=area, c=colors2, label='train_loss')
plt.legend()
plt.show()
colors1 = '#00CED4'  # 点的颜色
colors2 = '#DC143C'
area = np.pi * 4 ** 1  # 点面积
plt.scatter([2017,2018,2019,2020,2021], test_pred, s=area, c=colors1, alpha=0.4, label='test_pred')
plt.scatter([2017,2018,2019,2020,2021], test_true, s=area, c=colors2, alpha=0.4, label='test_true')
# plt.plot([0,9.5],[9.5,0],linewidth = '0.5',color='#000000')
plt.legend()
plt.show()

colors1 = '#00CED4'  # 点的颜色
colors2 = '#DC143C'
area = np.pi * 4 ** 1  # 点面积b
plt.plot([2017,2018,2019,2020,2021], test_pred,label='test_pred',color="r")
plt.plot([2017,2018,2019,2020,2021], test_true, label='test_true',color="b")
# plt.plot([0,9.5],[9.5,0],linewidth = '0.5',color='#000000')
plt.legend()
plt.show()

from sklearn import metrics
MSE = metrics.mean_squared_error(test_true, test_pred)
RMSE = metrics.mean_squared_error(test_true, test_pred)**0.5
MAE = metrics.mean_absolute_error(test_true, test_pred)
MAPE = metrics.mean_absolute_percentage_error(test_true, test_pred)
print("MSE,RMSE,MAE,MAPE")
print(MSE,RMSE,MAE,MAPE)



