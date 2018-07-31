# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 16:33:16 2018

@author: Zhangziyu
"""

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

torch.set_default_tensor_type('torch.DoubleTensor')
# 生成数据
x_train = np.linspace(-1, 1, 100,dtype=np.double)
'''
x_train = np.array([[3.3], [4.4], [5.5], [6.71],[6.93],[4.168],
                    [9.779],[6.182],[7.59],[2.167],[7.042],
                    [10.791],[5.313],[7.997],[3.1]],dtype=np.float64)
'''
y_train = x_train * 2 + 1 + np.random.randn(100) * 0.3
x_train = np.reshape(x_train, (100,1))
y_train = np.reshape(y_train, (100,1))
'''
y_train = np.array([[1.7], [2.76], [2.09], [3.19],[1.697],[1.573],
                    [3.366],[2.596],[2.53],[1.221],[2.87],
                    [3.645],[1.65],[2.904],[1.3]],dtype=np.float64)
'''
'''显示
plt.plot(x_train, y_train, 'ro', label='Origin data')
plt.legend()
plt.show()
'''
# np数据转Tensor
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
# 建立一阶线性torch模型
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)# input and outpue is 1 dimension
    
    def forward(self, x):
        out = self.linear(x)
        return out
    
# 如果能使用gpu,则将模型放入gpu中。
if torch.cuda.is_available():
    model = LinearRegression().cuda()
else:
    model = LinearRegression()
##定义loss和Opt
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
# 开始训练模型
num_epochs = 10000
for epoch in range(num_epochs):
    if torch.cuda.is_available():
        inputs = torch.autograd.variable(x_train).cuda()
        target = torch.autograd.variable(y_train).cuda()
    else:
        inputs = torch.autograd.variable(x_train)
        target = torch.autograd.variable(y_train)
    # 前向计算
    out = model(inputs)
    loss = criterion(out, target)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print('Epoch[{}/{}], loss:{:.6f})'.format(epoch+1, num_epochs, loss.data[0]))
    
# 训练完进行预测
model.eval()
model.cpu()
predict = model(torch.autograd.variable(x_train))
predict = predict.data.numpy()
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original Data')
plt.plot(x_train.numpy(), predict, label='Fitting Data')
plt.show()
        
        

        
        
