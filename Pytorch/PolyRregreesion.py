# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 10:25:54 2018

@author: Zhangziyu
Pytorch实现多项式回归
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
torch.set_default_tensor_type('torch.DoubleTensor')
# 准备训练数据
def make_features(x):
    '''builds featrues i.e. a matrix with columns [x, x^2, x^3].'''
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, 4)], 1) 
    '''torch.cat()实现Tensor的拼接'''
W_target = torch.DoubleTensor([0.5, 3, 2.4]).unsqueeze(1)
b_target = torch.DoubleTensor([0.9])

def f(x):
    '''approximated function'''
    return x.mm(W_target) + b_target[0]

def get_batch(batch_size=32):
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    return torch.autograd.Variable(x).cuda(), torch.autograd.Variable(y).cuda()

# 定义多项式回归模型
class poly_model(torch.nn.Module):
    def __init__(self):
        super(poly_model, self).__init__()
        self.poly = torch.nn.Linear(3, 1)
    
    def forward(self, x):
        out = self.poly(x)
        return out

model = poly_model().cuda()
# 选择loss和优化器
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
epoch = 0#不指定迭代次数，用loss来退出循环
while True:
    # Get data
    batch_x, batch_y = get_batch()
    # forward pass
    output = model(batch_x)
    loss = criterion(output, batch_y)
    print_loss = loss.data[0]
    # Reset grad
    optimizer.zero_grad()
    #backword pass
    loss.backward()
    # updata para
    optimizer.step()
    epoch += 1
    if print_loss < 1e-3:
        break

print(epoch)
# 训练完预测结果
model.eval()
model.cpu()
x_pred = np.linspace(-1, 1, 20)

x_feat = make_features(torch.from_numpy(x_pred))
y_pred = f(x_feat)
predict = model(x_feat)
x_pred = np.reshape(x_pred, (20,1))
plt.plot(x_pred, y_pred.numpy(), 'ro', label='Origin data')
plt.plot(x_pred, predict.data.numpy(), label='Fitting line')    
plt.legend()
plt.show()  
    

        
        
    
    
    
    