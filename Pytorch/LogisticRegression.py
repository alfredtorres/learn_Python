# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 15:29:02 2018

@author: Zhangziyu
"""

# logistic regression 二分类问题
import torch
import matplotlib.pyplot as plt

# make fake data
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)     # class 0 x data Tensor, shape=(100, 2)
y0 = torch.zeros(100)              # class 0 y data Tensor, shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)    # class 1 x data Tensor, shape=(100, 2)
y1 = torch.ones(100)               # class 1 y data Tensor, shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.FloatTensor)
# show data
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), 
            s=100, lw=0, cmap='RdYlGn')
plt.show()
# model define
class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr = torch.nn.Linear(2, 1)
        self.sm = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.lr(x)
        x = self.sm(x)
        return x
    
logistic_model = LogisticRegression().cuda()
# loss and optim
criterion = torch.nn.BCELoss()   # Binary Cross Entropy between the target and putput
optimizer = torch.optim.SGD(logistic_model.parameters(), lr=1e-3)
for epoch in range(5000):
    x_train = torch.autograd.Variable(x).cuda()
    y_train = torch.autograd.Variable(y).cuda()
    # forward pass
    out = logistic_model(x_train)
    loss = criterion(out, y_train)
    print_loss = loss.data[0]
    mask = out.ge(0.5).float()  # 判断输出结果大于0.5等于1，小于0.5等于0
    correct = (mask==y_train).sum()
    acc = correct.data[0]/x_train.size(0)
    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 1000==0:
        print('*'*10)
        print('epoch {}'.format(epoch+1))
        print('loss is {:.4f}'.format(print_loss))
        print('acc is {:.4f}'.format(acc))
        
logistic_model.cpu()
w0, w1 = logistic_model.lr.weight[0]
w0 = w0.data[0]
w1 = w1.data[0]
b= logistic_model.lr.bias.data[0]
plot_x = torch.arange(-5, 5, 0.1)
plot_y = (-w0 * plot_x - b) / w1
plt.plot(plot_x.numpy(), plot_y.numpy())
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), 
            s=100, lw=0, cmap='RdYlGn')
plt.show()











