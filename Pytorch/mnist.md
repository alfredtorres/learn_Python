# Pytorch实现mnist手写数字识别
## 1 网络结构
设计了3个网络进行对比
* 3层全连接网络
* 3层全连接+RELU
* 3层全连接+RELU+BN
```
# 3层全连接网络
class simpleNet(torch.nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(simpleNet, self).__init__()
        self.layer1 = torch.nn.Linear(in_dim, n_hidden_1)
        self.layer2 = torch.nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = torch.nn.Linear(n_hidden_2, out_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
 # 添加激活函数，改进全连接网络   
class Activation_Net(torch.nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Activation_Net, self).__init__()
        self.layer1 = torch.nn.Sequential(
                torch.nn.Linear(in_dim, n_hidden_1), torch.nn.ReLU(True))
        self.layer2 = torch.nn.Sequential(
                torch.nn.Linear(n_hidden_1, n_hidden_2), torch.nn.ReLU(True))
        self.layer3 = torch.nn.Linear(n_hidden_2, out_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    
# 添加BN，加快收敛速度
# BatchNorm 一般放在全连接层后面，激活函数前面
class Batch_Net(torch.nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = torch.nn.Sequential(
                torch.nn.Linear(in_dim, n_hidden_1), 
                torch.nn.BatchNorm1d(n_hidden_1), 
                torch.nn.ReLU(True))
        self.layer2 = torch.nn.Sequential(
                torch.nn.Linear(n_hidden_1, n_hidden_2), 
                torch.nn.BatchNorm1d(n_hidden_2), 
                torch.nn.ReLU(True))
        self.layer3 = torch.nn.Linear(n_hidden_2, out_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
```
## 2 训练参数
```
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import net

# 超参数 Hyperparameters
batch_size = 64
learning_rate = 1e-2
num_epoches = 20
# data preprocess
data_tf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])]
        )
# 下载数据集 prepare data
train_dataset = datasets.MNIST(
        root='D:/Software/PyTorch/data', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(
        root='D:/Software/PyTorch/data', train=False, transform=data_tf)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
# 导入网络
# define loss and optim
model = net.simpleNet(28 * 28, 300, 100, 10)
model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# 开始训练
for epoch in range(num_epoches):
    run_loss = 0.0
    for i, train_data in enumerate(train_loader, 0):
        # forward pass
        img, label = train_data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        out = model(img)
        loss = criterion(out, label)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print loss
        run_loss +=loss.item() * label.size(0)
    # print every epoch
    print('epoch: %d, loss: %.6f' %
          (epoch + 1, run_loss / len(train_dataset)))
# 训练完成
print('train done.')
# 预测
model.eval()
eval_loss = 0
eval_acc = 0
with torch.no_grad(): 
    for data in test_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred==label).sum().item()
        eval_acc += num_correct
print('Test Loss:{:.6f}, Acc{:.6f}'.format(
        eval_loss / (len(test_dataset)),
        eval_acc / (len(test_dataset))))
```
## 3 结果
### 3.1 简单3层全连接网络
lr=1e-2
```
epoch: 1, loss: 0.762809
epoch: 2, loss: 0.363149
epoch: 3, loss: 0.328053
epoch: 4, loss: 0.312956
epoch: 5, loss: 0.304503
epoch: 6, loss: 0.297647
epoch: 7, loss: 0.293210
epoch: 8, loss: 0.289277
epoch: 9, loss: 0.286048
epoch: 10, loss: 0.283788
epoch: 11, loss: 0.281018
epoch: 12, loss: 0.278887
epoch: 13, loss: 0.277247
epoch: 14, loss: 0.275849
epoch: 15, loss: 0.274103
epoch: 16, loss: 0.272892
epoch: 17, loss: 0.271125
epoch: 18, loss: 0.269829
epoch: 19, loss: 0.268939
epoch: 20, loss: 0.268331
train done.
Test Loss:0.287246, Acc0.918800
```
lr=1e-3时，`Test Loss:0.315639, Acc0.909600`

### 3.2 3层全连接+RELU
```
epoch: 1, loss: 1.013060
epoch: 2, loss: 0.370626
epoch: 3, loss: 0.313586
epoch: 4, loss: 0.281176
epoch: 5, loss: 0.254788
epoch: 6, loss: 0.232635
epoch: 7, loss: 0.212524
epoch: 8, loss: 0.194143
epoch: 9, loss: 0.178646
epoch: 10, loss: 0.164829
epoch: 11, loss: 0.153017
epoch: 12, loss: 0.141990
epoch: 13, loss: 0.132835
epoch: 14, loss: 0.124171
epoch: 15, loss: 0.116364
epoch: 16, loss: 0.109466
epoch: 17, loss: 0.103453
epoch: 18, loss: 0.097604
epoch: 19, loss: 0.092519
epoch: 20, loss: 0.087344
train done.
Test Loss:0.101480, Acc0.968900
```
### 3.3 3层全连接+RELU+BN
```
epoch: 1, loss: 0.733026
epoch: 2, loss: 0.231444
epoch: 3, loss: 0.151939
epoch: 4, loss: 0.115250
epoch: 5, loss: 0.091842
epoch: 6, loss: 0.075793
epoch: 7, loss: 0.063315
epoch: 8, loss: 0.055989
epoch: 9, loss: 0.049139
epoch: 10, loss: 0.042940
epoch: 11, loss: 0.038607
epoch: 12, loss: 0.034571
epoch: 13, loss: 0.030546
epoch: 14, loss: 0.027187
epoch: 15, loss: 0.024033
epoch: 16, loss: 0.022607
epoch: 17, loss: 0.019951
epoch: 18, loss: 0.018433
epoch: 19, loss: 0.017939
epoch: 20, loss: 0.016841
train done.
Test Loss:0.063228, Acc0.980200
```
