# 自学Pytorch
## 《深度学习之Pytorch入门》
### 1 多层全连接神经网络
1. [线性回归](https://github.com/alfredtorres/learn_Python/blob/master/Pytorch/LinearRregression.py)
<div align="center">
<img src="https://github.com/alfredtorres/learn_Python/blob/master/Pytorch/images/linear%20regression.png">
</div>

2. [多项式线性回归](https://github.com/alfredtorres/learn_Python/blob/master/Pytorch/PolyRregreesion.py)
<div align="center">
<img src="https://github.com/alfredtorres/learn_Python/blob/master/Pytorch/images/poly%20regression.png">
</div>

3. [分类问题-逻辑回归](https://github.com/alfredtorres/learn_Python/blob/master/Pytorch/LogisticRegression.py)
<div align="center">
<img src="https://github.com/alfredtorres/learn_Python/blob/master/Pytorch/images/logistic%20regression.png">
</div>

4. [全连接网络实现mnist数字识别](https://github.com/alfredtorres/learn_Python/blob/master/Pytorch/mnist.md)  
   * 3层全连接网络           `Test Loss:0.287246, Acc0.918800`
   * 3层全连接网络+RELU      `Test Loss:0.101480, Acc0.968900`
   * 3层全连接网络+RENLU+BN  `Test Loss:0.063228, Acc0.980200`
--------------
### 2 卷积神经网络
1. 4层卷积神经网络
mnist训练结果，准确率99.37%
```
epoch: 1, loss: 0.469514
epoch: 2, loss: 0.077270
epoch: 3, loss: 0.052279
epoch: 4, loss: 0.040815
epoch: 5, loss: 0.033697
epoch: 6, loss: 0.028989
epoch: 7, loss: 0.024224
epoch: 8, loss: 0.021510
epoch: 9, loss: 0.018648
epoch: 10, loss: 0.016845
epoch: 11, loss: 0.014172
epoch: 12, loss: 0.012640
epoch: 13, loss: 0.011401
epoch: 14, loss: 0.010049
epoch: 15, loss: 0.008889
epoch: 16, loss: 0.008120
epoch: 17, loss: 0.006904
epoch: 18, loss: 0.006234
epoch: 19, loss: 0.005504
epoch: 20, loss: 0.004869
train done.
Test Loss:0.018960, Acc0.993700
```
