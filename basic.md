# 1 基础语法
## 1.1 sys.argv[]从系统外部输入  
一般python程序的调用方法是： 
```python test.py para1 para2```  
其中```para1 para2```就是从系统外部输入的参数
sys.argv[]表示是输入参数的list  结合具体例子说明
```python test.py para1 para2```
那么  
```sys.argv[0]```即表示 ```test.py```  
```sys.argv[1]```即表示 ```para1```  
```sys.argv[2]```即表示 ```para2```  
假如想获取所有的输入，用```sys.argv[1:]```即可
