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
参考链接：[Python中 sys.argv[]的用法简明解释](https://www.cnblogs.com/aland-1415/p/6613449.html)
# 2 进阶用法
## 2.1 pyinstaller
由于项目需要，需要把一个python程序打包成.exe文件，供给vs程序调用。 
原本的python程序是在opencv和sklearn两个库基础上，输出一张人脸图片，判断是否戴了眼镜。  
程序中用到了caffe的vgg网络和sklearn的svm 
```
import cv2 as cv
import numpy as np
from sklearn.externals import joblib
``` 
首先，安装installer，然后用命令`pyinstaller -F model_test_opencv.py`生成一个dist文件夹下的model_test_opencv.exe文件。    
但是运行exe文件会报错，`no module named xxxx`  
查了一些资料后发现是需要添加`--hidden-import`，添加hidden-import的方法有两种   
1 在命令行中加 `pyintsaller -F model_test_opencv.py --hidden-import xxx`  
2 在生成的.spec文件中修改
```
hiddenimports=['scipy.lib.messagestream',
               'sklearn.svm.classes',
               'sklearn.neighbors.typedefs']
``` 
然后重新生成exe,pyinstaller -F -model_test_opencv.spec  
这样生成的exe文件就不会报错了。
