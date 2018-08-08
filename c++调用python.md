# C++调用Python脚本中的函数
### 最终目标：在c++中实现facenet的前向计算
## 1 基础教程
参考链接[C++调用Python浅析](https://www.cnblogs.com/findumars/p/7124031.html)  
### VS2017中的配置
在VC++目录中的包含目录添加
```
D:\Software\anoconda2\envs\tensorflow\libs
D:\Software\anoconda2\envs\tensorflow\include
```
在库目录添加
```
D:\Software\anoconda2\envs\tensorflow\libs
D:\Software\anoconda2\envs\tensorflow\include
```
把Python路径下的`D:\Software\anoconda2\envs\tensorflow\libs`中的`python35.lib`复制一份，重命名为`python35_d.lib`
***********************
遇到如下问题时，按照[无法解析的外部符号](https://www.jb51.net/article/108588.htm)解决
```
1>pythonIniti.obj : error LNK2019: 无法解析的外部符号 __imp___Py_NegativeRefcount
1>pythonIniti.obj : error LNK2001: 无法解析的外部符号 __imp___Py_RefTotal
```
解决方法： 
> 修改两个头文件  
1 注释掉object.h第56行  
//#define Py_TRACE_REFS  
2 pyconfig.h 375行【这个行号可能不对】  
//#define Py_DEBUG  
以上两个宏定义注释掉以后重新编译 问题解决  
***********************
## 2示例  
**cpp文件如下**
**注意：初始化前的python home设置，这一步卡了半天**
```
#include "stdafx.h"
#include "Python.h"
#include <iostream>
using namespace std;
int main()
{
	int nRet = -1;
	PyObject* pName = NULL;
	PyObject* pModule = NULL;
	PyObject* pDict = NULL;
	PyObject* pFunc = NULL;
	PyObject* pArgs = NULL;
	PyObject* pRet = NULL;
	do
	{
		// 初始化Python
		// 在使用Python系统前，必须使用Py_Initialize对其
		// 进行初始化。它会载入Python的内建模块并添加系统路
		// 径到模块搜索路径中。这个函数没有返回值，检查系统
		// 是否初始化成功需要使用Py_IsInitialized。
		wchar_t pypath[] = L"D:/Software/anoconda2/envs/tensorflow";
		Py_SetPythonHome(pypath);
		Py_Initialize();
		// 检查初始化是否成功
		if (!Py_IsInitialized())
		{
			break;
		}
		// 添加当前路径
		// 把输入的字符串作为Python代码直接运行，返回
		// 表示成功，-1表示有错。大多时候错误都是因为字符串
		// 中有语法错误。
		PyRun_SimpleString("import sys");
		string path = "D:/Files/SimuAndSoftware/python2cpp";  //python文件路径
		string chdir_cmd = string("sys.path.append(\"");
		chdir_cmd += path;
		chdir_cmd += "\")";
		const char* cstr_cmd = chdir_cmd.c_str();
		PyRun_SimpleString(cstr_cmd);
		// 载入名为PyPlugin的脚本	
		pModule = PyImport_ImportModule("Pyplugin");
		if (!pModule)
		{
			printf("can't findPyPlugin.py\n");
			break;
		}
		pDict = PyModule_GetDict(pModule);
		if (!pDict)
		{
			break;
		}
		// 找出函数名为AddMult的函数
		pFunc = PyDict_GetItemString(pDict, "AddMult");
		if (!pFunc || !PyCallable_Check(pFunc))
		{
			printf("can't findfunction [AddMult]\n");
			break;
		}
		pArgs = Py_BuildValue("ii", 12, 14);
		PyObject* pRet = PyEval_CallObject(pFunc, pArgs);
		int a = 0;
		int b = 0;
		if (pRet && PyArg_ParseTuple(pRet, "ii", &a, &b))
		{
			printf("Function[AddMult] call successful a + b = %d, a * b = %d\n", a, b);
			nRet = 0;
		}
		if (pArgs)
			Py_DECREF(pArgs);
		if (pFunc)
			Py_DECREF(pFunc);
		// 找出函数名为HelloWorld的函数
		pFunc = PyDict_GetItemString(pDict, "HelloWorld");
		if (!pFunc || !PyCallable_Check(pFunc))
		{
			printf("can't findfunction [HelloWorld]\n");
			break;
		}
		pArgs = Py_BuildValue("(s)", "magictong");
		PyEval_CallObject(pFunc, pArgs);
		// 找出函数名为tfun1的函数
		pFunc = PyDict_GetItemString(pDict, "tfun1");
		if (!pFunc || !PyCallable_Check(pFunc))
		{
			printf("can't findfunction [tfun1]\n");
			break;
		}
		pArgs = Py_BuildValue("(s)", "start tf1");
		PyEval_CallObject(pFunc, pArgs);
		// 找出函数名为tfun2的函数
			pFunc = PyDict_GetItemString(pDict, "tfun1");
		if (!pFunc || !PyCallable_Check(pFunc))
		{
			printf("can't findfunction [tfun2]\n");
			break;
		}
		pArgs = Py_BuildValue("(s)", "start tf1");
		PyEval_CallObject(pFunc, pArgs);
	} while (0);
	if (pRet)
		Py_DECREF(pRet);
	if (pArgs)
		Py_DECREF(pArgs);
	if (pFunc)
		Py_DECREF(pFunc);
	if (pDict)
		Py_DECREF(pDict);
	if (pModule)
		Py_DECREF(pModule);
	if (pName)
		Py_DECREF(pName);
	Py_Finalize();
	return 0;
}
```
Pyplugin.py文件如下
```
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 14:47:22 2018

@author: Zhangziyu
"""

import string
import tensorflow as tf
import time
"""
class CMyClass:
    def HelloWorld(self):
        return 2

class SecondClass:
    def invoke(self,obj):
        obj.HelloWorld()
"""
def HelloWorld(str):
    print(str)
 
def Add(a, b, c):
    return a + b + c

def AddMult(a, b):
    print('a=', a)
    print('b=', b)
    return a + b, a * b

def StringToUpper(strSrc):
    return string.upper(strSrc)

def tfun1(str):
    print(str)
    time_start = time.time()
    # Build a graph.
    a = tf.constant([1.0, 2.0])
    b = tf.constant([3.0, 4.0])
    c = a * b
    # Launch the graph in a session.
    sess = tf.Session()
    time_end = time.time()
    # Evaluate the tensor 'c'.
    print(sess.run(c))
    print('tf1 time: ', time_end - time_start)
    
def tfun2(str):
    print(str)
    time_start = time.time()
    # Build a graph.
    a = tf.constant([1.0, 2.0])
    b = tf.constant([3.0, 4.0])
    c = a * b
    # Launch the graph in a session.
    sess = tf.Session()
    time_end = time.time()
    print(sess.run(c))
    print('tf2 time: ', time_end - time_start)
```

## 3 C++实现facenet前向计算
