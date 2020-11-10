from sklearn import svm
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib

'''define converts(字典)因为分类类别标签必须为数字，所以需将数据中的
最后一列类别(字符串)通过转换变为数字'''
def Iris_label(s):
    it={b'Iris-setosa':0,b'Iris-versicolor':1,b'Iris-virginica':2}  #将3种类别分别映射成0,1,2
    return it[s]
#1.读取数据集
path='D:/learnpython/data/Iris.data'
data=np.loadtxt(path,dtype=float,delimiter=',',converters={4:Iris_label})
#converters={4:Iris_label}中“4”指的是第5列：将第5列的str转化为label（number）
