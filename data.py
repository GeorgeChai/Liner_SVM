import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate # 交叉验证所需的函数


# 生成UCI数据集中iris的100个线性可分数据
def create_linear_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]
    data = np.array(df.iloc[:100, [0, 1, -1]]) # 只选columns中的第0，1和最后一项
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    # print(data)
    return data[:, :2], data[:, -1]


def create_non_linear_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]
    data = np.array(df.iloc[:100, [0, 1, -1]]) # 只选columns中的第0，1和最后一项
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    for i in range(45,50):
        data[i][2] *= -1
    for i in range(95,100):
        data[i][2] *= -1
    #print(data)
    return data[:, :2], data[:, -1]


def create_non_linear_return_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]
    data = np.array(df.iloc[:100, [0, 1, -1]]) # 只选columns中的第0，1和最后一项
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    for i in range(45,50):
        data[i][2] *= -1
    for i in range(95,100):
        data[i][2] *= -1
    return data
