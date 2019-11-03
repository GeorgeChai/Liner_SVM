from cvxopt import matrix,solvers  # 利用cvxopt解决凸优化问题
import numpy as np
# 二次规划教程 https://blog.csdn.net/goodxin_ie/article/details/84591530


def algorithm7_2(features, labels):
    N = len(labels)
    X = features
    y = labels
    # 矩阵P代表yiyj(xixj)
    P = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            P[i][j] = y[i]*y[j]*np.dot(X[i],X[j])
    P = matrix(P)
    # q为一维n长 —1 矩阵
    q = np.ones((N),dtype=float)*-1
    q = matrix(q)
    # G不是零矩阵，是对角线元素全为-1的n*n矩阵
    G = matrix(np.diag(np.ones(N)*-1))
    # h为一维n长零矩阵
    h = matrix(np.zeros(N))
    # A为labels的转置
    A = matrix(y)
    A = A.T
    # b就是0，但也要转化为矩阵形式
    b = matrix([0.0])
    result = solvers.qp(P,q,G,h,A,b)
    alpha = np.squeeze(np.array(result['x']))   # 求出的实际是alpha,转化为长度为100的向量
    w_ = np.dot(alpha*y,X)
    # 下求b_
    XX = np.zeros((N), dtype=float)
    for j in range(len(alpha)):
        if alpha[j] > 1e-5:
            for i in range(N):
                XX[i] = np.dot(X[i], X[j])
            b_ = y[j] - np.dot(alpha*y,XX)
            break
    return w_,b_


def algorithm7_3(features, labels, c):
    N = len(labels)
    X = features
    y = labels
    # 矩阵P代表yiyj(xixj)
    P = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            P[i][j] = y[i]*y[j]*np.dot(X[i],X[j])
    P = matrix(P)
    # q为一维n长 —1 矩阵
    q = np.ones((N),dtype=float)*-1
    q = matrix(q)
    # G要做修改，在对角线元素全为-1的n*n矩阵基础上，下面加n行1
    G = matrix(np.vstack((np.diag(np.ones(N)*-1),np.identity(N))))
    # h为一维n长零矩阵，下面再加N行C
    h = matrix(np.hstack((np.zeros(N), np.ones(N)*c)))
    # A为labels的转置
    A = matrix(y)
    A = A.T
    # b就是0，但也要转化为矩阵形式
    b = matrix([0.0])
    result = solvers.qp(P,q,G,h,A,b)
    alpha = np.squeeze(np.array(result['x']))   # 求出的实际是alpha,转化为长度为100的向量
    w_ = np.dot(alpha*y, X)
    # 下求b_
    XX = np.zeros((N), dtype=float)
    for j in range(len(alpha)):
        if alpha[j] > 1e-5:
            for i in range(N):
                XX[i] = np.dot(X[i], X[j])
            b_ = y[j] - np.dot(alpha*y,XX)
            break
    pass
    return w_,b_


def sign(w, b, X):
    return np.dot(w, X) + b


def predict_precision(test_X, test_y, w, b):
    right_num = 0
    for i in range(len(test_X)):
        if sign(w, b, test_X[i]) * test_y[i] >0:
            right_num +=1
    return right_num/len(test_X)
