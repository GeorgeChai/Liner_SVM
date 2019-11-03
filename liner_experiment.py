from data import *
from SVM import *
from hyperplane import *
if __name__ == "__main__":
    features,labels = create_linear_data()
    # 使用线性可分支持向量机学习算法进行训练
    w_,b_ = algorithm7_2(features,labels)
    plt.scatter(features[:50, 0], features[:50, 1], label='0')
    plt.scatter(features[50:, 0], features[50:, 1], label='1')
    plt.legend()
    hyperplane(w_, b_, 4, 7)
