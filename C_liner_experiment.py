from data import *
from SVM import *
from hyperplane import *
from sklearn.model_selection import KFold,LeaveOneOut,LeavePOut,ShuffleSplit # 交叉验证所需的子集划分方法,参考 https://www.jianshu.com/p/284581d9b189

'''  忘记当时为啥要定义了......打扰了
def sign(i):
    return 1 if(i>0) else -1
'''

if __name__ == "__main__":
    '''
    # 使用线性支持向量机学习算法进行训练(近似线性可分数据:将iris中前100个线性可分数据做修改，45-49个由负类改为正类，95-99个由正类改为负类）
    features, labels = create_non_linear_data()
    plt.scatter(np.hstack((features[:45, 0],features[95:100, 0])),np.hstack((features[:45, 1],features[95:100, 1])), label='0')
    plt.scatter(np.hstack((features[50:95, 0],features[45:50, 0])), np.hstack((features[50:95, 1],features[45:50, 1])),label='1')
    plt.legend()
    #plt.show()
    w_, b_ = algorithm7_3(features, labels)
    hyperplane(w_, b_, 4, 7)
    print(w_,b_)
    '''
    precision = []
    C_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5]
    for C in C_list:   #
        i = 0
        data = create_non_linear_return_data()
        kfold = KFold(n_splits=5, shuffle=True, random_state=1)
        sum = 0.0
        for train, test in kfold.split(data):
            w_, b_ = algorithm7_3(data[train][:, :2], data[train][:, -1], C)
            sum += predict_precision(data[test][:, :2],data[test][:, -1], w_, b_)
            #hyperplane(w_, b_, 4, 7)
        sum /=5
        precision.append(sum)
    # 下面把最优分类器可视化
    max_index = precision.index(max(precision))   # C_list[max_index]即为最好的C
    data = create_non_linear_return_data()
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    sum = 0.0
    for train, test in kfold.split(data):
        w_, b_ = algorithm7_3(data[train][:, :2], data[train][:, -1], C_list[max_index])
        sum += predict_precision(data[test][:, :2],data[test][:, -1], w_, b_)
        for i in range(len(data[train])):
            if data[train][i][2] == 1:
                plt.scatter(data[train][i][0], data[train][i][1], marker='o', color='red')
            else:
                plt.scatter(data[train][i][0], data[train][i][1], marker='o', color='blue')
        for i in range(len(data[test])):
            if data[test][i][2] == 1:
                plt.scatter(data[test][i][0], data[test][i][1], marker='*', color='red')
            else:
                plt.scatter(data[test][i][0], data[test][i][1], marker='*', color='blue')
        hyperplane(w_, b_, 4, 7)
        plt.show()




