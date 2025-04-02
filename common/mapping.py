import numpy as np
import pennylane as qml
from pdfo import uobyqa


dev = qml.device("default.qubit", wires=1)
@qml.qnode(dev, interface="autograd")
def u_map(onefts, theta):
    for i in range(len(theta)):
        qml.RX(theta[i] * onefts[i], wires=[0])
    return qml.state()

def get_mapping(node_num, features):
    fea_num = len(features[0])
    # set_seed(123)
    theta = np.random.rand(fea_num)
    res = uobyqa(train_mapping, theta, (node_num, features))
    theta = res['x']
    return theta

def train_mapping(theta,node_num,features):
    """"
    训练范围：单个子图
    在uobyqa算法中需要的输入输出形式：
    输入：变量，theta
    输出：定义的损失，loss
    但是这个函数应该还需具备以下功能：
    保存最后的theta或fai_fea，即Ucov的输入
    theta可以直接通过upobyqa的输出res['x']得到，但是最好直接输出fai_fea，不用重新计算。
    """
    # 计算欧式距离相关矩阵
    D = np.zeros((node_num, node_num))
    for i in range(node_num):
        for j in range(i + 1, node_num):
            D[i][j] = np.dot(features[i], features[j]) / (
                    np.linalg.norm(features[i]) * np.linalg.norm(features[j]))
            D[j][i] = D[i][j]
    D /= np.max(D)
    fai_fea = []  # 保存每个节点的φ
    for i in range(node_num):
        fai_fea.append(u_map(features[i], theta))
    D_ = np.zeros((node_num, node_num))
    for i in range(node_num):
        for j in range(i + 1, node_num):
            D_[i][j] = np.dot(fai_fea[i].conjugate(), fai_fea[j])  # 共轭转置
            D_[j][i] = D_[i][j]
    D_ /= np.max(D_)
    # plt.figure(1,figsize=(20,8)) #映射前后对比图
    # plt.subplot(121)
    # seaborn.heatmap(D, center=0, annot=True, cmap='YlGnBu')
    # plt.subplot(122)
    # seaborn.heatmap(D_, center=0, annot=True, cmap='YlGnBu')
    # plt.savefig("edcorr_uobyqa.png")
    minus = D - D_
    loss = np.sum(np.abs(minus))
    return loss