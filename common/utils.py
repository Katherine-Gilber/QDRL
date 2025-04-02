import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
import torch
import random
import pennylane as qml
import matplotlib.pyplot as plt
from .mapping import get_mapping
import time
# from mapping import train_mapping

def load_data(graph_num,path='./data/AIDS/', dataset='AIDS'):
    """"
    AIDS一共有200个graph
    A.txt：连接信息，每行代表一条边
    graph_indicator：标明每个节点属于哪个graph
    node_attributes：代表每个节点特征
    选用节点数最多的graph为例子
    """
    nodes_all=np.loadtxt(path+dataset+'_A.txt',dtype=int,delimiter=',')
    graph_indicator=np.loadtxt(path+dataset+'_graph_indicator.txt',dtype=int,delimiter=',')
    node_atrbs=np.loadtxt(path+dataset+'_node_attributes.txt',dtype=float,delimiter=',')
    node_atrbs=normalize(node_atrbs,axis=0,norm='max') #特征归一化
    graph_class=[[] for i in range(graph_num+1)] #将节点和graph一一映射，行标代表graph编号，每行元素代表该graph中的节点
    graph_cont=np.zeros(graph_num+1) #存储每个graph中的节点数目，以便找到节点数最多的graph
    for i in range(len(graph_class)):
        for j in range(len(graph_indicator)):
            if(graph_indicator[j]==i+1):
                graph_class[i+1].append(j+1)
                graph_cont[i+1]+=1
    max_indicator=graph_cont.argmax()
    G=nx.Graph()
    G.add_nodes_from(graph_class[max_indicator])
    for e in nodes_all:
        if e[0] and e[1] in graph_class[max_indicator]:
            G.add_edge(*e)
    for i in graph_class[max_indicator]:
        G.nodes[i]['feature']=node_atrbs[i-1] #这个好像没啥必要？检查下标对不对，for i in Gnode，返回编号，G.nodes[]中间是遍历not编号数，∴编号要从0开始
    return G

def load_synthetic(graph_num):
    path='../data/synthetic/synthetic.txt'
    nodes_all = np.loadtxt(path, dtype=str, delimiter=',')
    G=nx.Graph()
    G.add_edges_from(nodes_all)
    nx.write_gml(G,path='../data/synthetic/synthetic.gml')


def decomposition_graph(G,layer_nbr):
    """
    可以扩展到k阶邻居，自带的subgraph只有1阶邻居
    :param G: 一整个网络
    :param layer_nbr:k阶邻居
    :return:返回保存所有子图的list，subG和center的下标一一对应
    """
    subG=[]
    center=[]
    idx=[]
    for node in G.nodes():
        subG.append(nx.ego_graph(G,node,radius=layer_nbr))
        center.append(node)
        idx.append(list(subG[-1].nodes()).index(node))
    return subG,center,idx

#用l-hop ego network抽取节点数目为subg_num的子图
def l_hop_ego_subgraph(G, ego_node,subg_num):
    node_sample = np.zeros(G.number_of_nodes())
    node_sample[ego_node] = 1
    l_hop_subgraph_nodes = {ego_node}  # 存储l-hop子图的节点

    # 使用广度优先搜索来找到l-hop子图的节点
    queue = [ego_node]
    flag=False
    while queue:
        current_node = queue.pop(0)
        if flag:
            break
        for neighbor in G.neighbors(current_node):
            if len(l_hop_subgraph_nodes)==subg_num:
                flag=True
                break
            if node_sample[neighbor] == 0:
                queue.append(neighbor)
                node_sample[neighbor] = 1
                l_hop_subgraph_nodes.add(neighbor)

    l_hop_subgraph = G.subgraph(l_hop_subgraph_nodes)
    return g_copy(l_hop_subgraph)

def index_mapping(G):
    index_map={}
    for idx,node in enumerate(sorted(G.nodes(),key=lambda x:G.nodes[x]['feature'][0],reverse=True)):
        index_map[node] = idx
    return  index_map

def batchgraph(G,subg_num): #传完整图
    subg_list=[]
    # set_seed(123) #同一个图，固定了每次产生的子图都一样，可注释
    center_nodes=list(G.nodes)
    # center_nodes = np.random.choice(range(G.number_of_nodes()), size=samples_num, replace=False) #默认G node序号从0开始按顺序递增
    for node in center_nodes:
        g=l_hop_ego_subgraph(G, node, subg_num)
        subg_list.append(g)
    return subg_list


#初始化节点属性，暂用4个全1 可否改成node2vec更有说服力
def gen_features(g,gen_features_type='featureless'): # 默认[度,特征,介数,接近,聚类]
    if gen_features_type=='featureless':
        feat=[[] for _ in range(g.number_of_nodes())]
        degr = list(nx.degree_centrality(g).values())
        eigen = list(nx.eigenvector_centrality(g,max_iter=1000).values())
        betw = list(nx.betweenness_centrality(g).values())
        close = list(nx.closeness_centrality(g).values())
        clus = list(nx.clustering(g).values())

        for node in g.nodes:
            feat[node].append(degr[node])
            feat[node].append(eigen[node])
            feat[node].append(betw[node])
            feat[node].append(close[node])
            feat[node].append(clus[node])
        feat=np.array(feat)
        feat = normalize(feat, axis=0, norm='max')  # 特征归一化
        theta = get_mapping(g.number_of_nodes(), feat)
        for node in g.nodes: #生成图label都是从0开始
            g.nodes[node]['feature']=feat[node]*np.array(theta)
            # g.nodes[node]['feature'] = feat[node]
            # g.nodes[node]['layer_inf'] = []
            # g.nodes[node]['measure'] = 0


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 让显卡产生的随机数一致
    torch.cuda.manual_seed_all(seed)  # 多卡模式下，让所有显卡生成的随机数一致？这个待验证-
    np.random.seed(seed)
    random.seed(seed)

def g_copy(G):
    H = nx.Graph()
    H.add_nodes_from(G)
    H.add_edges_from(G.edges)
    for i in G.nodes:
        H.nodes[i]['feature'] = G.nodes[i]['feature']
        # H.nodes[i]['measure'] = 0
    return H

def estimate_shadow_obervable(shadow, observable, k=2):
    """
    Calculate the estimator E[O] = median(Tr{rho_{(k)} O}) where rho_(k)) is set of k
    snapshots in the shadow. Use median of means to ameliorate the effects of outliers.

    Args:
        shadow (tuple): A shadow tuple obtained from `calculate_classical_shadow`.
        observable (qml.Observable): Single PennyLane observable consisting of single Pauli
            operators e.g. qml.PauliX(0) @ qml.PauliY(1).
        k (int): number of splits in the median of means estimator.

    Returns:
        Scalar corresponding to the estimate of the observable.
    """
    shadow_size, num_qubits = shadow[0].shape

    # convert Pennylane observables to indices
    map_name_to_int = {"PauliX": 0, "PauliY": 1, "PauliZ": 2}
    if isinstance(observable, (qml.PauliX, qml.PauliY, qml.PauliZ)):
        target_obs, target_locs = np.array(
            [map_name_to_int[observable.name]]
        ), np.array([observable.wires[0]])
    else:
        target_obs, target_locs = np.array(
            [map_name_to_int[o.name] for o in observable.obs]
        ), np.array([o.wires[0] for o in observable.obs])

    # classical values
    b_lists, obs_lists = shadow
    means = []

    # loop over the splits of the shadow:
    for i in range(0, shadow_size, shadow_size // k):

        # assign the splits temporarily
        b_lists_k, obs_lists_k = (
            b_lists[i: i + shadow_size // k],
            obs_lists[i: i + shadow_size // k],
        )

        # find the exact matches for the observable of interest at the specified locations
        indices = np.all(obs_lists_k[:, target_locs] == target_obs, axis=1)

        # catch the edge case where there is no match in the chunk
        if sum(indices) > 0:
            # take the product and sum
            product = np.prod(b_lists_k[indices][:, target_locs], axis=1)
            means.append(np.sum(product) / sum(indices))
        else:
            means.append(0)

    return np.median(means)

def shadow_bound(error, observables, failure_rate=0.01):
    """
    Calculate the shadow bound for the Pauli measurement scheme.
    Args:
        error (float): The error on the estimator.
        observables (list) : List of matrices corresponding to the observables we intend to
            measure.
        failure_rate (float): Rate of failure for the bound to hold.

    Returns:
        An integer that gives the number of samples required to satisfy the shadow bound and
        the chunk size required attaining the specified failure rate.
    """
    M = len(observables)
    K = 2 * np.log(2 * M / failure_rate)
    shadow_norm = (
        lambda op: np.linalg.norm(
            op - np.trace(op) / 2 ** int(np.log2(op.shape[0])), ord=np.inf
        )
        ** 2
    )
    N = 34 * max(shadow_norm(o) for o in observables) / error ** 2
    return int(np.ceil(N * K)), int(K)

def calculate_classical_shadow(circuit_template, params, shadow_size, num_qubits):
    """
    Given a circuit, creates a collection of snapshots consisting of a bit string
    and the index of a unitary operation.

    Args:
        circuit_template (function): A Pennylane QNode.
        params (array): Circuit parameters.
        shadow_size (int): The number of snapshots in the shadow.
        num_qubits (int): The number of qubits in the circuit.

    Returns:
        Tuple of two numpy arrays. The first array contains measurement outcomes (-1, 1)
        while the second array contains the index for the sampled Pauli's (0,1,2=X,Y,Z).
        Each row of the arrays corresponds to a distinct snapshot or sample while each
        column corresponds to a different qubit.
    """
    # applying the single-qubit Clifford circuit is equivalent to measuring a Pauli
    unitary_ensemble = [qml.PauliX, qml.PauliY, qml.PauliZ]

    # sample random Pauli measurements uniformly, where 0,1,2 = X,Y,Z
    unitary_ids = np.random.randint(0, 3, size=(shadow_size, num_qubits))
    outcomes = np.zeros((shadow_size, num_qubits))

    for ns in range(shadow_size):
        # for each snapshot, add a random Pauli observable at each location
        obs = [unitary_ensemble[int(unitary_ids[ns, i])](i) for i in range(num_qubits)]
        outcomes[ns, :] = circuit_template(params, observable=obs)

    # combine the computational basis outcomes and the sampled unitaries
    return (outcomes, unitary_ids)

if __name__=='__main__':
    # load_synthetic(22)
    num_qubits = 9
    dev = qml.device("lightning.qubit", wires=num_qubits, shots=1)

    def circuit_base(params, **kwargs):
        observables = kwargs.pop("observable")
        for w in range(num_qubits):
            qml.Hadamard(wires=w)
            qml.RY(params[w], wires=w)
        for w in dev.wires[:-1]:
            qml.CNOT(wires=[w, w + 1])
        for w in dev.wires:
            qml.RZ(params[w + num_qubits], wires=w)
        return [qml.expval(o) for o in observables]

    circuit = qml.QNode(circuit_base, dev)

    params = np.random.randn(2 * num_qubits)
    list_of_observables = (
            [qml.PauliX(i) @ qml.PauliX(i + 1) for i in range(num_qubits - 1)]
            + [qml.PauliY(i) @ qml.PauliY(i + 1) for i in range(num_qubits - 1)]
            + [qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(num_qubits - 1)]
    )
    shadow_size_bound, k = shadow_bound(
        error=2e-1, observables=[qml.matrix(o) for o in list_of_observables]
    )
    print(shadow_size_bound)

    # create a grid of errors
    epsilon_grid = [1 - 0.1 * x for x in range(9)]
    shadow_sizes = []
    estimates = []

    for error in epsilon_grid:
        # get the number of samples needed so that the absolute error < epsilon.
        shadow_size_bound, k = shadow_bound(
            error=error, observables=[qml.matrix(o) for o in list_of_observables]
        )
        shadow_sizes.append(shadow_size_bound)
        print(f"{shadow_size_bound} samples required ")
        # calculate a shadow of the appropriate size
        shadow = calculate_classical_shadow(circuit, params, shadow_size_bound, num_qubits)

        # estimate all the observables in O
        estimates.append([estimate_shadow_obervable(shadow, o, k=k) for o in list_of_observables])

    dev_exact = qml.device("lightning.qubit", wires=num_qubits)
    # change the simulator to be the exact one.
    circuit = qml.QNode(circuit_base, dev_exact)

    expval_exact = [
        circuit(params, observable=[o]) for o in list_of_observables
    ]

    for j, error in enumerate(epsilon_grid):
        plt.scatter(
            [shadow_sizes[j] for _ in estimates[j]],
            [np.abs(obs - estimates[j][i]) for i, obs in enumerate(expval_exact)],
            marker=".",
        )
    plt.plot(
        shadow_sizes,
        [e for e in epsilon_grid],
        linestyle="--",
        color="gray",
        label=rf"$\epsilon$",
        marker=".",
    )
    plt.xlabel(r"$N$ (Shadow size) ")
    plt.ylabel(r"$|\langle O_i \rangle_{exact} - \langle O_i \rangle_{shadow}|$")
    plt.legend()
    plt.show()
    # G=load_data(graph_num=2000) #数据中图的个数，因为AIDS是一个图集合
    # subG,center=decomposition_graph(G,1)
    # # networkx 导出网络为edgelist
    # nx.write_edgelist(subG[78], 'data_subg.txt', delimiter=' ', data=False)  # without properties by 'data=False'
    # # 将nodes带属性写入文件：
    # f = open('data_subg_nodes.txt', 'w')
    # for node in subG[78].nodes:
    #     f.write(str(node) + ',' + str(subG[78].nodes[node]['feature']) + '\n')
    # G = nx.Graph(name='test_network')
    # edges=[]
    # # networkx 导入edgelist
    # with open('../data_subg.txt', 'r') as edgeReader:  # 从文件中读取edgelist生成Graph of networkx
    #     for line in edgeReader.readlines():
    #         edges.append(tuple(map(int,line.strip().split(' '))))
    # G.add_edges_from(edges)
    # for i in G.nodes:
    #     print(i)
    #     print(G.nodes[i])
    # print(subG[78].nodes)
    # print(center[78])
    # cont=np.zeros(len(subG))
    # for i in range(len(subG)):
    #     cont[i]=len(subG[i])
    # ma=cont.argmax() #子图中节点数最多的,ma=78
    # train_mapping(subG[78])
    # nx.draw(subG[ma],with_labels=True)
    # plt.show()
