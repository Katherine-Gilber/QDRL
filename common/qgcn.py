import argparse
import numpy as np
import networkx as nx
import pennylane as qml
import torch
import torch.nn as nn
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import math
import time
from .utils import g_copy
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from .MyClaassicalShadow import MyClassicalShadow

# parser = argparse.ArgumentParser()

# parser.add_argument("--device", default='cuda:0', type=str)
# args = parser.parse_args()
ground_node='new'

class Ulayer(object):
    """传子图，把分好的子图作为Ulayer类
    center是subgraph.nodes的最后一个
    """
    def __init__(self,G,totalnum,fake_index,e_layersnum,d_layersnum,center_weights,neigb_weights,y_weights,z_weights,postpro,device='cuda:0',mea=False):
        self.G=G
        self.nodesnum=G.number_of_nodes()
        self.totalnum=totalnum
        self.fake_index=fake_index
        # self.center=fake_index[-1] #utils.py中的decomposition_graph返回的子图中center是subgraph.nodes的最后一个nodes
        features = nx.get_node_attributes(G, 'feature')
        self.features=[]
        for node in G.nodes:
            self.features.append(features.get(node))

        # print(self.features)
        self.e_layersnum = e_layersnum
        self.d_layersnum = d_layersnum
        # self.theta=theta  #theta作用对象应该是整个图的node，不是子图，不然每个子图效率低下，theta是全局共享参数
        self.center_weights=center_weights
        self.neigb_weights=neigb_weights
        self.y_weights = y_weights
        self.z_weights = z_weights
        self.postpro = postpro
        self.device = device
        self.mea=mea
        self.center_idx=fake_index[-1]
        # self.meas=meas
        # self.decode=decode
        # self.measures=measures

    def u_init(self):
        for idx,wire in enumerate(self.fake_index):
            qml.Hadamard(wires=wire)
            for i in range(len(self.features[idx])):
                qml.RX(0.05*math.pi * self.features[idx][i],wires=wire)
            # for i in range(len(self.theta)):
            #     qml.RX(self.theta[i] * self.features[idx][i],wires=wire)

    def u_merge_init(self):   #需要一层跑完所有结点后才能做第二层
        standlen=2**self.totalnum
        merged_state=self.G.nodes[self.center]['layer_inf'][self.layer_idx-1]
        for node in self.G.nodes:
            if node==self.center: continue
            cur_state = self.G.nodes[node]['layer_inf'][self.layer_idx - 1]
            if len(cur_state) < standlen:
                # print(cur_state)
                # merged_state = merged_state + torch.cat(
                #     (cur_state, torch.zeros(standlen - len(cur_state), device=self.device, requires_grad=True)), 0)
                cur_state=F.pad(cur_state,(0,standlen-len(cur_state)))
            merged_state = merged_state+cur_state

            # if self.G.nodes[node]:pass
            # try:
            #     self.G.nodes[node]['layer_inf'][self.layer_idx-1]
            # except:
            #     print("out of index:",node,self.layer_idx)
            # cur_state=self.G.nodes[node]['layer_inf'][self.layer_idx-1]
            # if len(cur_state)<=standlen:
            #     merged_state = merged_state + torch.cat((cur_state,torch.zeros(standlen-len(cur_state),device=self.device,requires_grad=True)),0)
            #     # print(merged_state)
            # else:
            #     # print("len(cur_state)>standlen:")
            #     match_size= int(2**(np.log2(len(cur_state))-np.log2(standlen))) #多出的比特构成的态数目
            #     match=torch.split(cur_state,match_size,0) #tuple 按前面已有态分块
            #     match = list(map(lambda x: torch.mean(x), match)) #求平均，融合多出的态
            #     merged_state = merged_state + torch.stack([t for t in match]).requires_grad_(True)
                # print(merged_state)
        # print(merged_state)
        qml.AmplitudeEmbedding(features=merged_state, wires=range(self.totalnum),normalize=True)

    def u_cov(self,layer):
        # print("in u_cov center_weights:", self.center_weights.is_leaf, self.center_weights.requires_grad)
        qml.RX(self.center_weights[layer][0], wires=self.center_idx)
        qml.RY(self.center_weights[layer][1], wires=self.center_idx)
        qml.RZ(self.center_weights[layer][2], wires=self.center_idx)
        for i in self.fake_index[:-1]:
            # print("in u_cov neigb_weights:",self.neigb_weights,self.neigb_weights.is_leaf,self.neigb_weights.requires_grad) #false true
            qml.RX(self.neigb_weights[layer][0], wires=i)
            qml.RY(self.neigb_weights[layer][1], wires=i)
            qml.RZ(self.neigb_weights[layer][2], wires=i)
        for idx,wire in enumerate(self.fake_index):
            qml.CZ(wires=[wire, self.fake_index[(idx + 1) % len(self.fake_index)]])

    def ecoder(self):
        self.u_init()
        for layer in range(self.e_layersnum):
            self.u_cov(layer)

    def measurement(self):
        # m=qml.measure(wires=self.fake_index[:-1],reset=True)
        # return [qml.expval(m)]
        return [qml.expval(qml.PauliZ(wire)) for wire in self.fake_index[:-1]]  # 保持一致list

    def init_circuit(self):
        for idx,wire in enumerate(self.fake_index[:-1]):
            for i in range(len(self.features[idx])):
                qml.RX(0.05*math.pi * self.features[idx][i],wires=wire)
            # for i in range(len(self.theta)):
            #     qml.RX(self.theta[i] * self.features[idx][i],wires=wire)
        # qml.AmplitudeEmbedding(features=merged_state, wires=fake_index, normalize=True)
        # for idx,wire in enumerate(self.fake_index[:-1]):
        #     qml.RX(self.measures[idx], wires=wire)
        # for idx, wire in enumerate(self.fake_index):
        #     qml.RX(self.measures[idx], wires=wire)  # 一条线对应一个节点熵
    def d_layer(self, y, z):
        # print("in layer:",y_weight,y_weight.requires_grad,z_weight,z_weight.requires_grad)
        # for wire in fake_index:
        #     qml.RY(y_weight[wire],wires=wire)
        # for wire in fake_index:
        #     qml.RZ(z_weight[wire],wires=wire)
        for wire, tmpy in zip(self.fake_index, y):
            qml.RY(tmpy, wires=wire)
        for wire, tmpz in zip(self.fake_index, z):
            qml.RZ(tmpz, wires=wire)
        for idx, wire in enumerate(self.fake_index):
            qml.CZ(wires=[wire, self.fake_index[(idx + 1) % len(self.fake_index)]])
        # for idx, wire in enumerate(self.fake_index):
        #     qml.CZ(wires=[wire, self.fake_index[(idx + 1) % len(self.fake_index)]])

    def decoder(self):
        # @qml.qnode(self.dev, interface='torch',diff_method='best')
        # print("fake index in d_circuit:",inputs['fake_index'])
        # print("inputs['measure']:{},n_qubits:{}".format(len(inputs['measure']),n_qubits))
        if len(self.fake_index)-1 < self.totalnum:
            y = self.y_weights.index_select(1, torch.tensor(self.fake_index, device=self.device))
            z = self.z_weights.index_select(1, torch.tensor(self.fake_index, device=self.device))
        else:
            y = self.y_weights
            z = self.z_weights
        # print("in d_circuit y_weights:",y_weights.requires_grad,y_weights.is_leaf,y_weights.grad_fn)
        for layer_idx in range(self.d_layersnum):
            if layer_idx==0:
                self.init_circuit() #数据重传，标定action
            self.d_layer(y[layer_idx], z[layer_idx])  # 一层layer搭建
        if self.postpro:
            return qml.classical_shadow(wires=self.fake_index[:-1])
            # return qml.classical_shadow(wires=self.fake_index)
        else:
            H = [qml.PauliZ(wire) for wire in self.fake_index[:-1]]
            return qml.shadow_expval(H)
            # return [qml.expval(qml.PauliZ(wire)) for wire in self.fake_index]
        # def d_circuit():
                # return [qml.expval(qml.PauliZ(wire)) for wire in fake_index]
        # return d_circuit

    def u_layer(self):
        dev = qml.device("default.qubit", wires=self.totalnum+1, shots=15000)
        # if self.decode:
        #     dev = qml.device("default.qubit", wires=self.totalnum, shots=200)
        # else:
        #     dev = qml.device("default.qubit", wires=self.totalnum)
        @qml.qnode(dev, interface="torch",diff_method='parameter-shift')
        def circuit():
            self.ecoder()
            return self.decoder()
            # if self.postpro:
            #     return self.decoder()
            # else:
            #     if self.mea:
            #         return self.measurement()
            #     else:
            #         return self.decoder()

            # if self.layer_idx == 0:
            #     self.u_init()
            # else:
            #     self.u_merge_init()
            # if self.meas:
            #     return qml.expval(qml.PauliZ(self.center_idx)) #只用测当前center
                # return qml.vn_entropy(self.center_idx)
                # return qml.expval(qml.PauliZ(wires=[wire for wire in range(self.node_num)]))
                # return qml.vn_entropy(wires=[wire for wire in range(self.node_num)])
            # self.u_cov()
            # if self.decode:
            #     return self.decoder()
            # return qml.state()
        # if self.decode:
        #     fig,ax=qml.draw_mpl(circuit,decimals=5)()
        #     fig.set_size_inches(100,100)
        #     stamp=time.time()
        #     form=time.strftime('%H_%M_%S', time.localtime(stamp))
        #     fig.savefig(form+str(self.layer_idx)+str(self.center)+'fig.png')
        return circuit()
        # test_layer = TorchLayer(circuit, shapes)

def add_gnode(G,theta,e_layers,center_weights,neigb_weights,device):
    G.add_node(ground_node)
    new_edge=[]
    for i in G.nodes:
        if i == ground_node: continue
        new_edge.append((ground_node,i))
    G.add_edges_from(new_edge)
    G.nodes[ground_node]['feature']=torch.zeros(len(theta))
    G.nodes[ground_node]['layer_inf'] = []
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True)
    plt.savefig("add_gnoed.png")
    for layer_idx in range(e_layers):
        if layer_idx==e_layers-1:
            Ulayer(G, ground_node, layer_idx, G.number_of_nodes() - 1, center_weights[layer_idx],
                   neigb_weights[layer_idx], device=device).u_layer().to(device)
        G.nodes[ground_node]['layer_inf'].append(Ulayer(G, ground_node, layer_idx,G.number_of_nodes()-1, center_weights[layer_idx], neigb_weights[layer_idx],device=device).u_layer().to(device))

    state=G.nodes[ground_node]['layer_inf'][-1]
    match_size = 2 #多出一个比特,该比特对应全局节点,要减去该比特，∵在解码中一条线对应一个节点，解码用该state初始化
    match = torch.split(state, match_size, 0)  # tuple 按前面已有态分块
    match = list(map(lambda x: torch.mean(x), match))  # 求平均，融合多出的态
    fin_state = torch.stack([t for t in match]).requires_grad_()
    G.nodes[ground_node]['layer_inf'][-1]=fin_state
    return G

def compute_e_loss(G,mcms,fake_index):
    H=g_copy(G)
    # H = nx.Graph()
    # H.add_nodes_from(G)
    # H.add_edges_from(G.edges)
    # for node in H.nodes:
    #     H.nodes[node]['measure'] = G.nodes[node]['measure']
    #     # print(H.nodes[node]['measure'])
    H.remove_nodes_from([ground_node])
    for idx,node in enumerate(H.nodes):
        H.nodes[node]['fakeindex']=fake_index[idx]
        H.nodes[node]['measure'] = mcms[idx]
    e_loss=0
    for node in H.nodes:
        for neib in H[node]:
            # print(H.nodes[node]['measure'],H.nodes[neib]['measure'])
            e_loss=e_loss+torch.abs(H.nodes[node]['measure']-H.nodes[neib]['measure']) #数非向量，L2用于向量距离
    # print("eloss/2:",e_loss/2)
    return e_loss/2


def encoder(G,totalnum,fake_index,e_layersnum,d_layersnum,center_weights,neigb_weights,y_weights,z_weights,postpro,device):
    # print("in encoder center_weights:",center_weights.is_leaf,center_weights.requires_grad)
    # print("in encoder neigb_weights:", neigb_weights.is_leaf,neigb_weights.requires_grad)
    H=g_copy(G)
    H.add_node(ground_node)
    new_edge = []
    for i in H.nodes:
        if i == ground_node: continue
        new_edge.append((ground_node, i))
    H.add_edges_from(new_edge)
    # G.nodes[ground_node]['feature'] = torch.zeros(len(theta))  # 存疑，需要使用其他节点的初始化吗
    gfea=[]
    gfea.append(nx.degree_centrality(H)[ground_node])
    gfea.append(nx.eigenvector_centrality(H)[ground_node])
    gfea.append(nx.betweenness_centrality(H)[ground_node])
    gfea.append(nx.closeness_centrality(H)[ground_node])
    gfea.append(nx.clustering(H)[ground_node])
    H.nodes[ground_node]['feature']=gfea
    fake_index.append(totalnum)


    if postpro:
        return Ulayer(H, totalnum, fake_index,e_layersnum,d_layersnum,center_weights,neigb_weights,y_weights
                      ,z_weights,postpro,device).u_layer().to(device)
    else:
        # mcms=Ulayer(H, totalnum, fake_index,e_layersnum,d_layersnum,theta,center_weights,neigb_weights,y_weights
        #               ,z_weights,postpro,device,mea=True).u_layer().to(device)
        return Ulayer(H, totalnum, fake_index,e_layersnum,d_layersnum,center_weights,neigb_weights,y_weights
                      ,z_weights,postpro,device).u_layer().to(device)

    # G.remove_nodes_from([ground_node])
    # subG, center, center_idx = decomposition_graph(G, 1)
    # for layer_idx in range(e_layers + 1): #一层全做完了才下一层
    #     for g, ctr,c_idx in zip(subG, center,center_idx):
    #         fake_index=[]
    #         for node in g.nodes:
    #             fake_index.append(index_map[node])
            # if layer_idx==0:
            #     plt.figure(figsize=(8, 6))
            #     nx.draw(g, with_labels=True)
            #     plt.title("center:"+str(ctr))
            #     stamp=time.time()
            #     form=time.strftime('%H_%M_%S', time.localtime(stamp))
            #     plt.savefig(form+str(ctr)+"_subG.png")
            # if layer_idx == e_layers:  # 最后一层加上期望
            #     if g.number_of_nodes()==1:G.nodes[ctr]['measure'] = 0
            #     G.nodes[ctr]['measure'] = Ulayer(g,totalnum,fake_index, ctr, layer_idx,c_idx, meas=True,device=device).u_layer().to(device)
                # print(type(G.nodes[ctr]['measure']))
                # G.nodes[ctr]['measure'] = torch.tensor(G.nodes[ctr]['measure'].tolist()).to(device)
            # else:
            #     # print("in u_cov neigb_weights:", neigb_weights, neigb_weights.is_leaf,neigb_weights.requires_grad)
            #     G.nodes[ctr]['layer_inf'].append(
            #         Ulayer(g,totalnum,fake_index, ctr, layer_idx, c_idx, center_weights[layer_idx], neigb_weights[layer_idx],
            #                theta,device=device).u_layer().to(device))
                # print(G.nodes[ctr]['layer_inf'][-1])

    # measures = []
    # for node in G.nodes:
    #     if node == ground_node: continue
    #     # print(G.nodes[node]['measure'])
    #     measures.append(G.nodes[node]['measure'])
    # measures = torch.tensor(measures, device=device, requires_grad=True)
    #
    # fake_index=[]
    # for node in G.nodes:
    #     fake_index.append(index_map[node])
    # G.add_node(ground_node)
    # new_edge = []
    # for i in G.nodes:
    #     if i == ground_node: continue
    #     new_edge.append((ground_node, i))
    # G.add_edges_from(new_edge)
    # G.nodes[ground_node]['feature'] = torch.zeros(len(theta)) #存疑，需要使用其他节点的初始化吗
    # G.nodes[ground_node]['layer_inf'] = []
    # fake_index.append(totalnum)
    # # plt.figure(figsize=(8, 6))
    # # nx.draw(G, with_labels=True)
    # # plt.savefig("add_gnoed.png")
    # for layer_idx in range(e_layers):
    #     if layer_idx == e_layers-1:
    #         if postpro:
    #             return Ulayer(G,totalnum+1,fake_index, ground_node, layer_idx, G.number_of_nodes() - 1, center_weights[layer_idx],
    #                           neigb_weights[layer_idx], theta, device=device, d_layersnum=d_layersnum,
    #                           measures=measures, y_weights=y_weights, z_weights=z_weights, postpro=postpro, decode=True).u_layer().to(device)
    #         else:
    #             return Ulayer(G,totalnum+1,fake_index, ground_node, layer_idx, G.number_of_nodes() - 1, center_weights[layer_idx],
    #                    neigb_weights[layer_idx], theta,device=device,d_layersnum=d_layersnum,measures=measures,y_weights=y_weights,z_weights=z_weights,
    #                           postpro=postpro,decode=True).u_layer().to(device)
    #     G.nodes[ground_node]['layer_inf'].append(
    #         Ulayer(G,totalnum+1,fake_index, ground_node, layer_idx, G.number_of_nodes() - 1, center_weights[layer_idx],
    #                neigb_weights[layer_idx], theta, device=device).u_layer().to(device))
        # print(G.nodes[ground_node]['layer_inf'][-1],G.nodes[ground_node]['layer_inf'][-1].requires_grad)

    # G = add_gnode(G, theta, e_layers, center_weights, neigb_weights,device)
    # merged_state = G.nodes[ground_node]['layer_inf'][-1]

    # print("measure:",measure)
    # print(merged_state)
    # print("measure:",measure,measure.grad,measure.requires_grad)

    # e_loss=compute_e_loss(G)
    # print("\n")
    # print(merged_state,measure)
    # return e_loss
    # return merged_state,measure, e_loss

def extract(m,indexs):
    result=[]
    for row in m:
        col=[row[idx] for idx in indexs]
        result.append(col)
    return result

def once_init(func):
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, '_initialized'):
            func(self, *args, **kwargs)
            setattr(self, '_initialized', True)
    return wrapper

class QuantumNet(nn.Module):
    @once_init
    def __init__(self, G,e_layers,d_layers,device,first=False):
        super(QuantumNet, self).__init__()
        self.number_nodes=G.number_of_nodes()
        self.n_qubits = G.number_of_nodes() #subg
        self.n_actions = G.number_of_nodes()
        # self.theta=theta
        self.device=device
        self.first=first
        self.e_layers=e_layers
        self.d_layers=d_layers
        self.ep_theta=0.05
        self.des_ep=0.9
        self.alpha_page=0.5
        self.lr=0.01
        self.lr_des=0.99
        self.init_bound=self.ep_theta*math.pi #0.05=ε_θ，超参可调
        # set_seed(123)
        self.center_weights = Parameter(torch.Tensor(self.e_layers, 3))
        nn.init.uniform_(self.center_weights,-self.init_bound,self.init_bound)
        self.neigb_weights = Parameter(torch.Tensor(self.e_layers, 3))
        nn.init.uniform_(self.neigb_weights,-self.init_bound,self.init_bound)
        self.y_weights = Parameter(torch.Tensor(self.d_layers, self.n_qubits+1))
        nn.init.uniform_(self.y_weights,-self.init_bound,self.init_bound)
        self.z_weights = Parameter(torch.Tensor(self.d_layers, self.n_qubits+1))
        nn.init.uniform_(self.z_weights,-self.init_bound,self.init_bound)
        self.dev = qml.device("default.qubit", wires=self.n_qubits, shots=15000) #shots=N_M采用的pauli测量的个数，超参可调
        # self.q_layers = self.decoder()
        # self.e_loss=compute_e_loss(G)
        self.e_loss=0
        self.num_subsystem=2  #k
        self.recons_flag=True

    def reconstruct_theta(self):
        self.ep_theta=self.ep_theta*self.des_ep
        self.init_bound=self.ep_theta*math.pi
        nn.init.uniform_(self.center_weights,-self.init_bound,self.init_bound)
        nn.init.uniform_(self.neigb_weights,-self.init_bound,self.init_bound)
        nn.init.uniform_(self.y_weights,-self.init_bound,self.init_bound)
        nn.init.uniform_(self.z_weights,-self.init_bound,self.init_bound)


#1.output是临时变量，2.outputs初始化为numpy，tensor赋值给numpy梯度消失
    def forward(self, inputs_state,index_map,update_lr=None,re_target=None):
        # print("self.center_weights:", self.center_weights)
        # print("self.neigb_weights:", self.neigb_weights)
        # print("self.y_weights:", self.y_weights)
        # print("self.z_weights:", self.z_weights)
        outputs = torch.zeros((len(inputs_state),self.number_nodes),device=self.device)
        total_e_loss=0
        output=[[] for _ in range(len(inputs_state))]
        fake_index = [[] for _ in range(len(inputs_state))]
        for idx in range(len((inputs_state))):
            for node in inputs_state[idx]:
                fake_index[idx].append(index_map[node])
            # print("len(fake_index[idx]):",len(fake_index[idx]))
            # print("fake_index:",fake_index)
            s_page = 2 * math.log(2) - (1 / (2 ** (len(fake_index[idx]) - 2 * 2 + 1)))
            cnt=0
            subnodeidx=[i for i in range(len(fake_index[idx]))] #?????????????
            if (self.recons_flag and re_target is not None and self.first):
                while(True):
                    bits,recipes=encoder(G=inputs_state[idx],totalnum=self.number_nodes,fake_index=fake_index[idx][:],
                                        e_layersnum=self.e_layers,d_layersnum=self.d_layers,
                                        center_weights=self.center_weights,neigb_weights=self.neigb_weights,
                                        y_weights=self.y_weights,z_weights=self.z_weights,postpro=True,device=self.device)
                    shadow = MyClassicalShadow(bits, recipes, wire_map=subnodeidx) #会自动过滤空白线路
                    # bits,recipes=self.decoder()(inputs,fake_index[idx],postpro=True)
                    # shadow=MyClassicalShadow(bits,recipes,wire_map=fake_index[idx])

                    renyi_entropy=shadow.entropy(fake_index=subnodeidx,subsystem=self.num_subsystem,alpha=2)
                    if renyi_entropy >=self.alpha_page*s_page:
                        self.reconstruct_theta()
                        cnt += 1
                    else:
                        self.recons_flag=False
                        re_target()
                        # print("初始调过后：")
                        # for name, param in self.named_parameters():
                        #     print(f'{name}: {param}')
                        break

                    print("cnt:",cnt)
            else:
                bits, recipes = encoder(G=inputs_state[idx],totalnum=self.number_nodes,fake_index=fake_index[idx][:],
                                        e_layersnum=self.e_layers,d_layersnum=self.d_layers,
                                        center_weights=self.center_weights,neigb_weights=self.neigb_weights,
                                        y_weights=self.y_weights,z_weights=self.z_weights,postpro=True,device=self.device)
                # print("bits:",bits.requires_grad)
                # print("recipes:",recipes.requires_grad)
                # plt.figure(figsize=(25, 25))
                # qml.draw_mpl(self.decoder(), decimals=5)(inputs, fake_index[idx])
                # stamp = time.time()
                # form = time.strftime('%H_%M_%S', time.localtime(stamp))
                # plt.savefig(form + 'fig.png')
                shadow = MyClassicalShadow(bits,recipes,wire_map=subnodeidx)

                renyi_entropy = shadow.entropy(fake_index=subnodeidx, subsystem=self.num_subsystem, alpha=2)
                if renyi_entropy >= self.alpha_page * s_page and update_lr:
                    self.lr=self.lr*self.lr_des
                    update_lr(self.lr)
            # inputs, e_loss = encoder(G=inputs_state[idx], theta=self.theta,
            #                          e_layers=self.e_layers, center_weights=self.center_weights,
            #                          neigb_weights=self.neigb_weights, device=self.device)

            # print("bits:", bits.requires_grad, bits.is_leaf, bits.grad_fn)
            # print("recipes:", recipes.requires_grad, recipes.is_leaf, recipes.grad_fn)
            # fig, ax = qml.draw_mpl(self.decoder(), decimals=5)(inputs, fake_index[idx])
            # fig.savefig(str(idx) + 'fig.png')
            # print(cnt)
            # print(self.ep_theta)
            # print("renyi_entropy:",renyi_entropy)
            # print("spage_entropy:", s_page)
            # print("bits:",bits)
            # print("recipes:",recipes)
            # H=[qml.PauliZ(wire) for wire in fake_index[idx]]


            output[idx]  = encoder(G=inputs_state[idx],totalnum=self.number_nodes,fake_index=fake_index[idx][:],
                                        e_layersnum=self.e_layers,d_layersnum=self.d_layers,
                                        center_weights=self.center_weights,neigb_weights=self.neigb_weights,
                                        y_weights=self.y_weights,z_weights=self.z_weights,postpro=False,device=self.device)
            # total_e_loss=total_e_loss+compute_e_loss(inputs_state[idx],mcms,fake_index[idx])
            # print(output[idx])
            # H = qml.Hamiltonian([1., 1.], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)])
            # output[idx]=shadow.expval(H,self.device)
            # print("output:",output[idx])
            # exit()
            output[idx] = (output[idx]+1)/2 #[-1,1]调为[0,1]
            output[idx]=torch.cat((output[idx],torch.zeros(self.number_nodes-len(output[idx]),device=self.device, requires_grad=True)),0)

            # output[idx].register_hook(lambda x: print("output in register hook:", x))
        # print("fake_index:",fake_index)
        output=torch.stack(output)
        for idx in range(len((inputs_state))):
            for j in range(len(fake_index[idx])):
                outputs[idx][fake_index[idx][j]]=output[idx][j] #索引映射问题
        # self.e_loss=total_e_loss/len(inputs_state)
        # print("outputs:",outputs,outputs.requires_grad)
        return outputs

    # def remap_G(self,index_map):
    #     self.q_layers = decode(n_qubits=self.n_qubits, d_layers=d_layers)

def main():
    # 保存图的基本信息，用utils.py中的load_graph
    # G = load_data(graph_num=2000)  # 数据中图的个数，因为AIDS是一个图集合
    # subG, center = decomposition_graph(G, 1)
    # cont=np.zeros(len(subG))
    # for i in range(len(subG)):
    #     cont[i]=len(subG[i])
    # ma=cont.argmax() #子图中节点数最多的,ma=78
    # G=subG[ma]
    features=np.array([[2.0, 0.0, 15.735899925231934, -2.10479998588562],
                       [1.0, 0.0, 7.557499885559082, -2.8722000122070312],
                       [1.0, 0.0, 12.816900253295898, -3.8661999702453613],
                       [1.0, 0.0, 22.84709930419922, 2.5896999835968018],
                       [1.0, 0.0, 28.809900283813477, 0.6226999759674072],
                       [1.0, 0.0, 28.066699981689453, -0.04639999940991402],
                       [1.0, 0.0, 27.115699768066406, 0.26260000467300415],
                       ])
    features=normalize(features, axis=0, norm='max')  # 特征归一化
    G = nx.Graph()
    G.add_edges_from([(0, 1), (2, 3), (1, 2), (1, 3), (0, 3),(1,4),(4,5),(4,6),(5,6)])
    for i in G.nodes:
        G.nodes[i]['feature']=features[i] #这个好像没啥必要？
        G.nodes[i]['layer_inf']=[]
        G.nodes[i]['measure'] = 0
    # nx.draw(G, with_labels=True)
    plt.show()


if __name__ == '__main__':
    main()
