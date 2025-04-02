import csv
import os
import pandas as pd
import gym
import numpy as np
import torch
from datetime import datetime
import yaml
import ast
import re
import urllib.request
import io
import zipfile
import matplotlib.pyplot as plt
import networkx as nx
import math
import argparse
from common.mapping import get_mapping
from common.utils import batchgraph,index_mapping,gen_features
from common.qgcn import QuantumNet
from common import metrics as mt
from common.agent import Agent
from common.evaluator import evaluate_agent

parser=argparse.ArgumentParser()
parser.add_argument("--device",default='cuda:0',type=str)
parser.add_argument("--last_iter",default=47 ,type=int)
parser.add_argument("--subg_num",default=9,type=int)
parser.add_argument("--e_layers",default=3,type=int)
parser.add_argument("--d_layers",default=3,type=int)
# parser.add_argument("--save_path",default="./data/USAir/USAir",type=str)
args=parser.parse_args()

def read_football():
    # url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"
    # sock = urllib.request.urlopen(url)  # open URL
    # s = io.BytesIO(sock.read())  # read into BytesIO "file"
    # sock.close()
    s='D:/Downloads/football.zip'

    zf = zipfile.ZipFile(s)  # zipfile object
    txt = zf.read("football.txt").decode()  # read info file
    gml = zf.read("football.gml").decode()  # read gml data
    # throw away bogus first line with # from mejn files
    gml = gml.split("\n")[1:]
    G = nx.parse_gml(gml)  # parse gml data
    # nx.write_gml(G,"./data/football/football.gml")
    return G

def read_USAIR():
    path = "./data/USAir/USAir.txt"
    edges=np.loadtxt(path,dtype=int,delimiter=' ')
    G = nx.Graph()
    for e in edges:
        G.add_edge(*e)
    return G

def read_VK():
    path = "./data/Valdis_Krebs/Valdis_Krebs.txt"
    edges=np.loadtxt(path,dtype=int,delimiter=' ')
    G = nx.Graph()
    for e in edges:
        G.add_edge(*e)
    return G

def read_CG():
    path = "./data/Corruption_Gcc/Corruption_Gcc.txt"
    edges=np.loadtxt(path,dtype=int,delimiter=' ')
    G = nx.Graph()
    for e in edges:
        G.add_edge(*e)
    return G

def read_CN():
    path = "./data/CrimeNet/CrimeNet.txt"
    edges=np.loadtxt(path,dtype=int,delimiter=' ')
    G = nx.Graph()
    for e in edges:
        G.add_edge(*e)
    return G

# number: label
def relable(G,gen_features_type='default'):
    orig_map={}
    for idx,node in enumerate(G.nodes):
        orig_map[idx]=node
    H=nx.convert_node_labels_to_integers(G)
    gen_features(H)
    return H,orig_map

#给出分数predict
def mytest(G,G_name,last_iter):
    # G=nx.read_gml('./data/synthetic/synthetic.gml')
    # node_num = G.number_of_nodes()
    # print(G.nodes)
    G,orig_map=relable(G)
    # for last_iter in range(47, 69):
    # print("samples_num:",samples_num)
    subg_list = batchgraph(G, args.subg_num)
    # print(node_frequency)
    # print("len(subg_list):",len(subg_list))
    # print("len(subg_list):",len(subg_list))
    all_q_values=np.zeros(G.number_of_nodes())
    if args.device == "auto":
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    for idx, subg in enumerate(subg_list):
        # print(idx)
        index_map = index_mapping(subg)
        # features = nx.get_node_attributes(subg, 'feature')
        # features = list(features.values())
        # theta = get_mapping(subg.number_of_nodes(), features) #每次算新的
        # print("theta:",theta)

        env = gym.make('GraphEnv-v0',G=subg)


        model_name = 'QuanDQN-{}_iter'.format(last_iter)
        model_path='./logs/'+model_name
        # with open(model_path + '/config.yaml', 'r') as f:
        #     hparams = yaml.safe_load(f)

        # net = QuantumNet(subg, 3, 3, theta)
        # print(id(subg))
        # net = QuantumNet(subg, hparams['e_layers'], hparams['d_layers'], theta,device)
        net = QuantumNet(subg, args.e_layers,args.d_layers, device)
        state_dict = torch.load(model_path + '/episode_final.pt')
        net.load_state_dict(state_dict)
        # print(state_dict)

        if device.type=='cuda':
            tmp_q=net([env.reset()],index_map).cpu().detach().numpy()
        else:
            tmp_q = net([env.reset()], index_map).detach().numpy()
        # for node in subg.nodes:
        #     for key,value in subg.nodes[node].items():
        #         print(key,value)
        # print(subg.nodes)
        # print("this subg q value:",tmp_q)
        for idx,node in enumerate(subg.nodes):
            all_q_values[node]=max(tmp_q[0][idx], all_q_values[node])
        # agent = Agent(net)
        # result = evaluate_agent(env, agent, index_map)
    q_sum=0
    for i in all_q_values:
        q_sum+=i
    for i in range(len(all_q_values)):
        all_q_values[i]=round(all_q_values[i]/q_sum,6)
    # print(all_q_values)
    # all_q_values /= node_frequency
    # max_value=np.max(all_q_values)
    # min_value=np.min(all_q_values)
    # dif=max_value-min_value
    # all_q_values=(all_q_values-min_value)/dif
    # print(all_q_values)
    # all_q_values=all_q_values/max_value
    # print(all_q_values)
    path='./data/'+G_name
    if not os.path.exists(path+'/QGCN/'):
        os.makedirs(path+'/QGCN/')
    path=path+'/QGCN/'
    draw_q(G,all_q_values,path+'iter={}_'.format(last_iter))

    labels={}
    for idx,q in enumerate(all_q_values):
        labels[idx]=q
    nx.set_node_attributes(G,labels,'qvalue')
    nx.set_node_attributes(G, orig_map, 'name')

    for node in G.nodes:
        for key,value in G.nodes[node].items():
            G.nodes[node][key] = str(value)
            # if key =='feature':
            #     G.nodes[node][key]=str(value)
            # else:
            #     G.nodes[node][key] = str(value)
    nx.write_gml(G, path=path+'iter={}.gml'.format(last_iter))
    # for node in G.nodes:
    #     f=re.split(r'[\s\[\]]+', G.nodes[node]['feature'])
    #     f=[float(item) for item in f if item]
    #     G.nodes[node]['feature'] = np.array(f)
        # print(G.nodes[node]['feature'])
    # nx.write_gml(G,path='./data/synthetic/synthetic_1.gml')

    sort_list=sorted(labels.items(),key=lambda x: x[1],reverse=True)
    act_list=[x[0] for x in sort_list]
    anc_list=[]
    total=connectivity(G)
    for i in range(len(act_list)):
        anc_one=anc(act_list[:i],G,total)
        anc_list.append(anc_one)
        if anc_one<0.01:break
    act_lab=[orig_map[i] for i in act_list]
    gap=1/G.number_of_nodes()
    x_lab=[0]
    for i in act_list[1:]:
        tmp=x_lab[-1]+gap
        x_lab.append(tmp)
    cul_area=0
    for i in range(len(anc_list)):
        cul_area+=anc(act_list[:i],G,total)
    draw_anc(G,anc_list,act_lab,x_lab[:len(anc_list)],cul_area,path+'_iter={}_'.format(last_iter))
    write_anc(anc_list,path)
    gcc_size(act_list,G,path)

def qgcn_p(G,path,g_id,last_iter):
    G,orig_map=relable(G)
    subg_list = batchgraph(G, args.subg_num)
    all_q_values=np.zeros(G.number_of_nodes())
    if args.device == "auto":
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    for idx, subg in enumerate(subg_list):
        if subg.number_of_nodes()==1:continue
        index_map = index_mapping(subg)
        env = gym.make('GraphEnv-v0',G=subg)

        model_name = 'QuanDQN-{}_iter'.format(last_iter)
        model_path='./logs/'+model_name
        net = QuantumNet(subg, args.e_layers,args.d_layers, device)
        state_dict = torch.load(model_path + '/episode_final.pt')
        state_dict['y_weights']=state_dict['y_weights'][:,:subg.number_of_nodes()+1]
        state_dict['z_weights'] = state_dict['z_weights'][:, :subg.number_of_nodes()+1]

        net.load_state_dict(state_dict)

        if device.type=='cuda':
            tmp_q=net([env.reset()],index_map).cpu().detach().numpy()
        else:
            tmp_q = net([env.reset()], index_map).detach().numpy()
        for idx,node in enumerate(subg.nodes):
            all_q_values[node]=max(tmp_q[0][idx], all_q_values[node])
    q_sum=0
    for i in all_q_values:
        q_sum+=i
    for i in range(len(all_q_values)):
        all_q_values[i]=round(all_q_values[i]/q_sum,6)
    path_gid=path+str(g_id)+'/'
    if not os.path.exists(path_gid):
        os.makedirs(path_gid)

    labels={}
    for idx,q in enumerate(all_q_values):
        labels[idx]=q

    sort_list=sorted(labels.items(),key=lambda x: x[1],reverse=True)
    act_list=[x[0] for x in sort_list]
    anc_list=[]
    total=connectivity(G)
    for i in range(len(act_list)):
        anc_one=anc(act_list[:i],G,total)
        anc_list.append(anc_one)
        if anc_one<0.01:break
    write_anc(anc_list,path_gid)
    gcc_size(act_list,G,path_gid)
    anc_bound(anc_list,path_gid)
    gcc_bound(act_list,G,path_gid)


def others(method,G,G_name):
    res={}
    path='./data/'+G_name+'/'
    if method=='degree':
        res=mt.degree(G)
    if method=='betweeness':
        res=mt.betweeness_centrality(G)
    if method == 'coreness':
        res = mt.coreness(G)
    if method == 'eigenector':
        res = mt.eigenector_centrality(G)
    if method == 'page_rank':
        res = mt.page_rank(G)
    if not os.path.exists(path + method):
        os.makedirs(path + method)

    path=path + method+'/'
    nx.set_node_attributes(G,res,'score')
    H=nx.convert_node_labels_to_integers(G)

    nx.write_gml(H,path+G_name+'.gml')
    draw_q(G, list(res.values()), path)
    act_list = [k for k, v in sorted(res.items(), key=lambda x: x[1],reverse=True)]
    print(method)
    print(act_list)
    anc_list = []
    total = connectivity(G)
    for i in range(len(act_list)):
        anc_one = anc(act_list[:i], G, total)
        anc_list.append(anc_one)
        if anc_one < 0.01: break
    gap = 1 / G.number_of_nodes()
    x_lab = [0]
    for i in act_list[1:]:
        tmp = x_lab[-1] + gap
        x_lab.append(tmp)
    cul_area = 0
    for i in range(len(anc_list)):
        cul_area += anc(act_list[:i], G, total)
    draw_anc(G, anc_list, anc_list, x_lab[:len(anc_list)], cul_area, path+'anc_line')
    write_anc(anc_list, path)
    gcc_size(act_list,G,path)


def connectivity(G):
    delta=0
    for i in nx.connected_components(G):
        tmp=len(i)
        c=(tmp*(tmp-1))/2
        delta+=c
    return delta

def anc(node_list,G,total):
    H = nx.Graph()
    H.add_nodes_from(G)
    H.add_edges_from(G.edges)
    H.remove_nodes_from(node_list)
    tmp=connectivity(H)
    tmp/=total
    return tmp

def gcc_size(node_list,G,path):
    H=nx.Graph()
    H.add_nodes_from(G)
    H.add_edges_from(G.edges)
    gcc_list=[1]
    x_list=[0]
    gap=1/G.number_of_nodes()
    for node in node_list:
        H.remove_nodes_from([node])
        gcc=sorted(nx.connected_components(H),key=len,reverse=True)
        if len(gcc)>0:
            gcc_list.append(len(gcc[0])/G.number_of_nodes())
            x_list.append(x_list[-1]+gap)
    try:
        df=pd.read_csv(path+'gcc_list.csv')
    except FileNotFoundError:
        df=pd.DataFrame()
    cur_time=datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    # df=df.assign(NewColumn=gcc_list)
    df[cur_time]=pd.Series(gcc_list)
    df.to_csv(path+'gcc_list.csv',index=False)
    # fig = plt.figure(figsize=(8, 6), dpi=100)
    # plt.plot(x_list,gcc_list,label=True)
    # plt.title('gcc_list')
    # plt.savefig(path+'gcc.png')
    # plt.close()

def gcc_bound(node_list,G,path):
    H=nx.Graph()
    H.add_nodes_from(G)
    H.add_edges_from(G.edges)
    cnt=0
    for node in node_list:
        H.remove_nodes_from([node])
        cnt+=1
        gcc=sorted(nx.connected_components(H),key=len,reverse=True)
        if len(gcc[0])/G.number_of_nodes()<=0.1:
            res=cnt/G.number_of_nodes()
            filename='gcc_bound.csv'
            filepath=os.path.join(path,filename)
            with open(filepath,mode='a',newline='') as file:
                writer=csv.writer(file)
                writer.writerow([res])
            break

def draw_anc(G,anc_list,act_list,x_lab,area,path):
    fig = plt.figure(figsize=(8, 6), dpi=100)
    plt.cla()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x_lab,anc_list, label=True)
    # ax1.set_title("test t={}".format(last_iter),loc='left')
    ax1.text(x_lab[int(len(x_lab)*0.6)],0.8,'len(<0.01)={}'.format(len(anc_list)))
    ax1.text(x_lab[int(len(x_lab)*0.6)],0.7,'area={:.4f}'.format(area))
    y = anc_list
    for i in range(len(y)):
        plt.text(x_lab[i], y[i], f"{y[i]:.2f}", ha='center', va='bottom',fontsize=8)
    ax2 = fig.add_subplot(1, 2, 2)
    nx.draw(G, with_labels=True)
    ax2.set_title('act_list')
    # plt.savefig(args.save_path+'_iter={}.png'.format(args.last_iter))
    plt.savefig(path+'.png')
    plt.close(fig)

def draw_q(G,qvalue,path):
    data=np.around(qvalue,decimals=4)
    # 创建节点列表和对应的标签值列表
    nodes= G.nodes
    # nodes = [orig_map[node] for node in G.nodes]
    # labels = data[0]
    labels=data
    formatted_data={i:j for i,j in zip(nodes,labels)}

    pos = nx.spring_layout(G)
    nodes_color = [value for _, value in formatted_data.items()]
    nx.draw_networkx_nodes(G, pos=pos, nodelist=nodes, node_color=nodes_color, cmap='coolwarm',
                           vmin=min(labels), vmax=max(labels), node_size=200)
    nx.draw_networkx_edges(G, pos=pos)
    nx.draw_networkx_labels(G, pos=pos, labels=formatted_data, font_size=7, verticalalignment='center',
                            horizontalalignment='left')
    # 创建一个虚拟的标量映射对象，并将其用作可映射对象
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=min(labels), vmax=max(labels)))
    sm.set_array([])  # 设置数组为空
    plt.colorbar(sm)  # 添加颜色条
    plt.title('q_value')
    # plt.savefig(args.save_path + '_q_iter={}.png'.format(args.last_iter))
    plt.savefig(path+'all_q_value.png')
    plt.close()

def write_anc(anc_list,path):
    try:
        df=pd.read_csv(path+'anc_list.csv')
    except FileNotFoundError:
        df=pd.DataFrame()
    cur_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    df[cur_time]=pd.Series(anc_list)
    df.to_csv(path+'anc_list.csv',index=False)

def anc_bound(anc_list,path):
    res=sum(anc_list)
    filename='anc_bound.csv'
    filepath=os.path.join(path,filename)
    with open(filepath,mode='a',newline='') as file:
        writer=csv.writer(file)
        writer.writerow([res])

def generate_graph_res(methods,probs,path,total_graph_nums):
    for p in probs:
        for method in methods:
            if not os.path.exists(path + '{:.2f}/'.format(p) + method):
                os.makedirs(path + '{:.2f}/'.format(p) + method)
            path_method = path + '{:.2f}/'.format(p) + method + '/'
            # for i in range(total_graph_nums):
            for i in range(100):
                g_path=path+'{:.2f}/{:d}.gml'.format(p,i)
                G = nx.read_gml(g_path)
                if method == 'QGCN':
                    for last_iter in [53,61,63]:
                        qgcn_p(G,path_method,i,last_iter)
                else:
                    res={}
                    if method=='degree':
                        res=mt.degree(G)
                    if method=='betweeness':
                        res=mt.betweeness_centrality(G)
                    if method == 'coreness':
                        res = mt.coreness(G)
                    if method == 'eigenector':
                        res = mt.eigenector_centrality(G)
                    if method == 'page_rank':
                        res = mt.page_rank(G)

                    act_list = [k for k, v in sorted(res.items(), key=lambda x: x[1], reverse=True)]
                    anc_list = []
                    total = connectivity(G)
                    for j in range(len(act_list)):
                        anc_one = anc(act_list[:j], G, total)
                        anc_list.append(anc_one)
                        if anc_one < 0.01: break
                    write_anc(anc_list, path_method)
                    gcc_size(act_list, G, path_method)
                    anc_bound(anc_list,path_method)
                    gcc_bound(act_list, G, path_method)



def gen_graph(total_graph_nums,graph_size): #被调用g_pool_num次
    # for p in [round(x*0.05,2) for x in range(1,11)]:
    #     for i in range(total_graph_nums):
    #         g = nx.erdos_renyi_graph(n=graph_size, p=p)
    #         path='./data/generate/ER/{:.2f}/{:d}.gml'.format(p,i)
    #         nx.write_gml(g,path)
    # for p in [round(x*0.01,2) for x in range(1,16)]:
    #     for i in range(total_graph_nums):
    #         g = nx.connected_watts_strogatz_graph(n=graph_size,k=4, p=p)
    #         path='./data/generate/SW/{:.2f}/{:d}.gml'.format(p,i)
    #         nx.write_gml(g,path)
    for m in [1,2,3,4,5]:
        for i in range(total_graph_nums):
            g = nx.barabasi_albert_graph(n=graph_size,m=m)
            path='./data/generate/BA/{:d}/{:d}.gml'.format(m,i)
            nx.write_gml(g,path)



if __name__ == '__main__':
    G_name='random'
    if G_name=='football':
        G=read_football()
    if G_name=='USAir':
        G=read_USAIR()
    if G_name=='Valdis_Krebs':
        G=read_VK()
    if G_name=='Corruption_Gcc':
        G=read_CG()
    if G_name=='CrimeNet':
        G=read_CN()
    if G_name=='Karate_Club':
        G=nx.karate_club_graph()
    if G_name == 'synthetic':
        G=nx.read_gml('./data/synthetic/synthetic.gml')
    if G_name=='random':
        total_graph_nums=100
        graph_size=40
        gen_graph(total_graph_nums,graph_size)
        # methods = ['betweeness', 'coreness','degree', 'eigenector', 'page_rank', 'QGCN']
        methods = ['QGCN']
        # probs=[round(x*0.05,2) for x in range(1,11)]
        # path='./data/generate/ER/'
        probs=[round(x*0.01,2) for x in range(7,16)]
        path = './data/generate/SW/'
        generate_graph_res(methods,probs,path,total_graph_nums)
    # method='dqgcn'
    # if method=='dqgcn': #dqgcn, degree, betweeness, coreness, eigenector, page_rank
    #         for last_iter in range(47,69):
    #             mytest(G,G_name,last_iter)
    # else:
    #     others(method,G,G_name)