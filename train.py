import argparse

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch
import gym
import yaml
import math
from common.mapping import get_mapping
from common.qgcn import QuantumNet
from common.utils import batchgraph,index_mapping,gen_features
from common.utils import set_seed
from trainer import Trainer

parser = argparse.ArgumentParser()

parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--lr", default=0.01, type=float)
# parser.add_argument("--lr_coder", default=0.01, type=float)
parser.add_argument("--batch_size", default=6, type=int)
parser.add_argument("--eps_init", default=1., type=float)
parser.add_argument("--eps_decay", default=0.9, type=int)
parser.add_argument("--eps_min", default=0.01, type=float)
parser.add_argument("--train_freq", default=7, type=int) #调用update_net self.global_step %
parser.add_argument("--target_freq", default=15, type=int) #更新targetnet self.global_step %
parser.add_argument("--memory", default=25, type=int)
parser.add_argument("--alpha", default=0.1, type=float) #0.01
parser.add_argument("--loss", default='SmoothL1', type=str)
parser.add_argument("--optimizer", default='Adam', type=str)
parser.add_argument("--total_episodes", default=15, type=int) #单个子图的训练次数
parser.add_argument("--logging", default=True, type=bool)
parser.add_argument("--log_train_freq", default=1, type=int)
parser.add_argument("--log_eval_freq", default=10, type=int) #test net
parser.add_argument("--log_ckp_freq", default=10, type=int)
parser.add_argument("--device", default='cuda:0', type=str)
parser.add_argument("--e_layers",default=3,type=int)
parser.add_argument("--d_layers",default=3,type=int)
parser.add_argument("--pool_num",default=5,type=int)
parser.add_argument("--subg_num",default=9,type=int) #拆分图大小，=qubit_num
parser.add_argument("--num_min",default=30,type=int) #生成图大小
parser.add_argument("--num_max",default=50,type=int)

args = parser.parse_args()
GROUND_NODE='new'
G_POOL=[]

def gen_graph(g_type): #被调用g_pool_num次
    gen_features_type='featureless'
    max_n = args.num_max
    min_n = args.num_min
    # set_seed(123) #固定每次产生的图大小都一样，可注释，暂用powerlaw
    cur_n = np.random.randint(max_n - min_n + 1) + min_n
    if g_type == 'erdos_renyi':
        g = nx.erdos_renyi_graph(n=cur_n, p=0.15)
    elif g_type == 'powerlaw':
        g = nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)
    elif g_type == 'small-world':
        g = nx.connected_watts_strogatz_graph(n=cur_n, k=8, p=0.1) #k=4
    elif g_type == 'barabasi_albert':
        g = nx.barabasi_albert_graph(n=cur_n, m=4)
    gen_features(g,gen_features_type)
    return g

#在人造图池中训练一次,暂用list结构
def gen_new_graphs():
    print('\ngenerating new training graphs...')
    g_type='small-world'
    for i in range(args.pool_num):
        g = gen_graph(g_type)
        G_POOL.append(g)
    g_type = 'erdos_renyi'
    for i in range(args.pool_num):
        g = gen_graph(g_type)
        G_POOL.append(g)

def train():
    gen_new_graphs()
    for iter,G in enumerate(G_POOL):
        iter+=59
        node_num = G.number_of_nodes()
        samples_num=3*math.ceil(node_num/args.subg_num)
        subg_list=batchgraph(G,args.subg_num)
        # for idx,g in enumerate(subg_list):
        #     fig=plt.figure(figsize=(8,6),dpi=100)
        #     nx.draw(g)
        #     plt.savefig("images/id={}".format(idx))
        if args.device == "auto":
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        # print("len(subg_list):",len(subg_list))
        for idx,subg in enumerate(subg_list):
            index_map = index_mapping(subg)
            features = nx.get_node_attributes(subg, 'feature')
            features = list(features.values())
            # theta=get_mapping(subg.number_of_nodes(),features)
            # print(theta)

            subg.remove_nodes_from([GROUND_NODE])
            env = gym.make('GraphEnv-v0', G=subg)

            if iter==0 and idx==0:
                net = QuantumNet(subg, args.e_layers, args.d_layers, device,first=True)
                target_net = QuantumNet(subg, args.e_layers, args.d_layers, device,first=True)
                target_net.load_state_dict(net.state_dict())
            else:
                net = QuantumNet(subg, args.e_layers, args.d_layers, device,first=False)
                target_net = QuantumNet(subg, args.e_layers, args.d_layers, device,first=False)
                target_net.load_state_dict(net.state_dict())

            if idx>0 or iter>0:
                if idx>0:
                    last_model='./logs/QuanDQN-{}_iter'.format(iter) #同一个iter下的idx依次覆盖,每个idx用的(idx-1)model
                elif iter>0:
                    last_model = './logs/QuanDQN-{}_iter'.format(iter-1)
                state_dict=torch.load(last_model+'/episode_final.pt')
                if len(state_dict['y_weights'][0])-1 != subg.number_of_nodes():continue
                net.load_state_dict(state_dict)
                target_net.load_state_dict(state_dict)
                print("idx={},iter={},ues model={}".format(idx, iter, last_model))

            if idx==0:
                trainer = Trainer(env,
                                  net,
                                  target_net,
                                  gamma=args.gamma,
                                  lr=args.lr,
                                  # lr_coder=args.lr_coder,
                                  batch_size=args.batch_size,
                                  exploration_initial_eps=args.eps_init,
                                  exploration_decay=args.eps_decay,
                                  exploration_final_eps=args.eps_min,
                                  train_freq=args.train_freq,
                                  target_update_interval=args.target_freq,
                                  buffer_size=args.memory,
                                  iter=iter,
                                  subg_id=idx,
                                  index_map=index_map,
                                  alpha=args.alpha,
                                  device=device,
                                  loss_func=args.loss,
                                  optim_class=args.optimizer,
                                  logging=args.logging)
                trainer.learn(args.total_episodes,
                              log_train_freq=args.log_train_freq,
                              log_eval_freq=args.log_eval_freq,
                              log_ckp_freq=args.log_ckp_freq)
                if args.logging:
                    with open(trainer.log_dir + '/config.yaml', 'w') as f:
                        yaml.safe_dump(trainer.get_saveable_dict(), f, indent=2)
            else:
                with open(last_model + '/config.yaml', 'r') as f:
                    config=yaml.safe_load(f)
                trainer = Trainer(env,
                                  net,
                                  target_net,
                                  gamma=config['gamma'],
                                  lr=config['lr'],
                                  # lr_coder=args.lr_coder,
                                  batch_size=config['batch_size'],
                                  exploration_initial_eps=config['exploration_initial_eps'],
                                  exploration_decay=config['exploration_decay'],
                                  exploration_final_eps=config['exploration_final_eps'],
                                  train_freq=config['train_freq'],
                                  target_update_interval=args.target_freq,
                                  buffer_size=config['buffer_size'],
                                  iter=iter,
                                  subg_id=idx,
                                  index_map=index_map,
                                  alpha=config['alpha'],
                                  device=device,
                                  loss_func=args.loss,
                                  optim_class=args.optimizer,
                                  logging=args.logging)
                trainer.learn(args.total_episodes,
                              log_train_freq=args.log_train_freq,
                              log_eval_freq=args.log_eval_freq,
                              log_ckp_freq=args.log_ckp_freq)
                if args.logging:
                    with open(trainer.log_dir + '/config.yaml', 'w') as f:
                        yaml.safe_dump(trainer.get_saveable_dict(), f, indent=2)

if __name__ == '__main__':
    train()