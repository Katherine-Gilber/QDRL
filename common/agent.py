import numpy as np
import torch
import networkx as nx
from .utils import set_seed

class Agent:
    def __init__(self,
                 net,
                 index_map,
                 action_space=None,
                 exploration_initial_eps=None,
                 exploration_decay=None,
                 exploration_final_eps=None):

        self.net = net
        self.index_map=index_map
        self.action_space = action_space
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_decay = exploration_decay
        self.exploration_final_eps = exploration_final_eps
        self.epsilon = 0.

    def __call__(self, state, device=torch.device('cpu')):
        if np.random.random() < self.epsilon:
            # print("random!!!!!")
            action = self.get_random_action()
        else:
            action = self.get_action(state, device)
            # print("\nselect!!!!!")

        return action

    # 随机选
    def get_random_action(self):
        # print("use_random_action:")
        # set_seed(123)
        # print('in get random action action space',self.action_space.get())
        action = self.action_space.sample()
        # print('in get random action selected action',action)
        return action

    #选最高
    def get_action(self, state,device=torch.device('cpu')):
        # if not isinstance(state, torch.Tensor):
        #     adj_matrix = nx.adjacency_matrix(state)
        #     # 将邻接矩阵转换为稀疏张量
        #     state = torch.FloatTensor(np.array(adj_matrix.todense()))
        #     # state = torch.tensor([state])

        # if device.type != 'cpu':
        #     state = state.cuda(device)
        # print("use_get_action!!")
        q_values = self.net.eval()([state],self.index_map) #state一个图
        _, action = torch.max(q_values, dim=1)
        action=[key for key,val in self.index_map.items() if val==action]
        return action[0]

    def update_action_space(self,action_space):
        self.action_space=action_space

    def update_epsilon(self, step):
        self.epsilon = max(
            self.exploration_final_eps, (self.exploration_initial_eps - self.exploration_final_eps) *
            self.exploration_decay**step)
        # print("\n----------------agent epsilon:--------------",self.epsilon)
        return self.epsilon
