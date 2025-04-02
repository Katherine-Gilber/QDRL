import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from random import choice
import networkx as nx
import matplotlib.pyplot as plt

def connectivity(G):
    delta=0
    for i in nx.connected_components(G):
        tmp=len(i)
        c=(tmp*(tmp-1))/2
        delta+=c
    return delta

def anc(node_list,G):
    if len(node_list)==0:return 1
    H = nx.Graph()
    H.add_nodes_from(G)
    H.add_edges_from(G.edges)
    total = connectivity(H)
    r = 0
    for i in range(1, len(node_list) + 1):
        H.remove_nodes_from(node_list[:i])
        tmp = connectivity(H)
        tmp /= total
        r += tmp
    r /= len(node_list)
    return r

class MyAction(object):
    def __init__(self,G):
        self.G=G
        self.action_space=list(G.nodes)
        self.n=G.number_of_nodes()

    def __call__(self):
        return self.action_space

    def sample(self):
        self.action_space=list(self.G.nodes)
        # print("action_space:",self.action_space)
        return choice(self.action_space)
    def get(self):
        return self.action_space

class GraphEnv(gym.Env):
    def __init__(self,G):
        self.origG=G
        self.state=nx.Graph()
        self.state.add_nodes_from(G)
        self.state.add_edges_from(G.edges)
        for i in G.nodes:
            self.state.nodes[i]['feature'] = G.nodes[i]['feature']
            self.state.nodes[i]['layer_inf'] = []
            self.state.nodes[i]['vn_entropy'] = 0
        self.next_state = nx.Graph()
        self.next_state.add_nodes_from(G)
        self.next_state.add_edges_from(G.edges)
        for i in G.nodes:
            self.next_state.nodes[i]['feature'] = G.nodes[i]['feature']
            self.next_state.nodes[i]['layer_inf'] = []
            self.next_state.nodes[i]['vn_entropy'] = 0
        self.action_space = MyAction(self.next_state)
        self.bud_anc=0.05
        self.action_list=[]

        self.seed()
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.state.remove_nodes_from(self.action_list)
        self.action_list.append(action)
        self.next_state.remove_node(action)
        self.action_space=MyAction(self.next_state)

        reward=anc(self.action_list[:-1],self.origG)-anc(self.action_list,self.origG)
        anc_one=anc(self.action_list,self.origG)
        done=bool(anc_one<=self.bud_anc)

        return self.state, self.next_state,reward, done, {'anc':anc_one},{'action':self.action_list}

    def reset(self):
        self.state = nx.Graph()
        self.state.add_nodes_from(self.origG)
        self.state.add_edges_from(self.origG.edges)
        for i in self.state.nodes:
            self.state.nodes[i]['feature'] = self.origG.nodes[i]['feature']
            self.state.nodes[i]['layer_inf'] = []
            self.state.nodes[i]['vn_entropy'] = 0
        self.next_state = nx.Graph()
        self.next_state.add_nodes_from(self.origG)
        self.next_state.add_edges_from(self.origG.edges)
        for i in self.state.nodes:
            self.next_state.nodes[i]['feature'] = self.origG.nodes[i]['feature']
            self.next_state.nodes[i]['layer_inf'] = []
            self.next_state.nodes[i]['vn_entropy'] = 0
        self.action_space = MyAction(self.state)
        self.action_list=[]
        return self.state

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

        if self.state is None:
            return None

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            plt.pause(2)  # pause a bit so that plots are updated
            plt.title('Action: {}'.format(self.action_list[-1]))
            nx.draw(self.state, with_labels=True)
            plt.show()
            self.viewer.close()
            self.viewer = None