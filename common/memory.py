from collections import namedtuple

import numpy as np
import torch
from .utils import set_seed

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'done', 'next_state'))


class ReplayMemory(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.memory = []
        self.position = 0

    def push(self, *args):
        # print("position:",self.position)
        if len(self.memory) < self.buffer_size:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        # print("state_memory:",args[0].nodes)
        # print("next_state:", args[-1].nodes)
        # print("state_memory:", self.memory[self.position].state.nodes)
        # print("next_state:", self.memory[self.position].next_state.nodes)
        self.position = (self.position + 1) % self.buffer_size

    def sample(self, batch_size, device):
        # set_seed(123)
        indices = np.random.choice(len(self), batch_size, replace=False)
        # print("indices:",indices)
        states, actions, rewards, dones, next_states = zip(
            *[self.memory[idx] for idx in indices])
        # for idx in indices:
        #     print("------------------------------------------------------------")
        #     print("state_sample:",self.memory[idx].state.nodes)
        #     print("next_state_sample:", self.memory[idx].next_state.nodes)
        # states = torch.from_numpy(np.array(states)).to(device)
        actions = torch.from_numpy(np.array(actions)).to(device)
        rewards = torch.from_numpy(np.array(rewards,
                                            dtype=np.float32)).to(device)
        dones = torch.from_numpy(np.array(dones, dtype=np.int32)).to(device)
        # next_states = torch.from_numpy(np.array(next_states)).to(device)
        return states, actions, rewards, dones, next_states

    def __len__(self):
        return len(self.memory)
