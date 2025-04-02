import os
import sys
from datetime import datetime
# os.environ["PATH"]+=os.pathsep+'E:/Graphviz/bin'
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import csv
import pennylane as qml

from common.utils import g_copy,set_seed
from common.agent import Agent
from common.memory import ReplayMemory
import matplotlib.pyplot as plt
import random
from torchviz import make_dot

total_feat_out = []
total_feat_in = []

torch.autograd.set_detect_anomaly(True)
def backward_hook_fn(module, grad_input, grad_output):
    print(module)
    print("hook grad_output:", grad_output)
    print("hook grad_input:", grad_input)
    total_feat_out.append(grad_output)
    total_feat_in.append(grad_input)

class Trainer:
    def __init__(self,
                 env,
                 net,
                 target_net,
                 gamma,
                 lr,
                 # lr_coder,
                 batch_size,
                 exploration_initial_eps,
                 exploration_decay,
                 exploration_final_eps,
                 train_freq,
                 target_update_interval,
                 buffer_size,
                 iter,
                 subg_id,
                 index_map,
                 alpha,
                 device,
                 loss_func='MSE',
                 optim_class='Adam',
                 logging=False):

        assert loss_func in ['MSE', 'L1', 'SmoothL1'
                             ], "Supported losses : ['MSE', 'L1', 'SmoothL1']"
        assert optim_class in [
            'SGD', 'RMSprop', 'Adam', 'Adagrad', 'Adadelta'
        ], "Supported optimizers : ['SGD', 'RMSprop', 'Adam', 'Adagrad', 'Adadelta']"
        # assert device in ['auto', 'cpu', 'cuda:0'
        #                   ], "Supported devices : ['auto', 'cpu', 'cuda:0']"

        self.env = env
        self.net = net
        self.target_net = target_net
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_decay = exploration_decay
        self.exploration_final_eps = exploration_final_eps
        self.train_freq = train_freq
        self.target_update_interval = target_update_interval
        self.buffer_size = buffer_size
        self.iter=iter
        self.subg_id=subg_id
        self.index_map=index_map
        self.alpha=alpha
        self.device = device
        self.loss_func = loss_func
        self.optim_class = optim_class
        self.logging = logging
        self.recons_flag = True

        self.build()
        self.reset()

    def get_saveable_dict(self):
        attrs=['gamma','lr','batch_size','exploration_initial_eps','exploration_decay','exploration_final_eps',
               'train_freq','buffer_size','alpha']
        return {attr: getattr(self,attr) for attr in attrs}

    def build(self):
        # set networks
        # if self.device == "auto":
        #     self.device = torch.device(
        #         "cuda:0" if torch.cuda.is_available() else "cpu")
        # else:
        #     self.device = torch.device(self.device)
        self.net = self.net.to(self.device)
        self.target_net = self.target_net.to(self.device)
        # self.net.register_backward_hook(backward_hook_fn)

        # set loss

        self.loss_func = getattr(nn, self.loss_func + 'Loss')()

        # set optimizer
        optim_class = getattr(optim, self.optim_class)
        params = []
        params.append({'params':self.net.parameters()})
        # for par in params[0]['params']:
        #     print(par)
        # params.append({'params': self.net.q_layers.parameters(), 'lr': self.lr_coder})
        # params.append({'params': self.net.center_weights, 'lr': self.lr_coder})
        # params.append({'params': self.net.neigb_weights, 'lr': self.lr_coder})
        # self.opt=qml.QNGOptimizer(0.01)
        self.opt = optim_class(params, lr=self.lr)
        # set agent
        self.agent = Agent(self.net,self.index_map, self.env.action_space,
                           self.exploration_initial_eps,
                           self.exploration_decay, self.exploration_final_eps)

        # set memory
        self.memory = ReplayMemory(self.buffer_size)

        # set loggers
        if self.logging:
            # exp_name = datetime.now().strftime("QuanDQN-%d_%m_%Y-%H_%M_%S")
            exp_name='QuanDQN-{}_iter'.format(self.iter)
            if not os.path.exists('./logs/'):
                os.makedirs('./logs/')
            self.log_dir = './logs/{}/'.format(exp_name)
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writer = SummaryWriter(log_dir=self.log_dir)

    def reset(self): #创建buffer
        self.global_step = 0
        self.episode_count = 0
        # self.env.seed(123)
        self.n_actions = self.env.action_space.n
        self.n_actions = self.env.action_space.n
        state = self.env.reset()
        while len(self.memory) < self.buffer_size:
            action = self.agent.get_random_action()
            state_t,next_state_t, reward, done, _, _ = self.env.step(action)
            self.agent.update_action_space(self.env.action_space)
            #创建副本，networkx默认传地址
            state=g_copy(state_t)
            next_state=g_copy(next_state_t)
            # print("-----------------------------")
            # print("state:", state.nodes)
            # print("actions, rewards, dones:", action, reward, done)
            # print("next_state:", next_state.nodes)
            # print("-----------------------------")
            self.memory.push(state, action, reward, done, next_state)
            if done:
                state = self.env.reset()

    def update_lr(self,lr_):
        self.lr=lr_
        self.opt.param_groups[0]['lr']=self.lr
        # print("Trainer opt:",self.opt)

    def re_target(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def update_net(self): # freq==10
        self.net.train()
        self.opt.zero_grad()
        # for par in self.opt.param_groups:
            # print(par)
        # sample transitions
        states, actions, rewards, dones, next_states = self.memory.sample(
            self.batch_size, self.device)

        # compute q-values
        # print("------------------------TRAIN---------------------------")
        if (self.recons_flag):
            state_action_values = self.net(states,self.index_map,self.update_lr,self.re_target) #只有这调用了forward！！这个的输出就是由量子线路那一块measure以后的值
            self.recons_flag=False
        else:state_action_values = self.net(states,self.index_map,self.update_lr)
        # print("---------------------TRAIN FINISH------------------------")
        # state_action_values.register_hook(lambda x: print("q value in register hook:",x))
        fake_actions=[]
        # print(state_action_values)
        for act in actions:
            fake_actions.append(self.index_map[act.item()])
        fake_actions=torch.tensor(fake_actions).to(torch.int64).to(self.device)
        # print("in trainer:",self.device)
        state_action_values = state_action_values.gather(
            1, fake_actions.unsqueeze(-1)).squeeze(-1).to(self.device)
        # print("fake_actions:",fake_actions)
        # print("s_a_values:",state_action_values)

        with torch.no_grad():
            # print("\n")
            # print("------------------------TEST---------------------------")
            next_state_values = self.target_net(next_states, self.index_map)  # 用next-state训练，∴target的qubits比原本net少1
            # print("---------------------TEST FINISH------------------------")
            next_state_values = next_state_values.max(1)[0].detach()
            # print("next_state_values:",next_state_values)
            expected_state_action_values = (1 - dones) * next_state_values.to(
                self.device) * self.gamma + rewards


        # compute loss
        # self.net.e_loss.register_hook(lambda x: print("e_loss in register hook:",x))
        # loss = self.loss_func(state_action_values, expected_state_action_values)+self.alpha*self.net.e_loss
        loss = self.loss_func(state_action_values, expected_state_action_values)
        # dot=make_dot(state_action_values)
        # dot.render("net_dag",format="svg")
        # print("e_loss:",self.net.e_loss)
        # print("loss:",loss)
        # print("================start backprop==================")
        loss.backward()
        # print("================end backprop==================")
        # print("step之前:")
        # for key,val in self.net.named_parameters():
        #     print(key,val.requires_grad,val.is_leaf,val.grad,val)
        self.opt.step()
        # print("step之后:")
        # for key,val in self.net.named_parameters():
        #     print(key,val.requires_grad,val.is_leaf,val.grad,val)
        # print("total_feat_out:",total_feat_out)
        # for idx in range(len(total_feat_in)):
        #     print("out:",total_feat_out[idx])
        #     print("in:",total_feat_in[idx])
        # print("loss:",loss.item())
        # for name, param in self.net.named_parameters():
        #     print(f'{name}\'s grad: {param.grad}')
        # print("**************update_net_down*******************************************")
        return loss.item()

    def update_target_net(self): # freq==30
        # print("-----------------call update_target_net-------------------------------")
        state_dict=self.net.state_dict()
        self.target_net.load_state_dict(state_dict)

    def train_step(self):
        # print("in_train_step:")
        episode_epsilon = self.agent.update_epsilon(self.episode_count)
        episode_steps = 0
        episode_reward = 0
        episode_loss = []
        anc_list=[1.0]
        state = self.env.reset()
        self.agent.update_action_space(self.env.action_space)
        done = False
        act_list=[]
        # print("\n########################  一回合 IN Train_Step #######################")

        while not done:

            # take action
            # print("state:",state.nodes)
            action = self.agent(state, self.device)
            state_t,next_state_t, reward, done, anc,act_list = self.env.step(action)
            self.agent.update_action_space(self.env.action_space)
            anc_list.append(anc['anc'])
            # print("Reward:",reward)

            # update memory
            state = g_copy(state_t)
            next_state = g_copy(next_state_t)
            self.memory.push(state, action, reward, done, next_state)
            state=next_state

            # optimize net
            if self.global_step % self.train_freq == 0: #10
                loss = self.update_net()
                episode_loss.append(loss)

            # update target net
            if self.global_step % self.target_update_interval == 0: #5
                self.update_target_net()

            self.global_step += 1
            episode_reward += reward
            episode_steps += 1

        self.episode_count += 1
        if len(episode_loss) > 0:
            episode_loss = np.mean(episode_loss)
        else:
            episode_loss = 0.
        # print("*************train_step_down******************")
        return {
            'steps': episode_steps,
            'loss': episode_loss,
            'reward': episode_reward,
            'epsilon': episode_epsilon,
            'anc_list':anc_list,
            'act_list':act_list,
            'state':self.env.reset()
        }

    def test_step(self):
        # print("in_test_step:")
        episode_steps = []
        episode_reward = []
        anc_list=[1.0]
        act_list=[]

        state = self.env.reset()
        self.agent.update_action_space(self.env.action_space)
        done = False
        episode_steps.append(0)
        episode_reward.append(0)
        while not done:
            # print:("----------------------------------------------------------")
            # print("state:",state.nodes)
            action = self.agent.get_action(state, self.device)
            # print("action:",action)
            state_t, next_state_t, reward, done, anc, act_list = self.env.step(action)
            self.agent.update_action_space(self.env.action_space)
            state = g_copy(next_state_t)
            # print("state_t:",state.nodes)
            episode_steps[-1] += 1
            episode_reward[-1] += reward
            anc_list.append(anc['anc'])

        episode_steps = np.mean(episode_steps)
        episode_reward = np.mean(episode_reward)
        # print("\n"+"********************test_step_down************************************")
        return {
            'steps': episode_steps,
            'reward': episode_reward,
            'anc_list':anc_list,
            'act_list':act_list,
            'state':self.env.reset()
        }

    def learn(self,
              total_episodes, #30，次数少了，epsilon的更新速度要变快
              log_train_freq=-1,
              log_eval_freq=-1,
              log_ckp_freq=-1):  #50

        # Stats
        postfix_stats = {}
        with tqdm(range(total_episodes), desc="QuanDQN",
                  unit="episode") as tepisodes:    #100次

            for t in tepisodes:

                # train qdqn
                train_stats = self.train_step()

                # update train stats
                postfix_stats['train/reward'] = train_stats['reward'] #留下来的是最后一次reward的结果
                postfix_stats['train/loss'] = train_stats['loss']

                if (t+1) % log_eval_freq == 0: #10

                    # test qdqn
                    test_stats = self.test_step()

                    # update test stats
                    postfix_stats['test/reward'] = test_stats['reward']
                    postfix_stats['test/steps'] = test_stats['steps']

                if not os.path.exists(self.log_dir+'anc'):
                    os.makedirs(self.log_dir+'anc')
                if not os.path.exists(self.log_dir+'anc/'+'train'):
                    os.makedirs(self.log_dir+'anc/'+'train')
                fig_dir_train=self.log_dir+'anc/'+'train'
                if not os.path.exists(self.log_dir+'anc/'+'test'):
                    os.makedirs(self.log_dir+'anc/'+'test')
                fig_dir_test=self.log_dir+'anc/'+'test'
                if self.logging and (t % log_train_freq == 0): #1
                    # n=len(train_stats)-2
                    # items=list(train_stats.items())
                    # for i,(key, item) in enumerate(items[:-2]):
                    #     if i==n-1:break
                    #     self.writer.add_scalar('train/' + key, item, t)
                    # plt.plot(train_stats['anc_list'],label='{}'.format())
                    # print("train_stats['anc_list']",train_stats['anc_list'])
                    # with plt.figure(figsize=(8,6),dpi=100) as fig:
                    # fig=plt.figure(figsize=(8,6),dpi=100)
                    # plt.cla()
                    # ax1=fig.add_subplot(1,2,1)
                    # ax1.plot(train_stats['anc_list'],label=True)
                    # y=train_stats['anc_list']
                    # for i in range(len(y)):
                    #     plt.text(i,y[i],f"{y[i]:.2f}",ha='center',va='bottom')
                    # ax1.set_title('train anc t={}'.format(t))
                    # ax2=fig.add_subplot(1,2,2)
                    # nx.draw(train_stats['state'],with_labels=True)
                    # ax2.set_title('act_list={}'.format(train_stats['act_list']))
                    # plt.savefig(fig_dir_train+'/anc_list_train_{}.png'.format(t))
                    # image_data = plt.imread(fig_dir_train+'/anc_list_train_{}.png'.format(t)).astype(np.float32) / 255.0  #转换为float32，并进行归一化
                    # image_data = np.transpose(image_data, (2, 0, 1))  # 将维度调整为(C, H, W)
                    # image_tensor=torch.from_numpy(image_data)
                    # plt.close(fig)
                    # self.writer.add_image('anc',image_tensor,dataformats='CHW')

                    anc_area = sum(train_stats['anc_list'][1:])
                    filename = 'anc_area_train.csv'
                    filepath = os.path.join('./', filename)
                    with open(filepath, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([anc_area,train_stats['loss']])

                if self.logging and ((t+1) % log_eval_freq == 0): #10
                    # n=len(test_stats)-2
                    # items=list(test_stats.items())
                    # for i,(key, item) in enumerate(items[:-2]):
                    #     if i==n-1:break
                    #     self.writer.add_scalar('test/' + key, item, t)
                    # print("test_stats['anc_list']",test_stats['anc_list'])
                    # with plt.figure(figsize=(8, 6), dpi=100) as fig:
                    # fig = plt.figure(figsize=(8, 6), dpi=100)
                    # plt.cla()
                    # ax1 = fig.add_subplot(1, 2, 1)
                    # ax1.plot(test_stats['anc_list'], label=True)
                    # y = test_stats['anc_list']
                    # for i in range(len(y)):
                    #     plt.text(i,y[i], f"{y[i]:.2f}", ha='center', va='bottom')
                    # ax1.set_title('test anc t={}'.format(t))
                    # ax2 = fig.add_subplot(1, 2, 2)
                    # nx.draw(test_stats['state'],with_labels=True)
                    # ax2.set_title('act_list={}'.format(test_stats['act_list']))
                    # plt.savefig(fig_dir_test + '/anc_list_test_{}.png'.format(t))
                    # image_data = plt.imread(fig_dir_test+'/anc_list_test_{}.png'.format(t)).astype(np.float32) / 255.0  # 转换为float32，并进行归一化
                    # image_data = np.transpose(image_data, (2, 0, 1))  # 将维度调整为(C, H, W)
                    # image_tensor = torch.from_numpy(image_data)
                    # plt.close(fig)
                    # self.writer.add_image('anc', image_tensor, dataformats='CHW')

                    anc_area = sum(test_stats['anc_list'][1:])
                    filename = 'anc_area_test.csv'
                    filepath = os.path.join('./', filename)
                    with open(filepath, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([anc_area])


            if self.logging and (t % log_ckp_freq == 0):
                    torch.save(self.net.state_dict(),
                               self.log_dir + 'episode_{}.pt'.format(t))

            # update progress bar
            tepisodes.set_postfix(postfix_stats) #每次能看见当前reward

            if self.logging and (log_ckp_freq > 0):
                torch.save(self.net.state_dict(),
                           self.log_dir + 'episode_final.pt')
                # sys.exit()
