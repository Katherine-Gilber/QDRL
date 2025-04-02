import numpy as np
from .utils import g_copy

def evaluate_agent(env, net,agent,index_map): #一步预测，先不动起来
    episode_steps = []
    episode_reward = []
    state_action_values=[]

    state = env.reset()
    state_action_values.append(net(state,index_map))
    done = False
    episode_steps.append(0)
    episode_reward.append(0)
    while not done:
        action = agent.get_action(state)
        state_t,next_state_t, reward, done, _ = env.step(action)
        state = g_copy(next_state_t)
        state_action_values.append(net(state,index_map))
        episode_steps[-1] += 1
        episode_reward[-1] += reward

    episode_steps = np.mean(episode_steps)
    episode_reward = np.mean(episode_reward)
    return {'steps': episode_steps, 'reward': episode_reward,'q_values':state_action_values}
