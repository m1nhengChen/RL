import numpy as np
import pandas as pd
import torch
import os
from collections import OrderedDict

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
device = torch.device('cuda:{}'.format(device_ids[0]))


class Fusion_policy():
    '''
    1: hit
    0: stick
    '''

    def _init_(self, threshold=17) -> None:
        self.threshold = threshold

    def load(self, csv_m, csv_q, model_path, csv_o):
        csv_m = pd.read_csv(csv_m, encoding="utf-8")
        self.csv_m = np.array(csv_m)
        csv_q = pd.read_csv(csv_q, encoding="utf-8")
        self.csv_q = np.array(csv_q)
        csv_o = pd.read_csv(csv_o, encoding="utf-8")
        self.csv_o = np.array(csv_o)
        self.agent = torch.load(model_path, map_location=device)

    def act_player(self, obs):
        if obs[0] < 12:
            return 1
        else:
            # vote = 0
            # index = (obs[0] - 12) * 20 + (obs[1] - 1) + obs[2]
            # # monte-carlo method
            # result = self.csv_m[index][3]
            # if result == 'hit':
            #     vote += 1
            # # Q-learning method
            # result = self.csv_q[index][3]
            # if result == 'hit':
            #     vote += 1
            # # DQN method
            # obs = np.array([obs[0], obs[1]])
            # legal_actions = OrderedDict({i: None for i in range(2)})
            # state = {'obs': obs, 'legal_actions': legal_actions}
            # # save_dir = 'model.pth'
            # # agent = torch.load(save_dir, map_location=device)
            # pred = self.agent.predict(state)
            # # 0: hit; 1:stand
            # if pred[0] > pred[1]:
            #     # print('hit')
            #     vote += 1
            # # counting the votes
            # if vote < 2:
            #     return 0
            # else:
            #     return 1
            # DQN method
            obs = np.array([obs[0], obs[1]])
            legal_actions = OrderedDict({i: None for i in range(2)})
            state = {'obs': obs, 'legal_actions': legal_actions}
            # save_dir = 'model.pth'
            # agent = torch.load(save_dir, map_location=device)
            pred = self.agent.predict(state)
            # 0: hit; 1:stand
            if pred[0] > pred[1]:
                return 1
            else:
                return 0
            # if obs[0] < 17: return 1
            # return 0
            # index = (obs[0] - 12) * 20 + (obs[1] - 1) + obs[2]
            # # opt method
            # result = self.csv_o[index][3]
            # if result == 'hit':
            #     return 1
            # return 0

    def act_dealer(self, obs):
        if obs[0] < 12:
            return 1
        else:
            # vote = 0
            # index = (obs[0] - 12) * 20 + (obs[1] - 1) + obs[2]
            # # monte-carlo method
            # result = self.csv_m[index][3]
            # if result == 'hit':
            #     vote += 1
            # # Q-learning method
            # result = self.csv_q[index][3]
            # if result == 'hit':
            #     vote += 1
            # # DQN method
            # obs = np.array([obs[0], obs[1]])
            # legal_actions = OrderedDict({i: None for i in range(2)})
            # state = {'obs': obs, 'legal_actions': legal_actions}
            # # save_dir = 'model.pth'
            # # agent = torch.load(save_dir, map_location=device)
            # pred = self.agent.predict(state)
            # # 0: hit; 1:stand
            # if pred[0] > pred[1]:
            #     # print('hit')
            #     vote += 1
            # # counting the votes
            # if vote < 2:
            #     return 0
            # else:
            #     return 1
            # DQN method
            # obs = np.array([obs[0], obs[1]])
            # legal_actions = OrderedDict({i: None for i in range(2)})
            # state = {'obs': obs, 'legal_actions': legal_actions}
            # # save_dir = 'model.pth'
            # # agent = torch.load(save_dir, map_location=device)
            # pred = self.agent.predict(state)
            # # 0: hit; 1:stand
            # if pred[0] > pred[1]:
            #     return 1
            # else:
            #     return 0
            if obs[0] < 17: return 1
            return 0
            # opt method
            # result = self.csv_o[index][3]
            # if result == 'hit':
            #     return 1
            # return 0


if __name__ == '__main__':
    csv_m = pd.read_csv("policy_m.csv", encoding="utf-8")
    csv_m = np.array(csv_m)
    a = 14
    b = 5
    c = False
    d = (a - 12) * 20 + (b - 1) * 2 + c
    print(csv_m[d][3])
    asm = csv_m[d][3]
    if asm == 'hit':
        m = 1
    else:
        m = 0
    print(m)
    obs = np.array([14, 5])
    legal_actions = OrderedDict({i: None for i in range(2)})
    state = {'obs': obs, 'legal_actions': legal_actions}
    save_dir = 'model.pth'
    agent = torch.load(save_dir, map_location=device)
    result = agent.predict(state)
    # 0: hit; 1:stand
    if result[0] > result[1]:
        print("hit")

    else:
        print("stick")
    print(result)
    csv_m = pd.read_csv("policy_q.csv", encoding="utf-8")
    csv_m = np.array(csv_m)
    a = 14
    b = 5
    c = False
    d = (a - 12) * 20 + (b - 1) * 2 + c
    print(csv_m[d][3])
    asm = csv_m[d][3]
