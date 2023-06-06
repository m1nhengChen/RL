import numpy as np
import torch
import os
import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import get_device, set_seed, tournament, reorganize, Logger
from rlcard.agents import DQNAgent_v2
from collections import OrderedDict
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
device = torch.device('cuda:{}'.format(device_ids[0]))


def load_model(model_path, env=None, position=None, device=None):
    if os.path.isfile(model_path):  # Torch model
        import torch
        agent = torch.load(model_path, map_location=device)
        agent.set_device(device)
    elif os.path.isdir(model_path):  # CFR model
        from rlcard.agents import CFRAgent
        agent = CFRAgent(env, model_path)
        agent.load()
    elif model_path == 'random':  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)
    else:  # A model in the model zoo
        from rlcard import models
        agent = models.load(model_path).agents[position]

    return agent


def test():
    obs = np.array([21, 3])
    legal_actions = OrderedDict({i: None for i in range(2)})
    state = {'obs': obs, 'legal_actions': legal_actions}
    save_dir = 'experiments/blackjack_dqnv2_result_3/model.pth'
    agent = torch.load(save_dir, map_location=device)
    result = agent.predict(state)
    # 0: hit; 1:stand
    if result[0]>result[1]:
        print("hit")
    else:
        print("stick")
    print(result)


if __name__ == '__main__':
    test()
