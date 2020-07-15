import absim.agent as agent
import absim.world as world

import torch
import numpy as np
from tianshou_policies import *

policy_net = 'Mlp64x2'
state_dict_path = './results/DQN_MlpPolicy_2020_07_15_00_35/dqn.pth'

class DQNMapAgent(agent.Agent):
    """The probability-greedy agent visits the location with the highest
    probability of finding any object. If two locations are equally
    favorable, the closer of the two is chosen.
    """
    def setup(self):
        super().setup()
        state_shape = len(self.world.graph.nodes)*2
        action_shape = len(self.world.graph.nodes)

        state_dict = {}
        state_dict_raw = torch.load(state_dict_path)
        for key in state_dict_raw.keys():
            if key.split('.')[0] == 'model':
                state_dict[key[6:]] = state_dict_raw[key]
        self.net = policies_dic[policy_net](state_shape, action_shape)
        self.net.load_state_dict(state_dict)

    def get_cost_map(self):
        return [self.world.graph.cost(self.loc, node)/500.0 \
                if node != self.loc else 0 \
                for node in self.world.graph.nodes]

    def get_obs(self):
        obs = []
        for node in self.world.graph.nodes:
            obs.append(self.arrangement_space.prob_any_obj(node))
        obs = [0 if self.visited_count[i] > 0 else o for i, o in enumerate(obs)]
        obs[self.loc] = -1
        obs += self.get_cost_map()
        return torch.tensor([obs])

    def choose_next_loc(self):
        actions = np.array(self.net(self.get_obs())[0].detach().cpu())
        action = np.argmax(actions.reshape(-1))
        return action

    def run(self):
        # Collect objects at current location and update the occurrence space
        super().run()

        if self.done():
            return

        # Visit next location in the active path
        self.go(self.choose_next_loc())
