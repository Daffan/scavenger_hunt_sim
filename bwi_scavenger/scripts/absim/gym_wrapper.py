import gym
import numpy as np
gym.logger.set_level(40)

import absim.agent as agent
from absim.hunt import parse_world
from absim.generate import generate


class AbstractSim(gym.Env):

    def __init__(self, world_file = 'absim/bwi.dat'):
        super(AbstractSim, self).__init__()
        self.world, self.hunt, self.start_loc = parse_world(world_file)
        self.agent = agent.Agent(self.world, self.hunt, self.world.node_id(self.start_loc))
        self.world.populate()
        self.agent.setup()

        self.action_space = gym.spaces.Discrete(len(self.world.graph.nodes))
        self.observation_space = gym.spaces.Box(np.array([-1]*(len(self.world.graph.nodes))),
                                                np.array([1]*(len(self.world.graph.nodes))),
                                                dtype=np.float32)

    def reset(self):
        self.world.populate()
        self.agent.setup()
        self.agent.objs_at_loc = self.world.objs_at(self.agent.loc)
        self.agent.run()
        self.step_count = 0
        return self.get_obs()

    def get_obs(self):
        obs = []
        for node in self.agent.world.graph.nodes:
            obs.append(self.agent.arrangement_space.prob_any_obj(node))
        obs = [0 if self.agent.visited_count[i] > 0 else o for i, o in enumerate(obs)]
        obs[self.agent.loc] = -1
        return obs

    def step(self, action):
        self.step_count += 1
        node = self.agent.world.graph.nodes[action]
        if node != self.agent.loc:
            rew = -self.agent.world.graph.cost(self.agent.loc, node)

            self.agent.go(node)
            self.agent.objs_at_loc = self.world.objs_at(self.agent.loc)
            self.agent.run()
        else:
            rew = -500

        done = self.agent.done() if self.step_count < 100 else True
        obs = self.get_obs()

        return obs, rew, done, {}

wrapper_args = {
    "nodes_range": [8, 8],
    "cost_range": [50, 500],
    "objects_range": [4, 4],
    "occurrences_range": [1, 4]
}

class RandomMap(gym.Wrapper):
    '''A wrapper of the original gym env that generate a new random hunt problem
    at the beginning of epsiode and append the cost map to the observation state
    (assume agent knows the map)
    args:
        some parameters when generate the new hunt problem
    '''

    def __init__(self, env, wrapper_args, world_file = 'hunts/gym.dat'):

        super(RandomMap, self).__init__(env)
        self.generator = lambda: generate(world_file, wrapper_args["nodes_range"],
                                          wrapper_args["cost_range"],
                                          wrapper_args["objects_range"],
                                          wrapper_args["occurrences_range"])
        self.world_file = world_file
        self.cost_range = wrapper_args["cost_range"]
        self.objects_range = wrapper_args["objects_range"]
        self.occurrences_range = wrapper_args["occurrences_range"]
        self.action_space = gym.spaces.Discrete(len(self.world.graph.nodes))
        self.observation_space = gym.spaces.Box(np.array([-1]*(len(self.world.graph.nodes)*2)),
                                                np.array([1]*(len(self.world.graph.nodes)*2)), dtype=np.float32)
    def get_cost_map(self):
        return [self.env.agent.world.graph.cost(self.env.agent.loc, node)/500.0 \
                if node != self.env.agent.loc else 0 \
                for node in self.agent.world.graph.nodes]

    def reset(self):

        self.generator()
        self.env.world, self.env.hunt, self.env.start_loc = parse_world(self.world_file)
        self.env.agent = agent.Agent(self.env.world, self.env.hunt,
                                    self.env.world.node_id(self.env.start_loc))
        return self.env.reset() + self.get_cost_map()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return obs + self.get_cost_map(), rew, done, info

class RandomMapMultiRun(gym.Wrapper):
    '''A wrapper of the original gym env that generate a new random hunt problem
    at the beginning of epsiode and append the cost map to the observation state
    (assume agent knows the map)
    args:
        some parameters when generate the new hunt problem
    '''
    def __init__(self, env, wrapper_args = wrapper_args, world_file = 'hunts/gym.dat'):
        super(RandomMapMultiRun, self).__init__(env)
        self.generator = lambda: generate(world_file, wrapper_args["nodes_range"], wrapper_args["cost_range"],
                                            wrapper_args["objects_range"], wrapper_args["occurrences_range"])

        self.world_file = world_file
        self.cost_range = wrapper_args["cost_range"]
        self.objects_range = wrapper_args["objects_range"]
        self.occurrences_range = wrapper_args["occurrences_range"]
        self.action_space = gym.spaces.Discrete(len(self.world.graph.nodes))
        self.observation_space = gym.spaces.Box(np.array([-1]*(len(self.world.graph.nodes)*2)),
                                                np.array([1]*(len(self.world.graph.nodes)*2)), dtype=np.float32)
        self.runs_count = wrapper_args["env_runs"]
        self.env_runs = wrapper_args["env_runs"]
    def get_cost_map(self):
        return [self.env.agent.world.graph.cost(self.env.agent.loc, node)/500.0 \
                if node != self.env.agent.loc else 0 \
                for node in self.agent.world.graph.nodes]

    def reset(self):
        if self.runs_count >= self.env_runs:
            self.generator()
            self.env.world, self.env.hunt, self.env.start_loc = parse_world(self.world_file)
            self.env.agent = agent.Agent(self.env.world, self.env.hunt,
                                        self.env.world.node_id(self.env.start_loc))
            self.runs_count = 0
        self.runs_count += 1
        return self.env.reset() + self.get_cost_map()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return obs + self.get_cost_map(), rew, done, info

class RandomMapFullMap(gym.Wrapper):
    '''A wrapper of the original gym env that generate a new random hunt problem
    at the beginning of epsiode and append the cost map to the observation state
    (assume agent knows the map)
    args:
        some parameters when generate the new hunt problem
    '''

    def __init__(self, env, wrapper_args = wrapper_args, world_file = 'hunts/gym.dat'):

        super(RandomMapFullMap, self).__init__(env)
        self.generator = lambda: generate(world_file, wrapper_args["nodes_range"], wrapper_args["cost_range"],
                                            wrapper_args["objects_range"], wrapper_args["occurrences_range"])
        self.world_file = world_file
        self.cost_range = wrapper_args["cost_range"]
        self.objects_range = wrapper_args["objects_range"]
        self.occurrences_range = wrapper_args["occurrences_range"] 
        self.action_space = gym.spaces.Discrete(len(self.world.graph.nodes))
        num_node = len(self.world.graph.nodes)
        self.cost_map_full = self.get_cost_map_full()
        self.observation_space = gym.spaces.Box(np.array([-1]*(num_node*(num_node+1))),\
                                                np.array([1]*(num_node*(num_node+1))), dtype=np.float32)

    def get_cost_map_single_node(self, node):
        return [self.env.agent.world.graph.cost(n, node)/500.0 \
                if n != node else 0 for n in self.agent.world.graph.nodes]

    def get_cost_map_full(self):
        m = []
        for n in self.agent.world.graph.nodes:
            m += self.get_cost_map_single_node(n)
        return m

    def reset(self):

        self.generator()
        self.env.world, self.env.hunt, self.env.start_loc = parse_world(self.world_file)
        self.env.agent = agent.Agent(self.env.world, self.env.hunt,
                                    self.env.world.node_id(self.env.start_loc))
        return self.env.reset() + self.cost_map_full

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return obs + self.cost_map_full, rew, done, info

wrapper_dict = {
    "RandomMap": RandomMap,
    "RandomMapFullMap": RandomMapFullMap,
    "RandomMapMultiRun": RandomMapMultiRun
}
