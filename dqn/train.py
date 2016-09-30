import pickle
import random
import numpy as np
import yaml
from lib import MDPExperiment, BiasedEpsilonGreedyExplorer, Agent, DQNLearner, QNetwork
from environment import POMDPEnv
from utils import Font

np.random.seed(seed=123)
random.seed(123)

CONFIG_FILE = 'dqn/config.cfg'
params = yaml.safe_load(open(CONFIG_FILE, 'r'))

actor = QNetwork(params=params, optimizer='adadelta')
learner = DQNLearner(params)
learner.explorer = BiasedEpsilonGreedyExplorer(epsilon=params['explorer_params']['epsilon'],
                                               decay=params['explorer_params']['decay'])
agent = Agent(actor, learner)
env = POMDPEnv(confusion_dim=params['general']['confusion_dim'],
               num_actual_states=params['general']['num_actual_state'],
               num_actions=params['general']['num_actions'],
               good_terminal_states=params['general']['good_terminal_state'],
               bad_terminal_states=params['general']['bad_terminal_state'],
               max_steps=params['general']['max_steps'])
expt = MDPExperiment(env=env, agent=agent)
env.reset()
rewards_list = []

for ex in range(params['general']['num_experiments']):
    print('\n')
    print(Font.bold + Font.red + '>>>>> Experiment ', ex, ' >>>>>' + Font.end)
    # time.sleep(.5)
    rewards = expt.do_episodes(params['general']['num_episodes'])
    rewards_list.append(rewards)

actor.dump_network()

with open('rewards_output.pkl', 'wb') as f:
    pickle.dump(rewards_list, f)
