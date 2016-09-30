import numpy as np
import sys
import os
import pickle
from utils import Transition, TransitionTable, Font
import logging
from copy import deepcopy
import yaml
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta
from keras.layers.core import Dense, Dropout, Activation
from keras.models import model_from_json, Model
from keras.layers import Input, merge

logger = logging.getLogger(__name__)
intX = 'int32'
floatX = 'float32'


class POMDPEnv(object):
    def __init__(self, confusion_dim, num_actual_states, num_actions, good_terminal_states=9, bad_terminal_states=8,
                 max_steps=1000):
        self.model = MDPUserModel(confusion_dim, num_actual_states, num_actions)
        self.state_buffer = self.model.id2state(0)  # current state of the environment (use self.reset)
        self.turn = 0
        if type(good_terminal_states) is int:
            self.good_terminals = [good_terminal_states]
        else:
            self.good_terminals = good_terminal_states
        if type(bad_terminal_states) is int:
            self.bad_terminals = [bad_terminal_states]
        else:
            self.bad_terminals = bad_terminal_states
        self.max_steps = max_steps

    def get_observations(self):
        return self.state_buffer

    def step(self, action):
        self.state_buffer = self.model.transition(self.state_buffer, action)
        self.turn += 1
        return self.state_buffer, self.get_reward(), self.is_done(), {}  # no additional information

    def reset(self, init_state=0):
        self.state_buffer = self.model.id2state(init_state)
        self.turn = 0

    def is_done(self):
        obs_id = self.model.state2id(self.state_buffer)
        if obs_id == self.good_terminals or obs_id == self.bad_terminals or self.turn >= self.max_steps:
            return True
        else:
            return False

    def get_reward(self):
        state = self.model.state2id(self.state_buffer)
        if state in self.good_terminals:
            r = 30.
        elif state in self.bad_terminals:
            r = -30
        else:
            r = -1.
        return r


class MDPUserModel(object):
    def __init__(self, confusion_dim, num_actual_states, num_actions):
        self.num_states = num_actual_states
        self.num_actions = num_actions
        self.confusion_dim = confusion_dim
        self.transition_table = np.array([
            [0, 0, 1],
            [0, 1, 2],
            [0, 2, 3],
            [1, 1, 4],
            [1, 2, 5],
            [2, 0, 4],
            [2, 2, 6],
            [3, 0, 5],
            [3, 1, 6],
            [4, 2, 7],
            [5, 1, 7],
            [6, 0, 7],
            [0, 3, 8],
            [1, 3, 8],
            [2, 3, 8],
            [3, 3, 8],
            [4, 3, 8],
            [5, 3, 8],
            [6, 3, 8],
            [7, 3, 9]], dtype='int32')

    def transition(self, state, action):
        state_id = self.state2id(state)
        assert state_id < self.num_states and action < self.num_actions
        next_state_id = state_id
        for t in self.transition_table:
            if t[0] == state_id and t[1] == action:
                next_state_id = t[2]
                break
        return self.id2state(next_state_id)

    def id2state(self, s_id):
        s = np.zeros(self.num_states + self.confusion_dim, dtype='float32')
        #s[: self.confusion_dim] = numpy.random.randint(0, 2, self.confusion_dim)
        s[: self.confusion_dim] = np.random.uniform(-.5, .5, size=self.confusion_dim)
        s[self.confusion_dim + s_id] = 1.
        if not hasattr(self, "randproj"):
            shape = self.num_states + self.confusion_dim
            shape = (shape, shape)
            self.randproj = np.random.uniform(-1, 1, size=shape)
            self.invrandproj = np.linalg.inv(self.randproj)
            np.save('confusion.npz', (self.randproj, self.invrandproj))
        s = np.dot(self.randproj, s)
        return s

    def state2id(self, s):
        s = np.dot(self.invrandproj, s)
        s1 = s[self.confusion_dim:]
        return np.argmax(s1)

    def write_mdp_to_dot(self, file_path='mdp.dot', overwrite=False,
                         init_state=0, good_terminals=9, bad_terminals=8):
        # To save DOT files as image files use for example: $ dot -T png -O mdp.dot
        if type(good_terminals) is int:
            good_terminals = [good_terminals]
        if type(bad_terminals) is int:
            bad_terminals = [bad_terminals]
        if not os.path.isfile(file_path) or overwrite:
            with open(file_path, 'w') as writer:
                writer.write('digraph MDP {\n')
                for tr in self.transition_table:
                    writer.write(str(tr[0]) + ' -> ' + str(tr[2]) +
                                 ' [label="a:' + str(tr[1]) + ' ; p:' + str(tr[3]) + '"];\n')
                writer.write(str(init_state) + " [shape=diamond,color=lightblue,style=filled]\n")
                for node in good_terminals:
                    writer.write(str(node) + " [shape=box,color=green,style=filled]\n")
                for node in bad_terminals:
                    writer.write(str(node) + " [shape=box,color=red,style=filled]\n")
                writer.write('}')
        else:
            logger.warning('File "{0}" exists. Call with `overwrite=True` to permit overwrite.'.format(file_path))


class MDPExperiment(object):
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.last_state = None
        self.step_id = 0

    def do_episodes(self, number=1, is_learning=True, random_init=True):
        all_rewards = []

        for num in range(number):
            sys.stdout.flush()
            print('='*30)
            print(Font.darkcyan + Font.bold + '::Episode::  ' + Font.end + str(num))
            print('\n')
            self.agent.new_episode()
            rewards = []
            self.step_id = 0
            if random_init:
                init_state = np.random.randint(0, self.env.model.num_states)
            else:
                init_state = 0
            self.env.reset(init_state)
            self.agent.reset()
            self.last_state = self.env.get_observations()
            while not self.env.is_done():
                reward = self._step()
                rewards.append(reward)
                if is_learning and self.agent.learner.transitions.size >= self.agent.learner.minibatch_size:
                    self.agent.learn()
            all_rewards.append(rewards)
        return all_rewards

    def evaluate(self, number=10):
        all_rewards = []
        print('\n')
        print(Font.yellow + Font.bold + 'Evaluation ...' + Font.end)

        for e_num in range(number):
            print('Evaluate episode: ', str(e_num))
            self.agent.new_episode()
            rewards = []
            self.step_id = 0
            self.env.reset()
            self.agent.reset()
            self.last_state = self.env.get_observations()
            while not self.env.is_done():
                reward = self._step(evaluate=True)
                rewards.append(reward)
            all_rewards.append(rewards)
        return all_rewards

    def _step(self, evaluate=False):
        self.step_id += 1
        print('last_state: ', self.last_state)
        if evaluate:
            action = self.agent.module.get_max_action(self.last_state)  # no exploration
        else:
            action = self.agent.get_action(self.last_state)
        print('action: ', action)
        self.env.step(action)
        reward = self.env.get_reward()
        print('reward: ', reward)
        new_state = self.env.get_observations()
        if not evaluate:
            tr = Transition(current_state=self.last_state.astype('float32'),
                            action=action,
                            reward=reward,
                            next_state=new_state.astype('float32'),
                            term=int(self.env.is_done()))
            self.agent.learner.transitions.add(tr)
        self.last_state = new_state
        return reward


class BiasedEpsilonGreedyExplorer(object):
    """ Discrete epsilon-greey explorer.
        At the exploration time, it selects action according to the given prior distribution.
        If no prior is provided, uniform prior is used as default (ordinary exploration).
    """
    def __init__(self, epsilon=0.05, decay=0.9999, prior=None):
        self.epsilon = epsilon
        self.decay = decay
        self.prior = prior
        self.module = None

    def __call__(self, state, action):
        """ Draws a random number between 0 and 1. If the number is less
            than epsilon, a random action is chosen. If it is equal or
            larger than epsilon, the greedy action is returned.
        """
        assert self.module
        if self.prior is None:
            self.prior = np.ones(self.module.numActions, dtype=floatX) / self.module.numActions  # equiprobable

        if np.random.binomial(1, self.epsilon):
            a = np.where(np.random.multinomial(1, self.prior) == 1)[0]
            action = a.astype('int32')
        self.epsilon *= self.decay
        return action


class Agent(object):
    def __init__(self, module, learner=None):
        self.module = module
        self.learner = learner
        if learner:
            self.learner.module = module
            self.learner.explorer.module = module

    def learn(self, episodes=1, *args, **kwargs):
        return self.learner.learnEpisodes(episodes, *args, **kwargs)

    def get_action(self, obs):
        """ Gets action for the module with the last observation and add the exploration.
        """
        action = self.module.get_max_action(obs)
        if self.learner:
            action = self.learner.explore(obs, action)
        return action

    def new_episode(self):
        """ Indicate the beginning of a new episode in the training cycle. """
        self.module.reset()
        self.learner.reset()

    def reset(self):
        self.module.reset()
        self.learner.reset()


class QNetwork(object):
    """ A network that approximates action values for continuous state /
        discrete action RL environments. To receive the maximum action
        for a given state, a forward pass is executed for all discrete
        actions, and the maximal action is returned. This network is used
        for the NFQ algorithm. """

    def __init__(self, params, optimizer=None, name=None, no_network=False):
        if not no_network:
            self.network = create_model(params, optimizer)
            self.target_network = create_model(params, optimizer)
            weight_transfer(from_model=self.network, to_model=self.target_network)
        else:
            self.network = []
            self.target_network = []
        self.numActions = params['general']['num_actions']
        self.state_dim = params['general']['state_dim']

    def get_max_action(self, states):
        """ Return the action with the maximal value for the given state(s).
            If there are more than one of such actions, one of them in random
            will be returned.
        """
        values_array = self.get_action_values(states)
        actions = []
        for values in values_array:
            action = np.where(values == max(values))[0]  # maybe more than one
            actions.append(np.random.choice(action))
        return np.array(actions, dtype=intX)

    def get_action_values(self, states):
        """ Run forward activation of the QNetwork.
            :param states: Each row of states is one state.
            :return same num of rows as states and num cols as num of actions
        """
        if states.ndim == 1:
            minibatch_size = 1
            states = states.reshape((1, -1))
        else:
            minibatch_size = states.shape[0]
        no_mask = np.ones((minibatch_size, self.numActions), dtype=floatX)
        # return self.network.predict(states)
        output = self.network.predict({'states': states, 'actions_mask': no_mask})
        return output

    def get_values(self, states, actions):
        values = self.get_action_values(states)
        q = np.zeros_like(actions, dtype=floatX)
        for i, single_a in enumerate(actions):
            q[i] = values[i, single_a]
        return q

    def getTargetMaxAction(self, states):
        """ Return the action with the maximal value for the given state.
            If there are more than one of such actions, one of them in random
            will be returned.
        """
        values_array = self.getTargetActionValues(states)
        actions = []
        for values in values_array:
            action = np.where(values == max(values))[0]  # maybe more than one
            actions.append(np.random.choice(action))
        return np.array(actions, dtype=intX)

    def getTargetActionValues(self, states):
        """ Run forward activation of the QNetwork.
            :param states: Each row of states is one state.
            :return same num of rows as states and num cols as num of actions
        """
        minibatch_size = states.shape[0]
        no_mask = np.ones((minibatch_size, self.numActions), dtype=floatX)
        # return self.target_network.predict(states)
        output = self.target_network.predict({'states': states, 'actions_mask': no_mask})
        return output

    def getTargetValues(self, states, actions):
        values = self.getTargetActionValues(states)
        q = np.zeros_like(actions, dtype=floatX)
        for i, single_a in enumerate(actions):
            q[i] = values[i, single_a]
        return q

    def _target_network_update(self):
        weight_transfer(from_model=self.network, to_model=self.target_network)

    def __getstate__(self):
        d = {k: v for k, v in self.__dict__.items()}
        del d['network']
        del d['target_network']
        return d

    def dump_network(self,
                     dqn_controller_file='dqn_controller.pkl',
                     network_file_path='q_network.json',
                     weights_file_path='q_network_weights.h5',
                     overwrite=True):
        json_string = self.network.to_json()
        with open(network_file_path, 'w') as f:
            f.write(json_string)
        self.network.save_weights(weights_file_path, overwrite=overwrite)
        with open(dqn_controller_file, 'wb') as f_c:
            pickle.dump(self, f_c)

    def load_network(self,
                     network_file_path='q_network.json',
                     weights_file_path='q_network_weights.h5',
                     target=False):
        self.network = model_from_json(open(network_file_path).read())
        self.network.load_weights(weights_file_path)
        if target:
            self.target_network = model_from_json(open(network_file_path).read())
            self._target_network_update()

    def reset(self):
        pass


class Learner(object):
    """
    Top-level class for all reinforcement learning algorithms.
    """
    module = None
    explorer = None

    def learn(self):
        """ The main method, that invokes a learning step. """
        raise NotImplementedError

    def explore(self, state, action, *args, **kwargs):
        if self.explorer is not None:
            return self.explorer(state, action, *args, **kwargs)
        else:
            logging.warning("No explorer found: no exploration could be done.")
            return action

    def learnEpisodes(self, episodes=1, *args, **kwargs):
        """ learn on the current dataset, for a number of episodes """
        for _ in range(episodes):
            self.learn(*args, **kwargs)

    def reset(self):
        pass


class DQNLearner(Learner):
    """ DQN Learner class.
    It provides the learning for a Q-network controller/interface

    Notice:
    the 'module' attribute of this learner should be QNetwork
    """
    def __init__(self, params):
        self.replay_max_size = params['learning_params']['replay_max_size']
        self.gamma = params['learning_params']['discount']
        self.minibatch_size = params['learning_params']['minibatch_size']
        self.n_actions = params['general']['num_actions']
        self.update_freq = params['learning_params']['update_freq']
        self.transitions = TransitionTable(self.replay_max_size)
        self.rescale_r = params['learning_params']['rescale_reward']
        self.r_divider = params['learning_params']['reward_divider']
        self.update_counter = 0
        self.ddqn = params['learning_params'].get('ddqn', True)

    def _get_q_target(self, a, r, s2, term):
        # q_target = r + (1-terminal) * gamma * max_a Q_target(s2, a)
        term = (1 - term).astype(floatX)

        # Compute max_a Q(s_2, a).
        # q2_max = self.module.getTargetActionValues(s2).max(axis=1)
        if self.ddqn:
            a_max = self.module.get_max_action(s2)
            q2_max = self.module.getTargetActionValues(s2)
            q2_max = np.array([q2_max[i, a_max[i]] for i in range(q2_max.shape[0])], dtype=floatX)
        else:
            q2_max = self.module.getTargetActionValues(s2).max(axis=1)

        # Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
        q2 = self.gamma * q2_max * term

        if self.rescale_r:
            r /= self.r_divider

        q_target = r + q2

        # neural network targets for states in the minibatch
        targets = np.zeros((self.minibatch_size, self.n_actions))
        for i in range(self.minibatch_size):
            targets[i, int(a[i])] = q_target[i]

        return targets, q_target, q2_max

    def _train_on_batch(self, s, a, r, s2, term):
        targets, delta, q2_max = self._get_q_target(a=a, r=r, s2=s2, term=term)
        a_mask = np.zeros((self.minibatch_size, self.n_actions), dtype=floatX)
        for i in range(self.minibatch_size):
            a_mask[i, int(a[i])] = 1.
        # objective = self.module.network.train_on_batch(s, targets)
        objective = self.module.network.train_on_batch(x={'states': s, 'actions_mask': a_mask}, y={'output': targets})
        # updating target network
        if self.update_counter == self.update_freq:
            self.module._target_network_update()
            self.update_counter = 0
        else:
            self.update_counter += 1
        return objective

    def learn(self):
        """
        Learning from one minibatch .
        """
        assert self.minibatch_size <= self.transitions.size, 'not enough data in the pool'
        # print colour_str('>>> Learner'+'-'*89, 36)
        # sampling one minibatch
        s, a, r, s2, term = self.transitions.sample(self.minibatch_size)
        objective = self._train_on_batch(s, a, r, s2, term)
        # print colour_str('<<< Learner'+'-'*89, 36)
        return objective

    def learn_offline_batch(self, nb_epochs=1, minibatch_size=128, reset=True):
        minibatches = self.transitions.shuffled_partition(minibatch_size)
        nb_minibatches = len(minibatches)
        for ep in range(nb_epochs):
            print(Font.bold + Font.yellow + 'Training epoch ' + str(ep) + Font.end)
            for bt, minibatch in enumerate(minibatches):
                print(Font.cyan + 'mini batch ' + str(bt) + ' out of ' + str(nb_minibatches) + Font.end + ' loss: ')
                s, a, r, s2, term = minibatch
                objective = self._train_on_batch(s, a, r, s2, term)
                print(objective)
        if reset:
            self.transitions = TransitionTable(self.replay_max_size)
            self.update_counter = 0

    def reset(self):
        pass


class ResourceManager(object):
    def __init__(self, config_file='config.cfg'):
        self.params = yaml.load(open(config_file, 'r'))

    def __call__(self):
        return self.params


def create_model(params, optimizer=None):
    # returns the keras nn model
    input_dim = params['general']['state_dim']
    states = Input(shape=(input_dim,), dtype=floatX, name='states')
    x = Dense(input_dim=input_dim,
              output_dim=input_dim * 2,
              init='glorot_uniform',
              activation='tanh')(states)
    x = Dropout(p=.05)(x)
    predictions = Dense(input_dim=input_dim * 2,
                        output_dim=params['general']['num_actions'],
                        init='glorot_uniform',
                        activation='tanh')(x)
    actions_mask = Input(shape=(params['general']['num_actions'],), dtype=floatX, name='actions_mask')
    output = merge(inputs=[actions_mask, predictions], mode='mul', name='output')
    model = Model(input=[states, actions_mask], output=[output])
    model.compile(loss={'output': 'mean_squared_error'}, optimizer=Adadelta())
    return model


def weight_transfer(from_model, to_model):
    to_model.set_weights(deepcopy(from_model.get_weights()))
