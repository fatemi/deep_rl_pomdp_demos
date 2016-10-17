import numpy as np
import sys
import os
import pickle
from utils import Transition, TransitionTable, Font
import logging
from copy import deepcopy
import yaml
from keras.optimizers import Adadelta
from keras.layers.core import Dense, Dropout
from keras.models import model_from_json, Model
from keras.layers import Input, merge

logger = logging.getLogger(__name__)
intX = 'int32'
floatX = 'float32'


class Experiment(object):
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.last_state = None
        self.step_id = 0

    def do_episodes(self, number=1, is_learning=True, random_init=True):
        all_rewards = []

        for num in range(number):
            print('='*30)
            print(Font.darkcyan + Font.bold + '::Episode::  ' + Font.end + str(num))
            self.agent.new_episode()
            rewards = []
            self.step_id = 0
            self.env.reset()
            self.agent.reset()
            self.last_state = self.env.get_observations()
            term = False
            while not term:
                reward, term = self._step()
                rewards.append(reward)
                if is_learning and self.agent.learner.transitions.size >= self.agent.learner.minibatch_size:
                    loss = self.agent.learn()
            if is_learning and self.agent.learner.transitions.size >= self.agent.learner.minibatch_size:
                print('Loss: {0:2.7f}'.format(float(loss)))
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
                reward, term = self._step(evaluate=True)
                rewards.append(reward)
            all_rewards.append(rewards)
        return all_rewards

    def _step(self, evaluate=False):
        self.step_id += 1
        if evaluate:
            action = self.agent.module.get_max_action(self.last_state)  # no exploration
        else:
            action = self.agent.get_action(self.last_state)
        new_state, reward, term = self.env.step(action)
        if not evaluate:
            tr = Transition(current_state=self.last_state.astype('float32'),
                            action=action,
                            reward=reward,
                            next_state=new_state.astype('float32'),
                            term=int(term))
            self.agent.learner.transitions.add(tr)
        self.last_state = new_state
        return reward, term


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
        assert self.module
        if self.prior is None:
            self.prior = np.ones(self.module.numActions, dtype=floatX) / self.module.numActions  # equiprobable

        if np.random.binomial(1, self.epsilon):
            a = np.where(np.random.multinomial(1, self.prior) == 1)[0]
            action = a.astype('int32')
        return action

    def anneal(self):
        self.epsilon *= self.decay


class Agent(object):
    def __init__(self, actor, learner=None):
        self.actor = actor
        self.learner = learner
        if learner:
            self.learner.module = actor
            self.learner.explorer.module = actor

    def learn(self, episodes=1, *args, **kwargs):
        return self.learner.learn_episodes(episodes, *args, **kwargs)

    def get_action(self, obs):
        """ Gets action for the module with the last observation and add the exploration. """
        action = self.actor.get_max_action(obs, target=False)
        if self.learner:
            action = self.learner.explore(obs, action)
        return action

    def new_episode(self):
        """ Indicate the beginning of a new episode in the training cycle. """
        self.actor.reset()
        self.learner.reset()

    def reset(self):
        self.actor.reset()
        self.learner.reset()


class QNetwork(object):
    def __init__(self, state_dim, num_actions, hidden_size, no_network):
        if not no_network:
            self.network = self.build(state_dim, num_actions, hidden_size)
            self.target_network = self.build(state_dim, num_actions, hidden_size)
            self.weight_transfer(from_model=self.network, to_model=self.target_network)
        else:
            self.network = []
            self.target_network = []
        self.numActions = num_actions
        self.state_dim = state_dim

    def build(self, state_dim, num_actions, hidden_size):
        states = Input(shape=(state_dim,), dtype=floatX, name='states')
        x = Dense(input_dim=state_dim, output_dim=hidden_size, init='glorot_uniform', activation='relu')(states)
        x = Dense(input_dim=hidden_size, output_dim=hidden_size, init='glorot_uniform', activation='relu')(x)
        x = Dense(input_dim=hidden_size, output_dim=num_actions, init='glorot_uniform', activation='relu')(x)
        actions_mask = Input(shape=(num_actions,), dtype=floatX, name='actions_mask')
        output = merge(inputs=[actions_mask, x], mode='mul', name='output')
        model = Model(input=[states, actions_mask], output=[output])
        model.compile(loss={'output': 'mean_squared_error'}, optimizer=Adadelta())
        return model

    def get_max_action(self, states, target):
        """ Return the action with the maximal value for the given state(s).
            If there are more than one of such actions, one of them in random
            will be returned.
        """
        values_array = self.get_action_values(states, target)
        actions = []
        for values in values_array:
            action = np.where(values == max(values))[0]  # maybe more than one
            actions.append(np.random.choice(action))
        return np.array(actions, dtype=intX)

    def get_action_values(self, states, target):
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
        if target:
            output = self.target_network.predict({'states': states, 'actions_mask': no_mask})
        else:
            output = self.network.predict({'states': states, 'actions_mask': no_mask})
        return output

    def get_values(self, states, actions, target):
        values = self.get_action_values(states, target)
        q = np.zeros_like(actions, dtype=floatX)
        for i, single_a in enumerate(actions):
            q[i] = values[i, single_a]
        return q

    def target_network_update(self):
        self.weight_transfer(from_model=self.network, to_model=self.target_network)

    def __getstate__(self):
        d = {k: v for k, v in self.__dict__.items()}
        del d['network']
        del d['target_network']
        return d

    @staticmethod
    def weight_transfer(from_model, to_model):
        to_model.set_weights(deepcopy(from_model.get_weights()))

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
            self.target_network_update()

    def reset(self):
        pass


class Learner(object):
    """ Top-level class for all reinforcement learning algorithms. """
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

    def learn_episodes(self, episodes=1, *args, **kwargs):
        """ learn on the current replay pool, for given num episodes """
        for _ in range(episodes):
            loss = self.learn(*args, **kwargs)
        return loss

    def reset(self):
        pass


class DQNLearner(Learner):
    def __init__(self, replay_max_size, gamma, minibatch_size, nb_actions, update_freq, rescale_reward,
                 reward_divider, ddqn):
        self.replay_max_size = replay_max_size
        self.gamma = gamma
        self.minibatch_size = minibatch_size
        self.nb_actions = nb_actions
        self.update_freq = update_freq
        self.rescale_r = rescale_reward
        self.r_divider = reward_divider
        self.ddqn = ddqn
        self.update_counter = 0
        self.transitions = TransitionTable(self.replay_max_size)

    def _get_q_target(self, a, r, s2, term):
        # q_target = r + (1-terminal) * gamma * max_a Q_target(s2, a)
        term = (1 - term).astype(floatX)
        # Compute max_a Q(s_2, a).
        if self.ddqn:
            a_max = self.module.get_max_action(s2, target=False)
            q2_max = self.module.get_action_values(s2, target=True)
            q2_max = np.array([q2_max[i, a_max[i]] for i in range(q2_max.shape[0])], dtype=floatX)
        else:
            q2_max = self.module.get_action_values(s2, target=True).max(axis=1)
        # Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
        q2 = self.gamma * q2_max * term
        if self.rescale_r:
            r /= self.r_divider
        q_target = r + q2
        targets = np.zeros((self.minibatch_size, self.nb_actions))
        for i in range(self.minibatch_size):
            targets[i, int(a[i])] = q_target[i]
        return targets, q_target, q2_max

    def _train_on_batch(self, s, a, r, s2, term):
        targets, delta, q2_max = self._get_q_target(a=a, r=r, s2=s2, term=term)
        a_mask = np.zeros((self.minibatch_size, self.nb_actions), dtype=floatX)
        for i in range(self.minibatch_size):
            a_mask[i, int(a[i])] = 1.
        objective = self.module.network.train_on_batch(x={'states': s, 'actions_mask': a_mask}, y={'output': targets})
        # updating target network
        if self.update_counter == self.update_freq:
            self.module.target_network_update()
            self.update_counter = 0
        else:
            self.update_counter += 1
        return objective

    def learn(self):
        """
        Learning from one minibatch .
        """
        assert self.minibatch_size <= self.transitions.size, 'not enough data in the pool'
        # sampling one minibatch
        s, a, r, s2, term = self.transitions.sample(self.minibatch_size)
        return self._train_on_batch(s, a, r, s2, term)

    def reset(self):
        pass
