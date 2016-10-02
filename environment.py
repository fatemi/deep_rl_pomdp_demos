import numpy as np
import os
import logging

logger = logging.getLogger(__name__)
intX = 'int32'
floatX = 'float32'


class POMDPEnv(object):
    def __init__(self, confusion_dim, num_actual_states, num_actions, good_terminal_states=9, bad_terminal_states=8,
                 max_steps=1000):
        self.model = MDPUserModel(confusion_dim, num_actual_states, num_actions, make_confusion_matrix=True)
        self.state_buffer = self.model.state2obs(0)  # current state of the environment (use self.reset)
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
        self.state_buffer = self.model.step(self.state_buffer, action)
        self.turn += 1
        return self.state_buffer, self.get_reward(), self.is_done(), {}  # no additional information

    def reset(self, init_state=0):
        self.state_buffer = self.model.state2obs(init_state)
        self.turn = 0

    def is_done(self):
        state = self.model.obs2state(self.state_buffer)
        if state in self.good_terminals or state in self.bad_terminals or self.turn >= self.max_steps:
            return True
        else:
            return False

    def get_reward(self):
        state = self.model.obs2state(self.state_buffer)
        if state in self.good_terminals:
            r = 30.
        elif state in self.bad_terminals:
            r = -30.
        else:
            r = -1.
        return r


class MDPUserModel(object):
    def __init__(self, confusion_dim, num_actual_states, num_actions, make_confusion_matrix=False):
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
        if make_confusion_matrix:
            shape = self.num_states + self.confusion_dim
            shape = (shape, shape)
            self.randproj = np.random.uniform(-.1, .1, size=shape)
            np.fill_diagonal(self.randproj, 1.0)
            self.invrandproj = np.linalg.inv(self.randproj)
            np.save('confusion.npz', (self.randproj, self.invrandproj))
        else:
            self.randproj = None
            self.invrandproj = None

    def step(self, state, action):
        state_id = self.obs2state(state)
        assert state_id < self.num_states and action < self.num_actions
        next_state_id = state_id
        for t in self.transition_table:
            if t[0] == state_id and t[1] == action:
                next_state_id = t[2]
                break
        return self.state2obs(next_state_id)

    def state2obs(self, s_id):
        s = np.zeros(self.num_states + self.confusion_dim, dtype='float32')
        # s[: self.confusion_dim] = numpy.random.randint(0, 2, self.confusion_dim)
        s[: self.confusion_dim] = np.random.uniform(-.5, .5, size=self.confusion_dim)
        s[self.confusion_dim + s_id] = 1.
        s = np.dot(self.randproj, s)
        return s

    def obs2state(self, s):
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
