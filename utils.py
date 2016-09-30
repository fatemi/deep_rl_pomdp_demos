"""
Utilities
"""

from hashlib import sha1
import numpy as np
from numpy import all, array, uint8
import theano
floatX = theano.config.floatX


class NumpyHashable(object):
    """ Hashable wrapper for ndarray objects.

        Instances of ndarray are not hashable, meaning they cannot be added to
        sets, nor used as keys in dictionaries. This is by design - ndarray
        objects are mutable, and therefore cannot reliably implement the
        __hash__() method.

        The hashable class allows a way around this limitation. It implements
        the required methods for hashable objects in terms of an encapsulated
        ndarray object. This can be either a copied instance (which is safer)
        or the original object (which requires the user to be careful enough
        not to modify it).
    """
    def __init__(self, wrapped, tight=False):
        """ Creates a new hashable object encapsulating an ndarray.

            wrapped
                The wrapped ndarray.

            tight
                Optional. If True, a copy of the input ndaray is created.
                Defaults to False.
        """
        self.__tight = tight
        self.__wrapped = array(wrapped) if tight else wrapped
        self.__hash = int(sha1(wrapped.view(uint8)).hexdigest(), 16)

    def __eq__(self, other):
        return all(self.__wrapped == other.__wrapped)

    def __hash__(self):
        return self.__hash

    def unwrap(self):
        r'''Returns the encapsulated ndarray.

            If the wrapper is "tight", a copy of the encapsulated ndarray is
            returned. Otherwise, the encapsulated ndarray itself is returned.
        '''
        if self.__tight:
            return array(self.__wrapped)

        return self.__wrapped


def colour_str(str_in, c_code):
    return '\033[%dm%s' % (c_code, str_in) + '\033[0m'


class Font:
   purple = '\033[95m'
   cyan = '\033[96m'
   darkcyan = '\033[36m'
   blue = '\033[94m'
   green = '\033[92m'
   yellow = '\033[93m'
   red = '\033[91m'
   bgblue = '\033[44m'
   bold = '\033[1m'
   underline = '\033[4m'
   end = '\033[0m'


class Transition(object):
    def __init__(self, current_state, action, reward, next_state, term):
        assert current_state.size == next_state.size, 's and s2 are not of the same size'
        self.__dict__.update(locals())

    def __str__(self):
        s = 'transition object:\n  current state: {0}\n  next state: {1}\n  action: {2}\n  reward: {3}'.format(
            self.current_state, self.action, self.reward, self.next_state, self.term)
        return s

    def __repr__(self):
        return 'transition_object'

    def __eq__(self, other):
        if isinstance(other, self.__class__) and \
                    self.current_state == other.current_state and \
                    self.action == other.action and \
                    self.reward == other.reward and \
                    self.next_state == other.next_state and \
                    self.term == other.term:
            return True
        else:
            return False

    def __hash__(self):
        return hash((self.current_state, self.action, self.reward, self.next_state, self.term))


class TransitionTable(object):
    def __init__(self, max_size=100):
        #self.table = deque(maxlen=max_size)
        self.table = list()
        self.max_size = max_size

    @property
    def size(self):
        return len(self.table)

    def _init_sarst(self, number, state_size):
        s = np.zeros((number, state_size), dtype=self.table[0].current_state.dtype)
        s2 = np.zeros((number, state_size), dtype=self.table[0].current_state.dtype)
        action_indicator = self.table[0].action
        if isinstance(action_indicator, int):
            a = np.zeros(number, dtype='int32')
        else:
            a = np.zeros((number, action_indicator.size), dtype=action_indicator.dtype)
        r = np.zeros(number, dtype='float32')
        term = np.zeros(number, dtype='int32')
        return s, a, r, s2, term

    def sample(self, num=1):
        if self.size == 0:
            logging.error('cannot sample from empty transition table')
        elif num <= self.size:
            state_size = self.table[0].current_state.size
            samples = np.random.choice(np.arange(len(self.table)), num)
            samples = [self.table[i] for i in samples]
            s, a, r, s2, term = self._init_sarst(number=num, state_size=state_size)
            for i, sample in enumerate(samples):
                s[i, 0:] = sample.current_state
                a[i] = sample.action
                r[i] = sample.reward
                term[i] = sample.term
                s2[i, 0:] = sample.next_state
            return s, a, r, s2, term
        elif num > self.size:
            logging.error('transition table has only {0} elements; {1} requested'.format(self.size, num))

    def shuffled_partition(self, partition_size, is_full_batch=True):
        """
            Returns list of minibatches (lists) of randomized transitions [s, a, r, s2, term].
            Each list [s, a, ...] should be used as a minibatch for offline batch-learning.
        """
        if partition_size > self.size:
            logging.error('transition table has only {0} elements; {1} requested'.format(self.size, partition_size))
        else:
            nb_partitions = self.size // partition_size + 1
            indices = np.arange(self.size)
            np.random.shuffle(indices)
            shuffled_table = [self.table[k] for k in indices]
            partitioned = [shuffled_table[k*partition_size: (k+1)*partition_size] for k in range(nb_partitions)]
            state_size = self.table[0].current_state.size
            data_list = []
            s, a, r, s2, term = self._init_sarst(number=partition_size, state_size=state_size)
            for samples in partitioned[:-1]:  # last item may have size of less than partition_size
                for i, sample in enumerate(samples):
                    s[i, 0:] = sample.current_state
                    a[i] = sample.action
                    r[i] = sample.reward
                    term[i] = sample.term
                    s2[i, 0:] = sample.next_state
                data_list.append([s, a, r, s2, term])
            # last item
            last_item_size = len(partitioned[-1])
            if last_item_size == partition_size:
                is_full_batch = False
            if not is_full_batch:
                s, a, r, s2, term = self._init_sarst(number=last_item_size, state_size=state_size)
                for i, sample in enumerate(partitioned[-1]):
                    s[i, 0:] = sample.current_state
                    a[i] = sample.action
                    r[i] = sample.reward
                    term[i] = sample.term
                    s2[i, 0:] = sample.next_state
                data_list.append([s, a, r, s2, term])
            return data_list

    def add(self, transition):
        if isinstance(transition, Transition):
            if self.size < self.max_size:
                self.table.append(transition)
            elif self.size == self.max_size:
                del self.table[0]
                self.table.append(transition)
        else:
            logging.error('unrecognized transition: must be Transition object.')

    def reset(self):
        self.table = list()