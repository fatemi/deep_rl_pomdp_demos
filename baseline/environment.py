import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class Catch(object):
    def __init__(self, grid_size=10, length=1):
        self.grid_size = grid_size
        self.length = length
        self.play = 0
        self.state = None
        self.viewer = plt.figure()
        self.reset()

    def _update_state(self, action):
        """
        Input: action and states
        Output: new states and reward
        """
        state = self.state
        if action == 0:  # left
            action = -1
        elif action == 1:  # stay
            action = 0
        else:
            action = 1  # right
        f0, f1, basket = state[0]
        new_basket = min(max(1, basket + action), self.grid_size-1)
        f0 += 1
        out = np.asarray([f0, f1, new_basket])
        out = out[np.newaxis]
        assert len(out.shape) == 2
        self.state = out

    def _draw_state(self):
        im_size = (self.grid_size,)*2
        state = self.state[0]
        canvas = np.zeros(im_size)
        canvas[state[0], state[1]] = 1  # draw fruit
        canvas[-1, state[2]-1:state[2] + 2] = 1  # draw basket
        return canvas

    def _get_reward(self):
        fruit_row, fruit_col, basket = self.state[0]
        if fruit_row == self.grid_size-1:
            if abs(fruit_col - basket) <= 1:
                return 1
            else:
                return -1
        else:
            return 0

    def _is_over(self):
        if self.state[0, 0] == self.grid_size-1:
            self._reset_new_drop()
            self.play += 1
        if self.play >= self.length:
            return True
        else:
            return False

    def get_observations(self):
        canvas = self._draw_state()
        return canvas.reshape((1, -1))

    def step(self, action):
        if self._is_over():
            logger.warning('Calling step on a finished episode.')
            return self.get_observations(), None, True
        self._update_state(action)
        return self.get_observations(), self._get_reward(), self._is_over()

    def _reset_new_drop(self):
        self.state[0, 0] = 0
        self.viewer.clear()

    def reset(self):
        n = np.random.randint(0, self.grid_size-1, size=1)
        m = np.random.randint(1, self.grid_size-2, size=1)
        self.state = np.asarray([0, n, m])[np.newaxis]
        self.viewer.clear()
        self.play = 0

    def render(self, lag=0.01):
        plt.imshow(self.get_observations().reshape((self.grid_size,) * 2), interpolation='none', cmap='gray')
        plt.pause(lag)


if __name__ == '__main__':
    grid_size = 10
    length = 4
    env = Catch(grid_size, length)
    env.reset()
    env.render()
    for game in range(10):
        game_over = False
        env.reset()
        while not game_over:
            action = np.random.randint(0, 3)
            state, reward, game_over = env.step(action)
            print(' | r: ', reward, ' | game_over: ', game_over, ' |')
            env.render()
