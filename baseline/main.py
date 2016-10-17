import pickle
import random
import numpy as np
import yaml
from baseline.lib import Experiment, BiasedEpsilonGreedyExplorer, Agent, DQNLearner, QNetwork
from baseline.environment import Catch
from utils import Font
import click

np.set_printoptions(suppress=True, linewidth=200, precision=2)
np.random.seed(seed=123)

floatX = 'float32'
epsilon = .1  # exploration
gamma = .999
num_actions = 3  # [move_left, stay, move_right]
epoch = 4000
max_memory = 500
hidden_size = 100
batch_size = 50
grid_size = 10
length = 1
update_freq = 10
ddqn = True
state_dim = grid_size ** 2


def train():
    actor = QNetwork(state_dim=state_dim, num_actions=num_actions, hidden_size=hidden_size, no_network=False)
    learner = DQNLearner(replay_max_size=max_memory, gamma=gamma, minibatch_size=batch_size, nb_actions=num_actions,
                         update_freq=update_freq, rescale_reward=False, reward_divider=None, ddqn=ddqn)
    learner.explorer = BiasedEpsilonGreedyExplorer(epsilon=epsilon, decay=1.)
    agent = Agent(actor, learner)
    env = Catch(grid_size, length)
    expt = Experiment(env=env, agent=agent)
    env.reset()
    rewards_list = []

    for ex in range(1):
        print('\n')
        print(Font.bold + Font.red + '>>>>> Experiment ', ex, ' >>>>>' + Font.end)
        rewards = expt.do_episodes(epoch)
        rewards_list.append(rewards)

    actor.dump_network()

    with open('rewards_output.pkl', 'wb') as f:
        pickle.dump(rewards_list, f)


def test():
    with open('dqn_controller.pkl', 'rb') as f:
        actor = pickle.load(f)
    actor.load_network(network_file_path='q_network.json', weights_file_path='q_network_weights.h5', target=False)
    env = Catch(grid_size, length)
    env.reset()
    state = env.get_observations()
    env.render()
    for game in range(50):
        game_over = False
        env.reset()
        while not game_over:
            action = actor.get_max_action(state, target=False)[0]
            state, reward, game_over = env.step(action)
            print(' | r: ', reward, ' | game_over: ', game_over, ' |')
            env.render()


@click.command()
@click.option('--evaluate/--no-evaluate', default=False, help='Testing the agent.')
def main(evaluate):
    if not evaluate:
        train()
    else:
        test()

if __name__ == '__main__':
    main()