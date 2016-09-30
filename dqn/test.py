import pickle
import numpy
import yaml
from environment import MDPUserModel

CONFIG_FILE = 'dqn/config.cfg'
params = yaml.safe_load(open(CONFIG_FILE, 'r'))


def main():
    r, ir = numpy.load('confusion.npz.npy')
    m = MDPUserModel(confusion_dim=params['general']['confusion_dim'],
                     num_actual_states=params['general']['num_actual_state'],
                     num_actions=params['general']['num_actions'],
                     make_confusion_matrix=False)
    m.randproj = r
    m.invrandproj = ir
    with open('dqn_controller.pkl', 'rb') as f:
        actor = pickle.load(f)

    actor.load_network(network_file_path='q_network.json', weights_file_path='q_network_weights.h5', target=False)

    for s in range(10):
        state = m.state2obs(s)
        print('state: ', s, ' | action: ', str(actor.get_max_action(state, target=False)[0]))

if __name__ == '__main__':
    main()
