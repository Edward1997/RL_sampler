import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset

from karateclub.node_embedding.attributed import TENE
from GAT.model import GAT



class myenv():
    def __init__(self, GNN_model, embedding_model, dataset):
        self.GNN = GNN_model
        self.embedding_model = embedding_model
        self.dataset = dataset
        self.picked_nodes = list(set(np.random.choice(dataset.nodes().numpy(), 1)))
        self.graph = dgl.node_subgraph(self.dataset, self.picked_nodes)
        self.done = False

    def reset(self):
        self.picked_nodes = list(set(np.random.choice(dataset.nodes().numpy(), 1)))
        self.graph = dgl.node_subgraph(self.dataset, self.picked_nodes)

    def state(self):
        features = torch.stack([x for i, x in enumerate(self.dataset.ndata['feat']) if i in self.picked_nodes])
        features = np.array(features)
        graph = dgl.add_self_loop(self.graph)
        graph = dgl.to_networkx(graph).to_undirected()
        self.embedding_model.fit(graph, features)
        embedding = self.embedding_model.get_embedding()
        embedding = np.sum(embedding, axis=0)
        return embedding

    def step(self, pick_nodes, kick_nodes=[]):
        self.picked_nodes = list(set(self.picked_nodes + pick_nodes))
        self.picked_nodes = [x for x in self.picked_nodes if x not in kick_nodes]
        self.graph = dgl.node_subgraph(self.dataset, self.picked_nodes)
        features = torch.stack([x for i, x in enumerate(self.dataset.ndata['feat']) if i in self.picked_nodes])
        labels = torch.stack([x for i, x in enumerate(self.dataset.ndata['label']) if i in self.picked_nodes])
        loss_fcn = torch.nn.CrossEntropyLoss()

        logits = self.GNN(self.graph, features)
        loss = loss_fcn(logits, labels)

        if len(self.picked_nodes) >= 2000:
            self.done = True

        return -loss, self.done


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, embedding_size, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm1d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv1d_size_out(size, kernel_size=5, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        conv_size = conv1d_size_out(conv1d_size_out(conv1d_size_out(embedding_size)))
        linear_input_size = conv_size * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        print(x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        print(x.shape)
        return self.head(x.view(x.size(0), -1))

parser = argparse.ArgumentParser(description='GAT')
parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
parser.add_argument("--epochs", type=int, default=500,
                    help="number of training epochs")
parser.add_argument("--num-heads", type=int, default=8,
                    help="number of hidden attention heads")
parser.add_argument("--num-out-heads", type=int, default=1,
                    help="number of output attention heads")
parser.add_argument("--num-layers", type=int, default=1,
                    help="number of hidden layers")
parser.add_argument("--num-hidden", type=int, default=8,
                    help="number of hidden units")
parser.add_argument("--residual", action="store_true", default=False,
                    help="use residual connection")
parser.add_argument("--in-drop", type=float, default=.6,
                    help="input feature dropout")
parser.add_argument("--attn-drop", type=float, default=.6,
                    help="attention dropout")
parser.add_argument("--lr", type=float, default=0.005,
                    help="learning rate")
parser.add_argument('--weight-decay', type=float, default=5e-4,
                    help="weight decay")
parser.add_argument('--negative-slope', type=float, default=0.2,
                    help="the negative slope of leaky relu")
parser.add_argument('--early-stop', action='store_true', default=True,
                    help="indicates whether to use early stop or not")
parser.add_argument('--fastmode', action="store_true", default=False,
                    help="skip re-evaluate the validation set")
args = parser.parse_args()


data = CoraGraphDataset()
dataset = data[0]
dataset = dgl.add_self_loop(dataset)

embedding_model = TENE()
embedding_model.fit(dgl.to_networkx(dataset).to_undirected(), np.array(dataset.ndata['feat']))
dataset_embedding = embedding_model.get_embedding()
dataset_embedding = np.sum(dataset_embedding, axis=0)

heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
GNN_model = GAT(args.num_layers,
            dataset.ndata['feat'].shape[1],
            args.num_hidden,
            data.num_labels,
            heads,
            F.elu,
            args.in_drop,
            args.attn_drop,
            args.negative_slope,
            args.residual)
GNN_model.load_state_dict(torch.load('GAT/es_checkpoint.pt'))

env = myenv(GNN_model, embedding_model, dataset)
env.reset()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


# Get number of actions from gym action space
n_actions = len(dataset.nodes().numpy())
embedding_size = 4000

policy_net = DQN(embedding_size*2, n_actions).to(device)
target_net = DQN(embedding_size*2, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            pred = policy_net(state)
            values, indices = pred.topk(3)
            picked_noeds = list(indices)
            return picked_noeds
    else:
        return list(set(np.random.choice(n_actions, 3)))


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    state = np.concatenate((dataset_embedding, env.state()), axis=0)
    state = torch.from_numpy(state)
    state = state.view(state.shape[0], 1, 1)
    print("Round " + str(i_episode))
    for t in count():
        print("Count " + str(t))
        # Select and perform an action
        pick_nodes = select_action(state)

        reward, done = env.step(pick_nodes)
        reward = torch.tensor([reward], device=device)

        if not done:
            next_state = np.concatenate((dataset_embedding, env.state()), axis=0)
            next_state = torch.from_numpy(next_state)
            next_state = next_state.view(next_state.shape[0], 1, 1)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, pick_nodes, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')