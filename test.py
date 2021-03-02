import argparse
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
import networkx as nx
import torch
import torch.nn.functional as F
import numpy as np
from GAT.model import GAT

data = CoraGraphDataset()
dataset = data[0]
picked_nodes = list(set(np.random.choice(dataset.nodes().numpy(), 200)))
graph = dgl.node_subgraph(dataset, picked_nodes)
graph = dgl.add_self_loop(graph)
features = torch.stack([x for i, x in enumerate(dataset.ndata['feat']) if i in picked_nodes])
g = dgl.to_networkx(graph)
f = np.array(dataset.ndata['feat'])
print("*************")
print(graph.edata['_ID'])

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

heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
model = GAT(args.num_layers,
            features.shape[1],
            args.num_hidden,
            data.num_labels,
            heads,
            F.elu,
            args.in_drop,
            args.attn_drop,
            args.negative_slope,
            args.residual)


model.load_state_dict(torch.load('GAT/es_checkpoint.pt'))
print(graph)
print(len(features))
print(features)
pred = model(graph, features)