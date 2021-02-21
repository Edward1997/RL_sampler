import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
import torch

# g = dgl.graph(torch.tensor([1]), torch.tensor([2]))
g = dgl.graph(([1], []))
print(g)