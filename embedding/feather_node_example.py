import numpy as np
import networkx as nx
from karateclub.node_embedding.attributed import TENE

import dgl
from dgl.data import CoraGraphDataset

data = CoraGraphDataset()
g = data[0]
g = dgl.add_self_loop(dgl.node_subgraph(g, list(set(np.random.choice(len(g.nodes()),5)))))
X = np.array(g.ndata['feat'])
g = dgl.to_networkx(g).to_undirected()

# g = nx.newman_watts_strogatz_graph(200, 20, 0.05)
#
# X = np.random.uniform(0, 1, (200, 200))

model = TENE()

model.fit(g, X)
embedding = model.get_embedding()
embedding = np.sum(embedding, axis=0)
# print(np.concatenate((embedding, embedding), axis=0).shape)
print(embedding.shape)