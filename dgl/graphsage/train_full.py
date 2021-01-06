import argparse
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.data import CoraGraphDataset,CiteseerGraphDataset,PubmedGraphDataset
from dgl.nn.pytorch.conv import SAGEConv
from graphSAGE_config import GRAPHSAGE_CONFIG
from pprint import pprint


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type)) # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


def evaluate(model, graph, features, labels, nid):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[nid]
        labels = labels[nid]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def graphSAGE_eval_pipeline():
    if GRAPHSAGE_CONFIG['dataset'] == 'cora':
        data = CoraGraphDataset()
    elif GRAPHSAGE_CONFIG['dataset'] == 'citeseer':
        data = citegrh.load_citeseer()
    elif GRAPHSAGE_CONFIG['dataset'] == 'pubmed':
        data = citegrh.load_pubmed()
    else:
        raise ValueError('Unknown dataset: {}'.format(GRAPHSAGE_CONFIG['dataset']))

    g = data[0]
    features    = g.ndata['feat']
    labels      = g.ndata['label']
    train_mask  = g.ndata['train_mask']
    val_mask    = g.ndata['val_mask']
    test_mask   = g.ndata['test_mask']
    in_feats    = features.shape[1]
    n_classes   = data.num_labels
    n_edges     = data.graph.number_of_edges()
    # print("""----Data statistics------'
    # # Edges %d
    # # Classes %d
    # # Train samples %d
    # # Val samples %d
    # # Test samples %d""" %
    #     (n_edges, n_classes, train_mask.int().sum().item(), val_mask.int().sum().item(), test_mask.int().sum().item()))

    if GRAPHSAGE_CONFIG['gpu'] < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(GRAPHSAGE_CONFIG['gpu'])
        features    = features.cuda()
        labels      = labels.cuda()
        train_mask  = train_mask.cuda()
        val_mask    = val_mask.cuda()
        test_mask   = test_mask.cuda()
        print("use cuda:", GRAPHSAGE_CONFIG['gpu'])

    train_nid   = train_mask.nonzero().squeeze()
    val_nid     = val_mask.nonzero().squeeze()
    test_nid    = test_mask.nonzero().squeeze()

    # graph preprocess and calculate normalization factor
    g = dgl.remove_self_loop(g)
    n_edges = g.number_of_edges()
    if cuda:
        g = g.int().to(GRAPHSAGE_CONFIG['gpu'])

    # create GraphSAGE model
    model = GraphSAGE(
        in_feats,
        GRAPHSAGE_CONFIG['n-hidden'],
        n_classes,
        GRAPHSAGE_CONFIG['n-layers'],
        F.relu,
        GRAPHSAGE_CONFIG['dropout'],
        GRAPHSAGE_CONFIG['aggregator-type'],
    )

    if cuda:
        model.cuda()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=GRAPHSAGE_CONFIG['lr'],
                                 weight_decay=GRAPHSAGE_CONFIG['weight-decay'])

    # initialize graph
    dur = []
    for epoch in range(GRAPHSAGE_CONFIG['n-epochs']):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(g, features)
        loss = F.cross_entropy(logits[train_nid], labels[train_nid])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, g, features, labels, val_nid)
        # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
        #       "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(), acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, g, features, labels, test_nid)
    print("Test Accuracy {:.4f}".format(acc))


def set_torch_default_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # torch seed set effect result heavily
    set_torch_default_seed(GRAPHSAGE_CONFIG['seed'])
    pprint(GRAPHSAGE_CONFIG)
    graphSAGE_eval_pipeline()

