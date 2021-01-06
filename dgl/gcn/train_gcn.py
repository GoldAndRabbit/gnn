import time
import numpy    as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from gcn import GCN
from pprint import pprint
from gcn_config import GCN_CONFIG

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def gcn_eval_pipeline():
    if   GCN_CONFIG['dataset'] == 'cora':
        data = CoraGraphDataset()
    elif GCN_CONFIG['dataset'] == 'citeseer':
        data = CiteseerGraphDataset()
    elif GCN_CONFIG['dataset'] == 'pubmed':
        data = PubmedGraphDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(GCN_CONFIG['dataset']))

    g = data[0]
    if GCN_CONFIG['gpu'] < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(GCN_CONFIG['gpu'])

    features    = g.ndata['feat']
    labels      = g.ndata['label']
    train_mask  = g.ndata['train_mask']
    val_mask    = g.ndata['val_mask']
    test_mask   = g.ndata['test_mask']
    in_feats    = features.shape[1]
    n_classes   = data.num_labels
    n_edges     = data.graph.number_of_edges()
    print("""----Data statistics------'
    # Edges %d
    # Classes %d
    # Train samples %d
    # Val samples %d
    # Test samples %d""" %
        (n_edges, n_classes,train_mask.int().sum().item(), val_mask.int().sum().item(), test_mask.int().sum().item()))

    # add self loop
    if GCN_CONFIG['self-loop']:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = GCN(g, in_feats,
                GCN_CONFIG['n-hidden'],
                n_classes,
                GCN_CONFIG['n-layers'],
                F.relu,
                GCN_CONFIG['dropout'])

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=GCN_CONFIG['lr'], weight_decay=GCN_CONFIG['weight-decay'])

    # initialize graph
    dur = []
    for epoch in range(GCN_CONFIG['n-epochs']):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(model, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(), acc, n_edges / np.mean(dur) / 1000))

    acc = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))


def set_torch_default_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    set_torch_default_seed(GCN_CONFIG['seed'])
    pprint(GCN_CONFIG)
    # torch seed set effect result heavily
    gcn_eval_pipeline()
