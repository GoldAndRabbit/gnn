
GRAPHSAGE_CONFIG = {
    'seed'              : 13,
    'dataset'           : 'cora',
    'dropout'           : 0.5,      # dropout probability
    'gpu'               : 0,
    'lr'                : 1e-2,
    'n-epochs'          : 200,      # number of training epochs
    'n-hidden'          : 16,       # number of hidden gcn units
    'n-layers'          : 1,        # number of hidden gcn layers
    'weight-decay'      : 5e-4,     # Weight for L2 loss
    'self-loop'         : False,    # graph self-loop (default=False)
    'aggregator-type'   :'gcn'      # Aggregator type: mean/gcn/pool/lstm
}