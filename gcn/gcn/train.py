import time
import tensorflow as tf
from gcn.utils  import *
from gcn.models import GCN, MLP
from gcn.config import *

seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset)

features = preprocess_features(features)

if model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, max_degree)
    num_supports = 1 + max_degree
    model_func = GCN
elif model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(model))

placeholders = {
    'support'             : [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features'            : tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels'              : tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask'         : tf.placeholder(tf.int32),
    'dropout'             : tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)
}

model = model_func(placeholders, input_dim=features[2][1], logging=True)

sess = tf.Session()

def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


sess.run(tf.global_variables_initializer())
cost_val = []

for epoch in range(epochs):
    t = time.time()
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: dropout})

    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    print("Epoch:",         '%04d' % (epoch + 1),
          "train_loss=",    "{:.5f}".format(outs[1]),
          "train_acc=",     "{:.5f}".format(outs[2]),
          "val_loss=",      "{:.5f}".format(cost),
          "val_acc=",       "{:.5f}".format(acc),
          "time=",          "{:.5f}".format(time.time()-t)
          )

    if epoch > early_stopping and cost_val[-1] > np.mean(cost_val[-(early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:",
      "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
