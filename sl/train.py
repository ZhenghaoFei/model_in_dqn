import time
import numpy as np
from data  import process_gridworld_data
from model import *
from utils import *
import tensorflow as tf

# Data
imsize = 16
tf.app.flags.DEFINE_integer('imsize',         imsize,                      'Size of input image')
tf.app.flags.DEFINE_string('input',           'data/gridworld_'+str(imsize)+'.mat', 'Path to data')

# Parameters
tf.app.flags.DEFINE_boolean('skip_connection',True,                  'skip connection in dual model')
tf.app.flags.DEFINE_boolean('baseline',       False,                  'use baseline cnn model')

tf.app.flags.DEFINE_boolean('lr_decay',       True ,                  'lr_decay')

tf.app.flags.DEFINE_float('lr',               0.001,                  'Learning rate for RMSProp')
tf.app.flags.DEFINE_integer('epochs',         30,                     'Maximum epochs to train for')
tf.app.flags.DEFINE_integer('k',              1,                     'Number of value iterations')
tf.app.flags.DEFINE_integer('ch_i',           2,                      'Channels in input layer')
# tf.app.flags.DEFINE_integer('ch_h',           150,                    'Channels in initial hidden layer')
# tf.app.flags.DEFINE_integer('ch_q',           10,                     'Channels in q layer (~actions)')
tf.app.flags.DEFINE_integer('batchsize',      12,                     'Batch size')
tf.app.flags.DEFINE_integer('statebatchsize', 10,                     'Number of state inputs for each sample (real number, technically is k+1)')
tf.app.flags.DEFINE_boolean('untied_weights', False,                  'Untie weights of VI network')
# Misc.
tf.app.flags.DEFINE_integer('seed',           0,                      'Random seed for numpy')
tf.app.flags.DEFINE_integer('display_step',   1,                      'Print summary output every n epochs')
tf.app.flags.DEFINE_boolean('log',            False,                  'Enable for tensorboard summary')
tf.app.flags.DEFINE_string('logdir',          '/tmp/vintf/',          'Directory to store tensorboard summary')

config = tf.app.flags.FLAGS

np.random.seed(config.seed)

# symbolic input image tensor where typically first channel is image, second is the reward prior
X  = tf.placeholder(tf.float32, name="X",  shape=[None, config.imsize, config.imsize, config.ch_i])
# symbolic input batches of current positions
S = tf.placeholder(tf.int32,   name="S1", shape=[None, config.imsize, config.imsize, 1])
y  = tf.placeholder(tf.int32,   name="y",  shape=[None])

s_dim = [config.imsize, config.imsize]
a_dim = 8

if config.baseline:
    logits, nn = baseline_model(X, S, s_dim, a_dim)
else:
    #logits, nn = dual_model_mlayers(X, S, s_dim, a_dim, config.k, skip=config.skip_connection)
    logits, nn = dqn_model(X, S, s_dim, a_dim, config.k, skip=config.skip_connection)
count_parameters()

# Define loss and optimizer
y_ = tf.cast(y, tf.int64)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=y_, name='cross_entropy')
print "logits ", logits
print "label ", y_ 
cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
tf.add_to_collection('losses', cross_entropy_mean)

cost = tf.add_n(tf.get_collection('losses'), name='total_loss')
lr = tf.placeholder(dtype=tf.float32)
optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, epsilon=1e-6, centered=True).minimize(cost)

# Test model & calculate accuracy
cp = tf.cast(tf.argmax(nn, 1), tf.int32)
err = tf.reduce_mean(tf.cast(tf.not_equal(cp, y), dtype=tf.float32))
# Calculate accuracy
correct_prediction = tf.cast(tf.argmax(nn, 1), tf.int32)
accuracy = tf.reduce_mean(tf.cast(tf.not_equal(correct_prediction, y), dtype=tf.float32))
# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Xtrain, S1train, S2train, ytrain, Xtest, S1test, S2test, ytest = process_gridworld_data(input=config.input, imsize=config.imsize)
Xtrain, Strain, ytrain, Xtest, Stest,  ytest =  process_gridworld_data(input=config.input, imsize=config.imsize, statebatchsize=config.statebatchsize)

print "Xtrain shape: ", Xtrain.shape
print "Strain.shape: ", Strain.shape
print "ytrain.shape: ", ytrain.shape

# print Xtrain[0,:,:,0]
# print Xtrain[0,:,:,1]
# print Strain[0,:,:,0]

# Launch the graph
config_T = tf.ConfigProto()
config_T.gpu_options.allow_growth = True

with tf.Session(config=config_T) as sess:
    if config.log:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(config.logdir, sess.graph)
    sess.run(init)

    batch_size = config.batchsize
    print "imsize = ", imsize
    print "k = ", config.k
    print(fmt_row(10, ["Epoch", "Train Cost", "Train Err", "Epoch Time"]))
    print2save(fmt_row(10, ["Epoch", "Train Cost", "Train Err", "Epoch Time"]))

    learning_rate = config.lr
    for epoch in range(int(config.epochs)):
        if config.lr_decay:
            if epoch % 10 == 0 and epoch!=0:
                learning_rate /= 10
                print "learning_rate decay to: ", learning_rate
        tstart = time.time()
        avg_err, avg_cost = 0.0, 0.0
        num_batches = int(Xtrain.shape[0]/batch_size)
        # Loop over all batches
        for i in range(0, Xtrain.shape[0], batch_size):
            j = i + batch_size
            if j <= Xtrain.shape[0]:
                # Run optimization op (backprop) and cost op (to get loss value)
                fd = {X: Xtrain[i:j], S: Strain[i:j], y: ytrain[i:j].flatten(), lr:learning_rate}
                _, e_, c_ = sess.run([optimizer, err, cost], feed_dict=fd)
                avg_err += e_
                avg_cost += c_
        # Display logs per epoch step
        if epoch % config.display_step == 0:
            elapsed = time.time() - tstart
            print(fmt_row(10, [epoch, avg_cost/num_batches, avg_err/num_batches, elapsed]))
            print2save(fmt_row(10, [epoch, avg_cost/num_batches, avg_err/num_batches, elapsed]))

        if config.log:
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Average error', simple_value=float(avg_err/num_batches))
            summary.value.add(tag='Average cost', simple_value=float(avg_cost/num_batches))
            summary_writer.add_summary(summary, epoch)

        # if avg_err/num_batches< 0.1:
        eval_batch_size=10
        avg_acc = 0.0
        num_eval_batches = int(Xtest.shape[0]/eval_batch_size)
        # Loop over all batches
        for i in range(0, Xtest.shape[0], eval_batch_size):
            j = i + eval_batch_size
            if j <= Xtest.shape[0]:
                # Run optimization op (backprop) and cost op (to get loss value)
                fd = {X: Xtest[i:j], S: Stest[i:j], y: ytest[i:j].flatten(), lr:learning_rate}
                acc = accuracy.eval(fd)
                avg_acc += acc
        avg_acc/=num_eval_batches
        print("Accuracy: {}%".format(100 * (1 - avg_acc)))
        print2save("Accuracy: {}%".format(100 * (1 - avg_acc)))

    print("Finished training!")
    print2save("Finished training!")

    # Test model
    #correct_prediction = tf.cast(tf.argmax(nn, 1), tf.int32)
    # Calculate accuracy
    #accuracy = tf.reduce_mean(tf.cast(tf.not_equal(correct_prediction, y), dtype=tf.float32))
    #acc = accuracy.eval({X: Xtest, S:Stest, y: ytest.flatten()})
    #print('Accuracy: %', 100 * (1 - acc))
