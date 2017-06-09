import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils import conv2d_flipkernel

def count_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("total_parameters:", total_parameters)

def baseline_model(X, S, s_dim, a_dim):
    S = tf.cast(S, dtype=tf.float32)
    state = tf.concat([X, S], axis=3)
    ch_h = 32

    # store Q(s,a) value
    q_a = tf.Variable(0, dtype=tf.float32, name="q_a", trainable=False)

    # state feature extraction
    state_f = layers.convolution2d(state, num_outputs=ch_h, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    state_f = layers.batch_norm(state_f)

    state_f = layers.convolution2d(state_f, num_outputs=ch_h, kernel_size=3, stride=1,padding='SAME', activation_fn=tf.nn.relu)
    state_f = layers.batch_norm(state_f)

    # model 1 from state feature to action number of next state
    net = layers.convolution2d(state_f, num_outputs=ch_h, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    net = layers.batch_norm(net)

    net = layers.convolution2d(net, num_outputs=ch_h, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    net = layers.batch_norm(net)

    net = layers.convolution2d(net, num_outputs=ch_h, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    net = layers.convolution2d(net, num_outputs=ch_h, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    net = layers.batch_norm(net)

    # model 1 to model 2 state transition
    net = layers.convolution2d(net, num_outputs=ch_h, kernel_size=3, stride=1, padding='VALID', activation_fn=tf.nn.relu)
    net = layers.batch_norm(net)
    net = layers.convolution2d(net, num_outputs=a_dim, kernel_size=3, stride=1,padding='VALID', activation_fn=tf.nn.relu)
    net = layers.batch_norm(net)

    # model 2 state transition
    net = layers.convolution2d(net, num_outputs=ch_h, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    net = layers.batch_norm(net)
    net = layers.convolution2d(net, num_outputs=ch_h, kernel_size=3, stride=1,padding='SAME', activation_fn=tf.nn.relu)
    net = layers.batch_norm(net)

    # MLP
    net = layers.flatten(net)
    net = layers.fully_connected(net, num_outputs=196, activation_fn=None)
    net = layers.fully_connected(net, num_outputs=128, activation_fn=None)

    # map to q 
    net = layers.fully_connected(net, num_outputs=a_dim, activation_fn=None)

    # pi mapping from q to action
    pi = layers.fully_connected(net, num_outputs=32, activation_fn=None)
    pi_logits = layers.fully_connected(pi, num_outputs=a_dim, activation_fn=None)
    pi_action = tf.nn.softmax(pi_logits, name="pi_action")

    return pi_logits, pi_action

def dual_model(X, S, s_dim, a_dim, k, skip=False):

    S = tf.cast(S, dtype=tf.float32)
    state = tf.concat([X, S], axis=3)
    ch_h = 32

    # store Q(s,a) value
    q_a = tf.Variable(0, dtype=tf.float32, name="q_a", trainable=False)

    # state feature extraction
    state_f = layers.convolution2d(state, num_outputs=ch_h, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    state_f = layers.batch_norm(state_f)

    state_f = layers.convolution2d(state_f, num_outputs=ch_h, kernel_size=3, stride=1,padding='SAME', activation_fn=tf.nn.relu)
    state_f = layers.batch_norm(state_f)

    # model 1 from state feature to action number of next state
    state_m1_h1 = layers.convolution2d(state_f, num_outputs=ch_h, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    state_m1_h1 = layers.batch_norm(state_m1_h1)

    state_m1_h2 = layers.convolution2d(state_m1_h1, num_outputs=ch_h, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    state_m1_h2 = layers.batch_norm(state_m1_h2)

    state_m1_n = layers.convolution2d(state_m1_h2, num_outputs=ch_h, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    state_m1_n = layers.convolution2d(state_m1_n, num_outputs=a_dim, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    state_m1_n = layers.batch_norm(state_m1_n)

    reward_m1_n = layers.fully_connected(layers.flatten(state_m1_h1), num_outputs=ch_h, activation_fn=None)
    reward_m1_n = layers.fully_connected(reward_m1_n, num_outputs=a_dim, activation_fn=None)

    q_a += reward_m1_n

    # State from statespace m1 to state space m2 through cnn
    state_m1_n = layers.convolution2d(state_m1_n, num_outputs=ch_h, kernel_size=3, stride=1, padding='VALID', activation_fn=tf.nn.relu)
    state_m1_n = layers.batch_norm(state_m1_n)
    state_m1_n = layers.convolution2d(state_m1_n, num_outputs=ch_h, kernel_size=3, stride=1,padding='VALID', activation_fn=tf.nn.relu)
    state_m1_n = layers.batch_norm(state_m1_n)

    # model 2 latent model
    ch_latent_actions = 8
    # with tf.variable_scope("model2", reuse=reuse):
    # state transition functuon
    m2_w0 = tf.Variable(np.random.randn(3, 3, 1, ch_h) * 0.01, dtype=tf.float32)
    m2_b0  = tf.Variable(np.random.randn(1, 1, 1, ch_h)    * 0.01, dtype=tf.float32)

    m2_w1 = tf.Variable(np.random.randn(3, 3, ch_h, ch_latent_actions) * 0.01, dtype=tf.float32)
    m2_b1  = tf.Variable(np.random.randn(1, 1, 1, ch_latent_actions)    * 0.01, dtype=tf.float32)
    
    # reward function
    # dim = s_dim[0]*s_dim[1]*ch_h
    dim = state_m1_n.shape[1]* state_m1_n.shape[2]*ch_h
    dim2 =  state_m1_n.shape[1]* state_m1_n.shape[2]*ch_latent_actions
    reward_w0 = tf.Variable(np.random.randn(dim, ch_h)*0.01 , dtype=tf.float32)
    reward_b0 = tf.Variable(tf.zeros([ch_h]), dtype=tf.float32, name="reward_b")
    reward_w1 = tf.Variable(np.random.randn(ch_h, ch_latent_actions)*0.01, dtype=tf.float32)
    reward_b1 = tf.Variable(tf.zeros([ch_latent_actions]), dtype=tf.float32, name="reward_b")

    # state value function
    value_w0 = tf.Variable(np.random.randn(dim2, ch_h) * 0.01 , dtype=tf.float32)
    value_b0 = tf.Variable(tf.zeros([ch_h]) , dtype=tf.float32, name="value_b")
    value_w1 = tf.Variable(np.random.randn(ch_h, ch_latent_actions) * 0.01 , dtype=tf.float32)
    value_b1 = tf.Variable(tf.zeros([ch_latent_actions]) * 0.01 , dtype=tf.float32, name="value_b")

    # gamma(discount rate)  function
    gamma_w0 = tf.Variable(np.random.randn(dim, ch_h) * 0.01 , dtype=tf.float32)
    gamma_b0 = tf.Variable(tf.zeros([ch_h]), dtype=tf.float32, name="gamma_b")
    gamma_w1 = tf.Variable(np.random.randn(ch_h, ch_latent_actions) * 0.01 , dtype=tf.float32)
    gamma_b1 = tf.Variable(tf.zeros([ch_latent_actions]) * 0.01 , dtype=tf.float32, name="gamma_b")

    # lambda(discount rate)  function
    lambda_w0 = tf.Variable(np.random.randn(dim, ch_h) * 0.01 , dtype=tf.float32)
    lambda_b0 = tf.Variable(tf.zeros([ch_h]), dtype=tf.float32, name="lambda_b")
    lambda_w1 = tf.Variable(np.random.randn(ch_h, ch_latent_actions) * 0.01 , dtype=tf.float32)
    lambda_b1 = tf.Variable(tf.zeros([ch_latent_actions]), dtype=tf.float32, name="lamda_b")



    for i in range(a_dim):
        # state_n = state_m1_n[:,:,:,i]
        state_n = tf.expand_dims(state_m1_n[:,:,:,i], 3)
        # state_n = tf.reshape(state_n, shape=[-1, s_dim[0], s_dim[1], 1])
        # gamma = tf.Variable(1, dtype=tf.float32, name="gamma")

        # gamma, rewards, value
        zeros = 0 * tf.range(0, tf.shape(state)[0])
        zeros = tf.expand_dims(zeros, 1)
        zeros = tf.tile(zeros, [1, k])
        zeros = tf.cast(zeros, dtype=tf.float32)
        gammas = zeros
        rewards = zeros
        values = zeros
        lambdas = zeros

        for j in range(k):
            # state
            state_m2_h1 = tf.nn.relu(tf.nn.conv2d(state_n, m2_w0, strides=(1, 1, 1, 1), padding='SAME') + m2_b0)
            state_m2_h1 = layers.batch_norm(state_m2_h1)

            state_m2_ns = tf.nn.relu(tf.nn.conv2d(state_m2_h1, m2_w1, strides=(1, 1, 1, 1), padding='SAME') + m2_b1)
            state_m2_ns = layers.batch_norm(state_m2_ns)

            flat_state_m2_h1 = layers.flatten(state_m2_h1)
            flat_state_m2_ns = layers.flatten(state_m2_ns)

            # reward
            reward_n = tf.matmul(flat_state_m2_h1, reward_w0) +reward_b0
            reward_n = tf.matmul(reward_n, reward_w1) +reward_b1

            # gamma
            gamma_n = tf.nn.relu(tf.matmul(flat_state_m2_h1, gamma_w0) +gamma_b0)
            gamma_n = tf.nn.sigmoid(tf.matmul(gamma_n, gamma_w1) +gamma_b1)

            # value
            value_n = tf.nn.relu(tf.matmul(flat_state_m2_ns, value_w0) + value_b0)
            value_n = tf.nn.relu(tf.matmul(value_n, value_w1) + value_b1)

            # labmda
            lambda_n = tf.nn.relu(tf.matmul(flat_state_m2_h1, lambda_w0) +lambda_b0)
            lambda_n = tf.nn.sigmoid(tf.matmul(lambda_n, lambda_w1) +lambda_b1)

            # select action = argmaxQ(s,a)
            q_n = reward_n + gamma_n*value_n
            Act = tf.cast(tf.argmax(q_n, axis=1), tf.int32)
            # print Act
            idx = tf.stack([tf.range(0, tf.shape(Act)[0]), Act], axis=1)
            # print idx
            # select next state
            state_nt = tf.expand_dims(tf.gather_nd(tf.transpose(state_m2_ns, [0,3,1,2]), idx), 3)

            if skip:
                state_n = tf.nn.relu(state_n+state_nt)
            else:
                state_n = state_nt
            # mask next state rewards gammas values
            mask = tf.reshape(tf.one_hot(j, depth=k),[1,-1])
            gammas += mask * tf.reshape(tf.gather_nd(gamma_n, idx),[-1,1])
            rewards += mask * tf.reshape(tf.gather_nd(reward_n, idx),[-1,1])
            values += mask * tf.reshape(tf.gather_nd(value_n, idx),[-1,1])
            lambdas += mask * tf.reshape(tf.gather_nd(lambda_n, idx),[-1,1])

        # g lambda
        zeros = 0 * tf.range(0, tf.shape(state)[0])
        zeros = tf.cast(zeros, dtype=tf.float32)
        g_lambda = zeros
        g_lambda = values[:, j]     # t = k
        for j in reversed(range(k-1)):
            # a backward pass to calculate return g lambda
            g_lambda = (1 - lambdas[:, j]) * values[:, j] + lambdas[:, j] * (rewards[:, j] + gammas[:, j]*g_lambda)

        # q_a
        mask = tf.reshape(tf.one_hot(i, depth=a_dim),[1,-1])
        g_lambda = tf.expand_dims(g_lambda, 1)
        g_lambda = tf.tile(g_lambda, [1, a_dim])
        q_a += mask * g_lambda


    # pi mapping from q to action
    pi = layers.fully_connected(q_a, num_outputs=16, activation_fn=None)
    pi_logits = layers.fully_connected(pi, num_outputs=a_dim, activation_fn=None)
    pi_action = tf.nn.softmax(pi_logits, name="pi_action")

    return pi_logits, pi_action


# def VI_Block(X, S1, S2, config):
#     k    = config.k    # Number of value iterations performed
#     ch_i = config.ch_i # Channels in input layer
#     ch_h = config.ch_h # Channels in initial hidden layer
#     ch_q = config.ch_q # Channels in q layer (~actions)
#     state_batch_size = config.statebatchsize # k+1 state inputs for each channel

#     bias  = tf.Variable(np.random.randn(1, 1, 1, ch_h)    * 0.01, dtype=tf.float32)
#     # weights from inputs to q layer (~reward in Bellman equation)
#     w0    = tf.Variable(np.random.randn(3, 3, ch_i, ch_h) * 0.01, dtype=tf.float32)
#     w1    = tf.Variable(np.random.randn(1, 1, ch_h, 1)    * 0.01, dtype=tf.float32)
#     w     = tf.Variable(np.random.randn(3, 3, 1, ch_q)    * 0.01, dtype=tf.float32)
#     # feedback weights from v layer into q layer (~transition probabilities in Bellman equation)
#     w_fb  = tf.Variable(np.random.randn(3, 3, 1, ch_q)    * 0.01, dtype=tf.float32)
#     w_o   = tf.Variable(np.random.randn(ch_q, 8)          * 0.01, dtype=tf.float32)

#     # initial conv layer over image+reward prior
#     h = conv2d_flipkernel(X, w0, name="h0") + bias

#     r = conv2d_flipkernel(h, w1, name="r")
#     q = conv2d_flipkernel(r, w, name="q")
#     v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

#     for i in range(0, k-1):
#         rv = tf.concat([r, v], 3)
#         wwfb = tf.concat([w, w_fb], 2)
#         q = conv2d_flipkernel(rv, wwfb, name="q")
#         v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

#     # do one last convolution
#     q = conv2d_flipkernel(tf.concat([r, v], 3),
#                           tf.concat([w, w_fb], 2), name="q")

#     # CHANGE TO THEANO ORDERING
#     # Since we are selecting over channels, it becomes easier to work with
#     # the tensor when it is in NCHW format vs NHWC
#     q = tf.transpose(q, perm=[0, 3, 1, 2])

#     # Select the conv-net channels at the state position (S1,S2).
#     # This intuitively corresponds to each channel representing an action, and the convnet the Q function.
#     # The tricky thing is we want to select the same (S1,S2) position *for each* channel and for each sample
#     # TODO: performance can be improved here by substituting expensive
#     #       transpose calls with better indexing for gather_nd
#     bs = tf.shape(q)[0]
#     rprn = tf.reshape(tf.tile(tf.reshape(tf.range(bs), [-1, 1]), [1, state_batch_size]), [-1])
#     ins1 = tf.cast(tf.reshape(S1, [-1]), tf.int32)
#     ins2 = tf.cast(tf.reshape(S2, [-1]), tf.int32)
#     idx_in = tf.transpose(tf.stack([ins1, ins2, rprn]), [1, 0])
#     q_out = tf.gather_nd(tf.transpose(q, [2, 3, 0, 1]), idx_in, name="q_out")

#     # add logits
#     logits = tf.matmul(q_out, w_o)
#     # softmax output weights
#     output = tf.nn.softmax(logits, name="output")
#     return logits, output

# # similar to the normal VI_Block except there are separate weights for each q layer
# def VI_Untied_Block(X, S1, S2, config):
#     k    = config.k    # Number of value iterations performed
#     ch_i = config.ch_i # Channels in input layer
#     ch_h = config.ch_h # Channels in initial hidden layer
#     ch_q = config.ch_q # Channels in q layer (~actions)
#     state_batch_size = config.statebatchsize # k+1 state inputs for each channel

#     bias   = tf.Variable(np.random.randn(1, 1, 1, ch_h)    * 0.01, dtype=tf.float32)
#     # weights from inputs to q layer (~reward in Bellman equation)
#     w0     = tf.Variable(np.random.randn(3, 3, ch_i, ch_h) * 0.01, dtype=tf.float32)
#     w1     = tf.Variable(np.random.randn(1, 1, ch_h, 1)    * 0.01, dtype=tf.float32)
#     w_l    = [tf.Variable(np.random.randn(3, 3, 1, ch_q)   * 0.01, dtype=tf.float32) for i in range(0, k+1)]
#     # feedback weights from v layer into q layer (~transition probabilities in Bellman equation)
#     w_fb_l = [tf.Variable(np.random.randn(3, 3, 1, ch_q)   * 0.01, dtype=tf.float32) for i in range(0,k)]
#     w_o    = tf.Variable(np.random.randn(ch_q, 8)          * 0.01, dtype=tf.float32)

#     # initial conv layer over image+reward prior
#     h = conv2d_flipkernel(X, w0, name="h0") + bias

#     r = conv2d_flipkernel(h, w1, name="r")
#     q = conv2d_flipkernel(r, w_l[0], name="q")
#     v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

#     for i in range(0, k-1):
#         rv = tf.concat([r, v], 3)
#         wwfb = tf.concat([w_l[i+1], w_fb_l[i]], 2)
#         q = conv2d_flipkernel(rv, wwfb, name="q")
#         v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

#     # do one last convolution
#     q = conv2d_flipkernel(tf.concat([r, v], 3),
#                           tf.concat([w_l[k], w_fb_l[k-1]], 2), name="q")

#     # CHANGE TO THEANO ORDERING
#     # Since we are selecting over channels, it becomes easier to work with
#     # the tensor when it is in NCHW format vs NHWC
#     q = tf.transpose(q, perm=[0, 3, 1, 2])

#     # Select the conv-net channels at the state position (S1,S2).
#     # This intuitively corresponds to each channel representing an action, and the convnet the Q function.
#     # The tricky thing is we want to select the same (S1,S2) position *for each* channel and for each sample
#     # TODO: performance can be improved here by substituting expensive
#     #       transpose calls with better indexing for gather_nd
#     bs = tf.shape(q)[0]
#     rprn = tf.reshape(tf.tile(tf.reshape(tf.range(bs), [-1, 1]), [1, state_batch_size]), [-1])
#     ins1 = tf.cast(tf.reshape(S1, [-1]), tf.int32)
#     ins2 = tf.cast(tf.reshape(S2, [-1]), tf.int32)
#     idx_in = tf.transpose(tf.stack([ins1, ins2, rprn]), [1, 0])
#     q_out = tf.gather_nd(tf.transpose(q, [2, 3, 0, 1]), idx_in, name="q_out")

#     # add logits
#     logits = tf.matmul(q_out, w_o)
#     # softmax output weights
#     output = tf.nn.softmax(logits, name="output")
#     return logits, output
