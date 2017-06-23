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
    ch_h = 8

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
    ch_h = 16
    # store Q(s,a) value
    q_a = tf.Variable(0, dtype=tf.float32, name="q_a", trainable=False)

    # state feature extraction
    state_f = layers.convolution2d(state, num_outputs=ch_h, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    state_f = layers.batch_norm(state_f, decay=0.9)

    state_f = layers.convolution2d(state_f, num_outputs=ch_h, kernel_size=3, stride=1,padding='SAME', activation_fn=tf.nn.relu)
    state_f = layers.batch_norm(state_f, decay=0.9)

    # print state_f

    # model 1 from state feature to action number of next state
    state_m1_h1 = layers.convolution2d(state_f, num_outputs=ch_h, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    state_m1_h1 = layers.batch_norm(state_m1_h1, decay=0.9)

    state_m1_h2 = layers.convolution2d(state_m1_h1, num_outputs=ch_h, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    state_m1_h2 = layers.batch_norm(state_m1_h2,decay=0.9)

    state_m1_n = layers.convolution2d(state_m1_h2, num_outputs=ch_h, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    state_m1_n = layers.convolution2d(state_m1_n, num_outputs=a_dim, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    state_m1_n = layers.batch_norm(state_m1_n, decay=0.9)

    reward_m1_n = layers.fully_connected(layers.flatten(state_m1_h1), num_outputs=ch_h, activation_fn=None)
    reward_m1_n = layers.fully_connected(reward_m1_n, num_outputs=a_dim, activation_fn=None)

    q_a += reward_m1_n

    # State from statespace m1 to state space m2 through cnn
    state_m1_n = layers.convolution2d(state_m1_n, num_outputs=ch_h, kernel_size=3, stride=1, padding='VALID', activation_fn=tf.nn.relu)
    state_m1_n = layers.batch_norm(state_m1_n, decay=0.9)
    state_m1_n = layers.convolution2d(state_m1_n, num_outputs=a_dim, kernel_size=3, stride=1,padding='VALID', activation_fn=tf.nn.relu)
    state_m1_n = layers.batch_norm(state_m1_n, decay=0.9)

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

    # from (batch_size, dim1, dim2, a_dim) to (a_dim * batch_size,  dim1 , dim2 , 1)
    state_n = tf.transpose(state_m1_n, [0,3,1,2])
    state_n = tf.reshape(state_n, shape=[-1, int(state_n.shape[2]), int(state_n.shape[3]) , 1])

    # gamma, rewards, value
    zeros = 0 * tf.range(0, tf.shape(state_n)[0])
    zeros = tf.expand_dims(zeros, 1)
    zeros = tf.tile(zeros, [1, k])

    zeros = tf.cast(zeros, dtype=tf.float32)
    gammas = zeros
    rewards = zeros
    values = zeros
    lambdas = zeros

    # print "zeros: ", zeros

    for j in range(k):
        # state
        state_m2_h1 = tf.nn.relu(tf.nn.conv2d(state_n, m2_w0, strides=(1, 1, 1, 1), padding='SAME') + m2_b0)
        state_m2_h1 = layers.batch_norm(state_m2_h1, decay=0.9)

        state_m2_ns = tf.nn.relu(tf.nn.conv2d(state_m2_h1, m2_w1, strides=(1, 1, 1, 1), padding='SAME') + m2_b1)
        state_m2_ns = layers.batch_norm(state_m2_ns, decay=0.9)

        flat_state_m2_h1 = layers.flatten(state_m2_h1)
        flat_state_m2_ns = layers.flatten(state_m2_ns)

        # reward
        reward_n = tf.nn.relu(tf.matmul(flat_state_m2_h1, reward_w0) +reward_b0)
        reward_n = tf.matmul(reward_n, reward_w1) +reward_b1

        # gamma
        gamma_n = tf.nn.relu(tf.matmul(flat_state_m2_h1, gamma_w0) +gamma_b0)
        gamma_n = tf.nn.sigmoid(tf.matmul(gamma_n, gamma_w1) +gamma_b1)

        # value
        value_n = tf.nn.relu(tf.matmul(flat_state_m2_ns, value_w0) + value_b0)
        value_n = tf.matmul(value_n, value_w1) + value_b1

        # labmda
        lambda_n = tf.nn.relu(tf.matmul(flat_state_m2_h1, lambda_w0) +lambda_b0)
        lambda_n = tf.nn.sigmoid(tf.matmul(lambda_n, lambda_w1) +lambda_b1)

        # select action = argmaxQ(s,a)
        q_n = reward_n + gamma_n*value_n
        # print "q_n: ", q_n
        Act = tf.cast(tf.argmax(q_n, axis=1), tf.int32)
        # print Act
        idx = tf.stack([tf.range(0, tf.shape(Act)[0]), Act], axis=1)
        # print idx
        # select next state
        state_nt = tf.expand_dims(tf.gather_nd(tf.transpose(state_m2_ns, [0,3,1,2]), idx), 3)
        # print "state_nt: ", state_nt
        if skip:
            state_n = tf.nn.relu(state_n+state_nt)
        else:
            state_n = state_nt

        # mask next state rewards gammas values
        mask = tf.expand_dims(tf.one_hot(j, depth=k), axis=0)

        # print "mask: ", mask
        # print "gather nd: ",tf.expand_dims(tf.gather_nd(gamma_n, idx),axis=1)

        gammas += mask * tf.expand_dims(tf.gather_nd(gamma_n, idx),axis=1)
        rewards += mask * tf.expand_dims(tf.gather_nd(reward_n, idx),axis=1)
        values += mask * tf.expand_dims(tf.gather_nd(value_n, idx),axis=1)
        lambdas += mask * tf.expand_dims(tf.gather_nd(lambda_n, idx),axis=1)


    # g lambda
    g_lambda = zeros
    g_lambda = values[:, j]     # t = k
    for j in reversed(range(k-1)):
        # a backward pass to calculate return g lambda
        g_lambda = (1 - lambdas[:, j]) * values[:, j] + lambdas[:, j] * (rewards[:, j] + gammas[:, j]*g_lambda)


    g_lambda = tf.reshape(g_lambda, shape=[-1, a_dim])
    q_a += g_lambda


    # pi mapping from q to action
    pi = layers.fully_connected(q_a, num_outputs=16, activation_fn=None)
    pi_logits = layers.fully_connected(pi, num_outputs=a_dim, activation_fn=None)
    pi_action = tf.nn.softmax(pi_logits, name="pi_action")

    return pi_logits, pi_action

def dual_model_mlayers(X, S, s_dim, a_dim, k, skip=False):
    print "dual_model_mlayers"
    S = tf.cast(S, dtype=tf.float32)
    state = tf.concat([X, S], axis=3)
    ch_h = 16
    ch_state_1 = 8
    ch_state_2 = 8

    # store Q(s,a) value
    q_a = tf.Variable(0, dtype=tf.float32, name="q_a", trainable=False)

    # state feature extraction
    state_f = layers.convolution2d(state, num_outputs=ch_h, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    state_f = layers.batch_norm(state_f, decay=0.9)

    state_f = layers.convolution2d(state_f, num_outputs=ch_h, kernel_size=3, stride=1,padding='SAME', activation_fn=tf.nn.relu)
    state_f = layers.batch_norm(state_f, decay=0.9)

    # print state_f

    # model 1 from state feature to action number of next state
    state_m1_h1 = layers.convolution2d(state_f, num_outputs=ch_h, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    state_m1_h1 = layers.batch_norm(state_m1_h1, decay=0.9)

    state_m1_h2 = layers.convolution2d(state_m1_h1, num_outputs=ch_h, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    state_m1_h2 = layers.batch_norm(state_m1_h2,decay=0.9)

    state_m1_n = layers.convolution2d(state_m1_h2, num_outputs=ch_h, kernel_size=3, stride=1, padding='VALID', activation_fn=tf.nn.relu)
    state_m1_n = layers.convolution2d(state_m1_n, num_outputs=ch_state_1*a_dim, kernel_size=3, stride=1, padding='VALID', activation_fn=tf.nn.relu)
    state_m1_n = layers.batch_norm(state_m1_n, decay=0.9)

    reward_m1_n = layers.fully_connected(layers.flatten(state_m1_h1), num_outputs=ch_h, activation_fn=None)
    reward_m1_n = layers.fully_connected(reward_m1_n, num_outputs=a_dim, activation_fn=None)

    q_a += reward_m1_n

    # State from statespace m1 to state space m2 through cnn
    state_m1_n = layers.convolution2d(state_m1_n, num_outputs=ch_h, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    state_m1_n = layers.batch_norm(state_m1_n, decay=0.9)
    state_m1_n = layers.convolution2d(state_m1_n, num_outputs=ch_state_2*a_dim, kernel_size=3, stride=1,padding='VALID', activation_fn=tf.nn.relu)
    state_m1_n = layers.batch_norm(state_m1_n, decay=0.9)

    # model 2 latent model
    ch_latent_actions = 8
    # with tf.variable_scope("model2", reuse=reuse):
    # state transition functuon
    m2_w0 = tf.Variable(np.random.randn(3, 3, ch_state_2, ch_h) * 0.01, dtype=tf.float32)
    m2_b0  = tf.Variable(np.random.randn(1, 1, 1, ch_h)    * 0.01, dtype=tf.float32)

    m2_w1 = tf.Variable(np.random.randn(3, 3, ch_h, ch_latent_actions*ch_state_2) * 0.01, dtype=tf.float32)
    m2_b1  = tf.Variable(np.random.randn(1, 1, 1, ch_latent_actions*ch_state_2)    * 0.01, dtype=tf.float32)
    
    # reward function
    # dim = s_dim[0]*s_dim[1]*ch_h
    dim = state_m1_n.shape[1]* state_m1_n.shape[2]*ch_h
    dim2 =  state_m1_n.shape[1]* state_m1_n.shape[2]*ch_latent_actions*ch_state_2
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

    # from (batch_size, dim1, dim2, a_dim * ch_state_1) to (a_dim * batch_size,  dim1 , dim2 , ch_state_1)
    # (batch_size, a_dim* ch_state_1, dim1 , dim2)
    state_n = tf.transpose(state_m1_n, [0,3,1,2])
    # (a_dim * batch_size, ch_state_1, dim1 , dim2 )
    state_n = tf.reshape(state_n, shape=[-1, ch_state_1, int(state_n.shape[2]), int(state_n.shape[3])])
    # (a_dim * batch_size, dim1 , dim2, ch_state_1 )
    state_n = tf.transpose(state_n, [0,2,3,1])

    # gamma, rewards, value
    zeros = 0 * tf.range(0, tf.shape(state_n)[0])
    zeros = tf.expand_dims(zeros, 1)
    zeros = tf.tile(zeros, [1, k])

    zeros = tf.cast(zeros, dtype=tf.float32)
    gammas = zeros
    rewards = zeros
    values = zeros
    lambdas = zeros

    # print "zeros: ", zeros

    for j in range(k):
        # state
        state_m2_h1 = tf.nn.relu(tf.nn.conv2d(state_n, m2_w0, strides=(1, 1, 1, 1), padding='SAME') + m2_b0)
        state_m2_h1 = layers.batch_norm(state_m2_h1, decay=0.9)

        state_m2_ns = tf.nn.relu(tf.nn.conv2d(state_m2_h1, m2_w1, strides=(1, 1, 1, 1), padding='SAME') + m2_b1)
        state_m2_ns = layers.batch_norm(state_m2_ns, decay=0.9)

        flat_state_m2_h1 = layers.flatten(state_m2_h1)
        flat_state_m2_ns = layers.flatten(state_m2_ns)

        # reward
        reward_n = tf.nn.relu(tf.matmul(flat_state_m2_h1, reward_w0) +reward_b0)
        reward_n = tf.matmul(reward_n, reward_w1) +reward_b1

        # gamma
        gamma_n = tf.nn.relu(tf.matmul(flat_state_m2_h1, gamma_w0) +gamma_b0)
        gamma_n = tf.nn.sigmoid(tf.matmul(gamma_n, gamma_w1) +gamma_b1)

        # value
        value_n = tf.nn.relu(tf.matmul(flat_state_m2_ns, value_w0) + value_b0)
        value_n = tf.matmul(value_n, value_w1) + value_b1

        # labmda
        lambda_n = tf.nn.relu(tf.matmul(flat_state_m2_h1, lambda_w0) +lambda_b0)
        lambda_n = tf.nn.sigmoid(tf.matmul(lambda_n, lambda_w1) +lambda_b1)

        # select action = argmaxQ(s,a)
        q_n = reward_n + gamma_n * value_n
        # print "q_n: ", q_n
        Act = tf.cast(tf.argmax(q_n, axis=1), tf.int32)
        # print Act
        idx = tf.stack([tf.range(0, tf.shape(Act)[0]), Act], axis=1)
        # print idx

        # select next state
        # print state_m2_nss
        # state_m2_ns = (?, dim1, dim2, ch_latent_actions * ch_state_2)
        # state_nt = (?, ch_latent_actions * ch_state_2, dim1, dim2)
        state_nt = tf.transpose(state_m2_ns, [0,3,1,2])
        # state_nt = (? , ch_latent_actions, ch_state_2, dim1, dim2)
        # print "state nt:", state_nt
        state_nt = tf.reshape(state_nt, shape=[-1, ch_latent_actions, ch_state_2, int(state_nt.shape[2]), int(state_nt.shape[3])])
        # print "state nt:", state_nt
        state_nt = tf.gather_nd(state_nt, idx)
        state_nt = tf.transpose(state_nt, [0,2,3,1])


        if skip:
            state_n = tf.nn.relu(state_n+state_nt)
        else:
            state_n = state_nt

        # mask next state rewards gammas values
        mask = tf.expand_dims(tf.one_hot(j, depth=k), axis=0)

        # print "mask: ", mask
        # print "gather nd: ",tf.expand_dims(tf.gather_nd(gamma_n, idx),axis=1)

        gammas += mask * tf.expand_dims(tf.gather_nd(gamma_n, idx),axis=1)
        rewards += mask * tf.expand_dims(tf.gather_nd(reward_n, idx),axis=1)
        values += mask * tf.expand_dims(tf.gather_nd(value_n, idx),axis=1)
        lambdas += mask * tf.expand_dims(tf.gather_nd(lambda_n, idx),axis=1)

    # g lambda
    g_lambda = zeros
    g_lambda = values[:, j]     # t = k
    for j in reversed(range(k-1)):
        # a backward pass to calculate return g lambda
        g_lambda = (1 - lambdas[:, j]) * values[:, j] + lambdas[:, j] * (rewards[:, j] + gammas[:, j] * g_lambda)


    g_lambda = tf.reshape(g_lambda, shape=[-1, a_dim])
    q_a += g_lambda


    # pi mapping from q to action
    pi = layers.fully_connected(q_a, num_outputs=16, activation_fn=None)
    pi_logits = layers.fully_connected(pi, num_outputs=a_dim, activation_fn=None)
    pi_action = tf.nn.softmax(pi_logits, name="pi_action")

    return pi_logits, pi_action

def dual_model_FClayers(X, S, s_dim, a_dim, k, skip=False):
    print "dual_model_FClayers"
    S = tf.cast(S, dtype=tf.float32)
    state = tf.concat([X, S], axis=3)
    ch_h = 16
    ch_state = 8

    # store Q(s,a) value
    q_a = tf.Variable(0, dtype=tf.float32, name="q_a", trainable=False)

    # state feature extraction
    state_f = layers.convolution2d(state, num_outputs=ch_h, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    state_f = layers.batch_norm(state_f, decay=0.9)

    state_f = layers.convolution2d(state_f, num_outputs=ch_h, kernel_size=3, stride=1,padding='SAME', activation_fn=tf.nn.relu)
    state_f = layers.batch_norm(state_f, decay=0.9)

    # print state_f

    # model 1 from state feature to action number of next state
    state_m1_h1 = layers.convolution2d(state_f, num_outputs=ch_h, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    state_m1_h1 = layers.batch_norm(state_m1_h1, decay=0.9)

    state_m1_h2 = layers.convolution2d(state_m1_h1, num_outputs=ch_h, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    state_m1_h2 = layers.batch_norm(state_m1_h2,decay=0.9)

    state_m1_n = layers.convolution2d(state_m1_h2, num_outputs=ch_h, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    state_m1_n = layers.convolution2d(state_m1_n, num_outputs=ch_state*a_dim, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    state_m1_n = layers.batch_norm(state_m1_n, decay=0.9)

    reward_m1_n = layers.fully_connected(layers.flatten(state_m1_h1), num_outputs=ch_h, activation_fn=None)
    reward_m1_n = layers.fully_connected(reward_m1_n, num_outputs=a_dim, activation_fn=None)

    q_a += reward_m1_n

    # State from statespace m1 to state space m2 through cnn
    state_m1_n = layers.convolution2d(state_m1_n, num_outputs=ch_h, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
    state_m1_n = layers.batch_norm(state_m1_n, decay=0.9)
    state_m1_n = layers.convolution2d(state_m1_n, num_outputs=ch_state*a_dim, kernel_size=3, stride=1,padding='VALID', activation_fn=tf.nn.relu)
    state_m1_n = layers.batch_norm(state_m1_n, decay=0.9)

    # model 2 latent model
    ch_latent_actions = 8

    # with tf.variable_scope("model2", reuse=reuse):
    # state transition functuon
    m2_w0 = tf.Variable(np.random.randn(3, 3, ch_state, ch_h) * 0.01, dtype=tf.float32)
    m2_b0  = tf.Variable(np.random.randn(1, 1, 1, ch_h)    * 0.01, dtype=tf.float32)

    m2_w1 = tf.Variable(np.random.randn(3, 3, ch_h, ch_latent_actions*ch_state) * 0.01, dtype=tf.float32)
    m2_b1  = tf.Variable(np.random.randn(1, 1, 1, ch_latent_actions*ch_state)    * 0.01, dtype=tf.float32)
    
    # reward function
    # dim = s_dim[0]*s_dim[1]*ch_h
    dim = state_m1_n.shape[1]* state_m1_n.shape[2]*ch_h
    dim2 =  state_m1_n.shape[1]* state_m1_n.shape[2]*ch_latent_actions*ch_state
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

    # from (batch_size, dim1, dim2, a_dim * ch_state) to (a_dim * batch_size,  dim1 , dim2 , ch_state)
    # (batch_size, a_dim* ch_state, dim1 , dim2)
    state_n = tf.transpose(state_m1_n, [0,3,1,2])
    # (a_dim * batch_size, ch_state, dim1 , dim2 )
    state_n = tf.reshape(state_n, shape=[-1, ch_state, int(state_n.shape[2]), int(state_n.shape[3])])
    # (a_dim * batch_size, dim1 , dim2, ch_state )
    state_n = tf.transpose(state_n, [0,2,3,1])

    # fully connected part
    ch_h_fc1 = 32
    ch_h_fc2 = 32
    # state_fc = layers.flatten(state_n) # initial input
    # shared weights
    dim_fci = state_n.shape[1]* state_n.shape[2]*ch_state # input dimension
    dim_fco =  state_n.shape[1]* state_n.shape[2]*ch_latent_actions*ch_state # output dimension

    fc1_w = tf.Variable(np.random.randn(dim_fci, ch_h_fc1)*0.01 , dtype=tf.float32)
    fc1_b = tf.Variable(tf.zeros([ch_h_fc1]), dtype=tf.float32)
    fc2_w = tf.Variable(np.random.randn(ch_h_fc1, ch_h_fc2)*0.01, dtype=tf.float32)
    fc2_b = tf.Variable(tf.zeros([ch_h_fc2]),dtype=tf.float32)
    
    fco_w = tf.Variable(np.random.randn(ch_h_fc2, dim_fco)*0.01 , dtype=tf.float32)
    fco_b = tf.Variable(tf.zeros([dim_fco]), dtype=tf.float32)
    # gamma, rewards, value
    zeros = 0 * tf.range(0, tf.shape(state_n)[0])
    zeros = tf.expand_dims(zeros, 1)
    zeros = tf.tile(zeros, [1, k])

    zeros = tf.cast(zeros, dtype=tf.float32)
    gammas = zeros
    rewards = zeros
    values = zeros
    lambdas = zeros

    # print "zeros: ", zeros

    for j in range(k):

        # fc state transition
        state_fc = layers.flatten(state_n) # initial input
        state_fc = layers.batch_norm(state_fc, decay=0.9)

        state_fc1 = tf.nn.relu(tf.matmul(state_fc, fc1_w) + fc1_b)
        state_fc1 = layers.batch_norm(state_fc1, decay=0.9)

        state_fc2 = tf.nn.relu(tf.matmul(state_fc1,fc2_w) + fc2_b)
        state_fc2 = layers.batch_norm(state_fc2, decay=0.9)

        state_fco = tf.nn.relu(tf.matmul(state_fc2, fco_w) + fco_b) 
        state_fco = layers.batch_norm(state_fco, decay=0.9)

        # cnn state transition
        state_m2_h1 = tf.nn.relu(tf.nn.conv2d(state_n, m2_w0, strides=(1, 1, 1, 1), padding='SAME') + m2_b0)
        state_m2_h1 = layers.batch_norm(state_m2_h1, decay=0.9)

        state_m2_ns = tf.nn.relu(tf.nn.conv2d(state_m2_h1, m2_w1, strides=(1, 1, 1, 1), padding='SAME') + m2_b1)
        state_m2_ns = layers.batch_norm(state_m2_ns, decay=0.9)

        # combine fc and cnn state transition
        state_fc_ns = tf.reshape(state_fco, shape=[-1, int(state_m2_ns.shape[1]), int(state_m2_ns.shape[2]), int(state_m2_ns.shape[3])]) # change fc layer back to a map
        state_m2_ns = state_m2_ns + state_fc_ns

        flat_state_m2_h1 = layers.flatten(state_m2_h1)
        flat_state_m2_ns = layers.flatten(state_m2_ns)

        # reward
        reward_n = tf.nn.relu(tf.matmul(flat_state_m2_h1, reward_w0) +reward_b0)
        reward_n = tf.matmul(reward_n, reward_w1) +reward_b1

        # gamma
        gamma_n = tf.nn.relu(tf.matmul(flat_state_m2_h1, gamma_w0) +gamma_b0)
        gamma_n = tf.nn.sigmoid(tf.matmul(gamma_n, gamma_w1) +gamma_b1)

        # value
        value_n = tf.nn.relu(tf.matmul(flat_state_m2_ns, value_w0) + value_b0)
        value_n = tf.matmul(value_n, value_w1) + value_b1

        # labmda
        lambda_n = tf.nn.relu(tf.matmul(flat_state_m2_h1, lambda_w0) +lambda_b0)
        lambda_n = tf.nn.sigmoid(tf.matmul(lambda_n, lambda_w1) +lambda_b1)

        # select action = argmaxQ(s,a)
        q_n = reward_n + gamma_n * value_n
        # print "q_n: ", q_n
        Act = tf.cast(tf.argmax(q_n, axis=1), tf.int32)
        # print Act
        idx = tf.stack([tf.range(0, tf.shape(Act)[0]), Act], axis=1)
        # print idx

        # select next state
        # print state_m2_nss
        # state_m2_ns = (?, dim1, dim2, ch_latent_actions * ch_state)
        # state_nt = (?, ch_latent_actions * ch_state, dim1, dim2)
        state_nt = tf.transpose(state_m2_ns, [0,3,1,2])
        # state_nt = (? , ch_latent_actions, ch_state, dim1, dim2)
        # print "state nt:", state_nt
        state_nt = tf.reshape(state_nt, shape=[-1, ch_latent_actions, ch_state, int(state_nt.shape[2]), int(state_nt.shape[3])])
        # print "state nt:", state_nt
        state_nt = tf.gather_nd(state_nt, idx)
        state_nt = tf.transpose(state_nt, [0,2,3,1])


        if skip:
            state_n = tf.nn.relu(state_n+state_nt)
        else:
            state_n = state_nt

        # mask next state rewards gammas values
        mask = tf.expand_dims(tf.one_hot(j, depth=k), axis=0)

        # print "mask: ", mask
        # print "gather nd: ",tf.expand_dims(tf.gather_nd(gamma_n, idx),axis=1)

        gammas += mask * tf.expand_dims(tf.gather_nd(gamma_n, idx),axis=1)
        rewards += mask * tf.expand_dims(tf.gather_nd(reward_n, idx),axis=1)
        values += mask * tf.expand_dims(tf.gather_nd(value_n, idx),axis=1)
        lambdas += mask * tf.expand_dims(tf.gather_nd(lambda_n, idx),axis=1)

    # g lambda
    g_lambda = zeros
    g_lambda = values[:, j]     # t = k
    for j in reversed(range(k-1)):
        # a backward pass to calculate return g lambda
        g_lambda = (1 - lambdas[:, j]) * values[:, j] + lambdas[:, j] * (rewards[:, j] + gammas[:, j] * g_lambda)


    g_lambda = tf.reshape(g_lambda, shape=[-1, a_dim])
    q_a += g_lambda


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
