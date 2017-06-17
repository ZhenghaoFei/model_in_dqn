import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import dqn
from dqn_utils import *
from atari_wrappers import *

SUMMARY_DIR = "./summary"

def dual_model_old(img_in, a_dim, scope, k=1, skip=False, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):

        state = img_in
        ch_h = 1
        # print state
        # print a_dim
        # store Q(s,a) value
        q_a = tf.Variable(0, dtype=tf.float32, name="q_a", trainable=False)

        # state feature extraction (Same to normal DQN)
        state_f = layers.convolution2d(state, num_outputs=32, kernel_size=8, stride=4, padding='SAME', activation_fn=tf.nn.relu)
        state_f = layers.batch_norm(state_f)
        state_f = layers.convolution2d(state_f, num_outputs=64, kernel_size=4, stride=2, padding='SAME', activation_fn=tf.nn.relu)
        state_f = layers.batch_norm(state_f)
        state_f = layers.convolution2d(state_f, num_outputs=64, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
        state_f = layers.batch_norm(state_f)
        # print state_f

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
        state_m1_n = layers.convolution2d(state_m1_n, num_outputs=a_dim, kernel_size=3, stride=1,padding='VALID', activation_fn=tf.nn.relu)
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


        # print state_m1_n
        for i in range(a_dim):
            # state_n = state_m1_n[:,:,:,i]
            # state_n
            # print state_n
            # print i
            state_n = tf.expand_dims(state_m1_n[:,:,:,i], 3)
            # print state_n
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
                # mask = tf.reshape(tf.one_hot(j, depth=k),[1,-1])
                mask = tf.expand_dims(tf.one_hot(j, depth=k), axis=0)
                # print "mask: ", mask
                # print "gather nd: ", tf.gather_nd(gamma_n, idx)
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

        return q_a

def dual_modelmg_in, a_dim, scope, k=5, skip=False, reuse=False):
    
    with tf.variable_scope(scope, reuse=reuse):
        state = img_in
        # print state.shape[1]
        ch_h = 1
        # print state
        # print a_dim

        # store Q(s,a) value
        q_a = tf.Variable(0, dtype=tf.float32, name="q_a", trainable=False)

        # state feature extraction (Same to normal DQN)
        state_f = layers.convolution2d(state, num_outputs=32, kernel_size=8, stride=4, padding='SAME', activation_fn=tf.nn.relu)
        state_f = layers.batch_norm(state_f)
        state_f = layers.convolution2d(state_f, num_outputs=64, kernel_size=4, stride=2, padding='SAME', activation_fn=tf.nn.relu)
        state_f = layers.batch_norm(state_f)
        state_f = layers.convolution2d(state_f, num_outputs=64, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
        state_f = layers.batch_norm(state_f)
        # print state_f

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
        state_m1_n = layers.convolution2d(state_m1_n, num_outputs=a_dim, kernel_size=3, stride=1,padding='VALID', activation_fn=tf.nn.relu)
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

        # from (batch_size, dim1, dim2, a_dim) to (a_dim * dim1 * dim2 * 1)
        state_n = tf.reshape(state_m1_n, shape=[-1, int(state_m1_n.shape[1]), int(state_m1_n.shape[2]) , 1])
        print state_n

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
            state_m2_h1 = layers.batch_norm(state_m2_h1)

            state_m2_ns = tf.nn.relu(tf.nn.conv2d(state_m2_h1, m2_w1, strides=(1, 1, 1, 1), padding='SAME') + m2_b1)
            state_m2_ns = layers.batch_norm(state_m2_ns)

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

        # q_a
        # mask = tf.reshape(tf.one_hot(i, depth=a_dim),[1,-1])
        # g_lambda = tf.expand_dims(g_lambda, 1)
        # g_lambda = tf.tile(g_lambda, [1, a_dim])

        print "g_lambda: ", g_lambda
        g_lambda = tf.reshape(g_lambda, shape=[-1, a_dim])
        print "g_lambda: ", g_lambda
        print "q_a: ", q_a

        q_a += g_lambda
        return q_a

def atari_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
            print out
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out

def atari_learn(env,
                session,
                num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    dqn.learn(
        env,
        q_func=dual_model_old,
        optimizer_spec=optimizer,
        session=session,
        summary_dir=SUMMARY_DIR,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10
    )
    env.close()

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i) 
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

def get_env(task, seed):
    env_id = task.env_id

    env = gym.make(env_id)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/hw3_vid_dir2/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)

    return env

def main():
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')
    # Change the index to select a different game.
    task = benchmark.tasks[5]
    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)
    session = get_session()
    print "max time steps:", task.max_timesteps
    atari_learn(env, session, num_timesteps=task.max_timesteps)

if __name__ == "__main__":
    main()
