# This model test using batch transform to realize DQN and it's performance 
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

SUMMARY_DIR = "./summary_spaceinvader_batch_sim_dqn_718_2"

def dual_model(img_in, a_dim, scope, k=3, skip=True, reuse=False):
    print
    print "Dual_model "
    print 
    with tf.variable_scope(scope, reuse=reuse):
        state = img_in
        ch_h = 64
        ch_state_1 = 16
        ch_state_2 = 64

         # store Q(s,a) value

        # state feature extraction (Same to normal DQN)
        state_f = layers.convolution2d(state, num_outputs=32, kernel_size=8, stride=4, padding='SAME', activation_fn=tf.nn.relu)
        state_f = layers.convolution2d(state_f, num_outputs=64, kernel_size=4, stride=2, padding='SAME', activation_fn=tf.nn.relu)
        state_f = layers.convolution2d(state_f, num_outputs=64, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
        # print state_f

        # model 1 from state feature to action number of next state
        # !!!!!! Transition may be too deep

        # state_m1_h1 = layers.convolution2d(state_f, num_outputs=ch_h, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
        # state_m1_h1 = layers.batch_norm(state_m1_h1, decay=0.9)

        # state_m1_h2 = layers.convolution2d(state_m1_h1, num_outputs=ch_h, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)
        # state_m1_h2 = layers.batch_norm(state_m1_h2,decay=0.9)

        # state_m1_n = layers.convolution2d(state_m1_h2, num_outputs=ch_h, kernel_size=3, stride=1, padding='VALID', activation_fn=tf.nn.relu)
        state_m1_n = state_f
        print "1: ", state_m1_n

        state_m1_n = layers.convolution2d(state_m1_n, num_outputs=ch_state_1*a_dim, kernel_size=3, stride=1, padding='VALID', activation_fn=tf.nn.relu)
        state_m1_n = layers.batch_norm(state_m1_n, decay=0.9)
        print "2: ", state_m1_n

        # !!!!!! direct output without hidden fc layer
        reward_m1_n = layers.fully_connected(layers.flatten(state_m1_n), num_outputs=512, activation_fn=tf.nn.relu)
        # reward_m1_n = layers.flatten(state_m1_n)
        # print "2: ", reward_m1_n
        reward_m1_n = layers.fully_connected(reward_m1_n, num_outputs=a_dim, activation_fn=None)

        q_a += reward_m1_n
        # # !!!!! (exp1 )we can test here it is very much like DQN (possible diff, without a 512 hidden layer(to see if necessary)) 

        # # State from statespace m1 to state space m2 through cnn
        # # !!!!! this trainsition will mix all action-states, get rid of it. 
        # # state_m1_n = layers.convolution2d(state_m1_n, num_outputs=ch_h, kernel_size=3, stride=1, padding='VALID', activation_fn=tf.nn.relu)
        # # state_m1_n = layers.batch_norm(state_m1_n, decay=0.9)
        # # state_m1_n = layers.convolution2d(state_m1_n, num_outputs=ch_state_2*a_dim, kernel_size=3, stride=1,padding='VALID', activation_fn=tf.nn.relu)
        # # state_m1_n = layers.batch_norm(state_m1_n, decay=0.9)

        # # model 2 latent model
        # # ch_latent_actions = 8
        # # with tf.variable_scope("model2", reuse=reuse):

        # # state transition functuon with internal policy
        # m2_w0 = tf.Variable(np.random.randn(3, 3, ch_state_2, ch_h) * 0.01, dtype=tf.float32)
        # m2_b0  = tf.Variable(np.random.randn(1, 1, 1, ch_h)    * 0.01, dtype=tf.float32)

        # m2_w1 = tf.Variable(np.random.randn(3, 3, ch_h, ch_state_2) * 0.01, dtype=tf.float32)
        # m2_b1  = tf.Variable(np.random.randn(1, 1, 1, ch_state_2)    * 0.01, dtype=tf.float32)


        # # reward function
        # # dim = s_dim[0]*s_dim[1]*ch_h
        # dim = state_m1_n.shape[1]* state_m1_n.shape[2]*ch_h
        # dim2 =  state_m1_n.shape[1]* state_m1_n.shape[2]*ch_state_2

        # # !!!! try to use single layer mlp for all down(or reference to perceptorn)
        # reward_w0 = tf.Variable(np.random.randn(dim, ch_h)*0.01 , dtype=tf.float32)
        # reward_b0 = tf.Variable(tf.zeros([ch_h]), dtype=tf.float32, name="reward_b")
        # reward_w1 = tf.Variable(np.random.randn(ch_h, 1)*0.01, dtype=tf.float32)
        # reward_b1 = tf.Variable(tf.zeros([1]), dtype=tf.float32, name="reward_b")

        # # state value function
        # value_w0 = tf.Variable(np.random.randn(dim2, ch_h) * 0.01 , dtype=tf.float32)
        # value_b0 = tf.Variable(tf.zeros([ch_h]) , dtype=tf.float32, name="value_b")
        # value_w1 = tf.Variable(np.random.randn(ch_h, 1) * 0.01 , dtype=tf.float32)
        # value_b1 = tf.Variable(tf.zeros([1]) * 0.01 , dtype=tf.float32, name="value_b")

        # # gamma(discount rate)  function
        # gamma_w0 = tf.Variable(np.random.randn(dim, ch_h) * 0.01 , dtype=tf.float32)
        # gamma_b0 = tf.Variable(tf.zeros([ch_h]), dtype=tf.float32, name="gamma_b")
        # gamma_w1 = tf.Variable(np.random.randn(ch_h, 1) * 0.01 , dtype=tf.float32)
        # gamma_b1 = tf.Variable(tf.zeros([1]) * 0.01 , dtype=tf.float32, name="gamma_b")

        # # lambda(discount rate)  function
        # lambda_w0 = tf.Variable(np.random.randn(dim, ch_h) * 0.01 , dtype=tf.float32)
        # lambda_b0 = tf.Variable(tf.zeros([ch_h]), dtype=tf.float32, name="lambda_b")
        # lambda_w1 = tf.Variable(np.random.randn(ch_h, 1) * 0.01 , dtype=tf.float32)
        # lambda_b1 = tf.Variable(tf.zeros([1]), dtype=tf.float32, name="lamda_b")

        # # from (batch_size, dim1, dim2, a_dim * ch_state_1) to (a_dim * batch_size,  dim1 , dim2 , ch_state_1)
        # # (batch_size, a_dim* ch_state_1, dim1 , dim2)
        # state_n = tf.transpose(state_m1_n, [0,3,1,2])
        # # (a_dim * batch_size, ch_state_1, dim1 , dim2 )
        # state_n = tf.reshape(state_n, shape=[-1, ch_state_1, int(state_n.shape[2]), int(state_n.shape[3])])
        # # (a_dim * batch_size, dim1 , dim2, ch_state_1 )
        # state_n = tf.transpose(state_n, [0,2,3,1])

        # # gamma, rewards, value
        # zeros = 0 * tf.range(0, tf.shape(state_n)[0])
        # zeros = tf.expand_dims(zeros, 1)
        # zeros = tf.tile(zeros, [1, k])

        # zeros = tf.cast(zeros, dtype=tf.float32)
        # gammas = zeros
        # rewards = zeros
        # values = zeros
        # lambdas = zeros

        # # print "zeros: ", zeros

        # for j in range(k):
        #     # state
        #     state_m2_h1 = tf.nn.relu(tf.nn.conv2d(state_n, m2_w0, strides=(1, 1, 1, 1), padding='SAME') + m2_b0)
        #     state_m2_h1 = layers.batch_norm(state_m2_h1, decay=0.9)

        #     state_m2_ns = tf.nn.relu(tf.nn.conv2d(state_m2_h1, m2_w1, strides=(1, 1, 1, 1), padding='SAME') + m2_b1)
        #     state_m2_ns = layers.batch_norm(state_m2_ns, decay=0.9)

        #     flat_state_m2_h1 = layers.flatten(state_m2_h1)
        #     flat_state_m2_ns = layers.flatten(state_m2_ns)

        #     # reward
        #     reward_n = tf.nn.relu(tf.matmul(flat_state_m2_h1, reward_w0) +reward_b0)
        #     reward_n = tf.matmul(reward_n, reward_w1) +reward_b1

        #     # gamma
        #     gamma_n = tf.nn.relu(tf.matmul(flat_state_m2_h1, gamma_w0) +gamma_b0)
        #     gamma_n = tf.nn.sigmoid(tf.matmul(gamma_n, gamma_w1) +gamma_b1)

        #     # value
        #     value_n = tf.nn.relu(tf.matmul(flat_state_m2_ns, value_w0) + value_b0)
        #     value_n = tf.matmul(value_n, value_w1) + value_b1

        #     # labmda
        #     lambda_n = tf.nn.relu(tf.matmul(flat_state_m2_h1, lambda_w0) +lambda_b0)
        #     lambda_n = tf.nn.sigmoid(tf.matmul(lambda_n, lambda_w1) +lambda_b1)

        #     # select action = argmaxQ(s,a)
        #     q_n = reward_n + gamma_n * value_n

        #     # select next state
        #     state_nt = state_m2_ns

        #     if skip:
        #         state_n = tf.nn.relu(state_n+state_nt)
        #     else:
        #         state_n = state_nt

        #     # mask next state rewards gammas values
        #     mask = tf.expand_dims(tf.one_hot(j, depth=k), axis=0)

        #     # print "mask: ", mask
        #     # print "gather nd: ",tf.expand_dims(tf.gather_nd(gamma_n, idx),axis=1)

        #     gammas += mask * gamma_n
        #     rewards += mask * reward_n
        #     values += mask * value_n
        #     lambdas += mask * lambda_n

        # # g lambda
        # g_lambda = zeros
        # g_lambda = values[:, j]     # t = k
        # for j in reversed(range(k-1)):
        #     # a backward pass to calculate return g lambda
        #     g_lambda = (1 - lambdas[:, j]) * values[:, j] + lambdas[:, j] * (rewards[:, j] + gammas[:, j] * g_lambda)


        # g_lambda = tf.reshape(g_lambda, shape=[-1, a_dim])
        # q_a += g_lambda

    return q_a

def plan_model(img_in, a_dim, scope, k=6, skip=True, reuse=False):
    print "plan_model "
    with tf.variable_scope(scope, reuse=reuse):
        state = img_in
        print "a_dim: ", a_dim
         # store Q(s,a) value
        # q_a = tf.Variable(0, dtype=tf.float32, name="q_a", trainable=False)

        # state feature extraction (Same to normal DQN)
        state_f = layers.convolution2d(state, num_outputs=32, kernel_size=8, stride=4, padding='SAME', activation_fn=tf.nn.relu)
        state_f = layers.convolution2d(state_f, num_outputs=64, kernel_size=4, stride=2, padding='SAME', activation_fn=tf.nn.relu)
        state_f = layers.convolution2d(state_f, num_outputs=64, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)

        # model 1
        state_n = layers.convolution2d(state_f, num_outputs=64*a_dim, kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu)

        # batch trick
        # from (batch_size, dim1, dim2, a_dim * ch_state_1) to (a_dim * batch_size,  dim1 , dim2 , ch_state_1)
        # (batch_size, a_dim* ch_state_1, dim1 , dim2)
        state_n = tf.transpose(state_n, [0,3,1,2])
        # (a_dim * batch_size, ch_state_1, dim1 , dim2 )
        state_n = tf.reshape(state_n, shape=[-1, 64, int(state_n.shape[2]), int(state_n.shape[3])])
        # (a_dim * batch_size, dim1 , dim2, ch_state_1 )
        state_n = tf.transpose(state_n, [0,2,3,1])

        flat_state_n = layers.flatten(state_n)
        dim_flat_state_f = int(flat_state_n.shape[1])

        # define func weights
        reward_h = 256
        reward_w0 = tf.Variable(np.random.randn(dim_flat_state_f, reward_h)*0.01 , dtype=tf.float32)
        reward_b0 = tf.Variable(tf.zeros([reward_h]), dtype=tf.float32)
        reward_w1 = tf.Variable(np.random.randn(reward_h, 1)*0.01, dtype=tf.float32)
        reward_b1 = tf.Variable(tf.zeros([1]), dtype=tf.float32)

        # state value function
        value_h = 256
        value_w0 = tf.Variable(np.random.randn(dim_flat_state_f, value_h) * 0.01 , dtype=tf.float32)
        value_b0 = tf.Variable(tf.zeros([value_h]) , dtype=tf.float32, name="value_b")
        value_w1 = tf.Variable(np.random.randn(value_h, 1) * 0.01 , dtype=tf.float32)
        value_b1 = tf.Variable(tf.zeros([1]) * 0.01 , dtype=tf.float32, name="value_b")

        # gamma(discount rate)  function
        gamma_h = 32
        gamma_w0 = tf.Variable(np.random.randn(dim_flat_state_f, gamma_h) * 0.01 , dtype=tf.float32)
        gamma_b0 = tf.Variable(tf.zeros([gamma_h]), dtype=tf.float32, name="gamma_b")
        gamma_w1 = tf.Variable(np.random.randn(gamma_h, 1) * 0.01 , dtype=tf.float32)
        gamma_b1 = tf.Variable(tf.zeros([1]) * 0.01 , dtype=tf.float32, name="gamma_b")

        # lambda(discount rate)  function
        lambda_h = 32
        lambda_w0 = tf.Variable(np.random.randn(dim_flat_state_f, lambda_h) * 0.01 , dtype=tf.float32)
        lambda_b0 = tf.Variable(tf.zeros([lambda_h]), dtype=tf.float32, name="lambda_b")
        lambda_w1 = tf.Variable(np.random.randn(lambda_h, 1) * 0.01 , dtype=tf.float32)
        lambda_b1 = tf.Variable(tf.zeros([1]) * 0.01 , dtype=tf.float32, name="lambda_b")

        # # define state transition functuon with internal policy
        state_h = 64 # same to last layer of state_f
        model_h = 64
        model_w0 = tf.Variable(np.random.randn(3, 3, state_h, model_h) * 0.01, dtype=tf.float32)
        model_b0  = tf.Variable(np.random.randn(1, 1, 1, model_h)    * 0.01, dtype=tf.float32)

        model_w1 = tf.Variable(np.random.randn(3, 3, model_h, model_h) * 0.01, dtype=tf.float32)
        model_b1  = tf.Variable(np.random.randn(1, 1, 1, model_h)    * 0.01, dtype=tf.float32)

        model_w2 = tf.Variable(np.random.randn(3, 3, model_h, state_h) * 0.01, dtype=tf.float32)
        model_b2  = tf.Variable(np.random.randn(1, 1, 1, state_h)    * 0.01, dtype=tf.float32)

        # layer implementation
        reward_n = tf.nn.relu(tf.matmul(flat_state_n, reward_w0) + reward_b0)
        reward_n = tf.matmul(reward_n, reward_w1) +reward_b1

        gamma_nn = tf.nn.relu(tf.matmul(flat_state_n, gamma_w0) + gamma_b0)
        gamma_nn = tf.nn.sigmoid(tf.matmul(gamma_nn, gamma_w1) + gamma_b1)

        gamma_acc = gamma_nn

        q_a = reward_n

        state_n = state_f

        print "out: ", out
        out = tf.reshape(out, shape=[-1, a_dim])

    return out

def atari_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    print "atari_model"
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
        q_func=plan_model,
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
    task = benchmark.tasks[6]
    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)
    session = get_session()
    print "max time steps:", task.max_timesteps
    atari_learn(env, session, num_timesteps=task.max_timesteps)

if __name__ == "__main__":
    main()
