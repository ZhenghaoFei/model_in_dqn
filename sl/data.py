import numpy as np
import scipy.io as sio

# def process_gridworld_data(input, imsize):
#     # run training from input matlab data file, and save test data prediction in output file
#     # load data from Matlab file, including
#     # im_data: flattened images
#     # state_data: concatenated one-hot vectors for each state variable
#     # state_xy_data: state variable (x,y position)
#     # label_data: one-hot vector for action (state difference)
#     # X: [size, size, 2], first channel encodes obstcle, second channel encodes goal position
#     # S1, x of current position
#     # S2, y of current position
#     im_size=[imsize, imsize]
#     matlab_data = sio.loadmat(input)
#     im_data = matlab_data["batch_im_data"]
#     im_data = (im_data - 1)/255  # obstacles = 1, free zone = 0
#     value_data = matlab_data["batch_value_data"]
#     state1_data = matlab_data["state_x_data"]
#     state2_data = matlab_data["state_y_data"]
#     label_data = matlab_data["batch_label_data"]
#     ydata = label_data.astype('int8')
#     Xim_data = im_data.astype('float32')
#     Xim_data = Xim_data.reshape(-1, 1, im_size[0], im_size[1])
#     Xval_data = value_data.astype('float32')
#     Xval_data = Xval_data.reshape(-1, 1, im_size[0], im_size[1])
#     Xdata = np.append(Xim_data, Xval_data, axis=1)
#     # Need to transpose because Theano is NCHW, while TensorFlow is NHWC
#     Xdata = np.transpose(Xdata,  (0, 2, 3, 1))
#     S1data = state1_data.astype('int8')
#     S2data = state2_data.astype('int8')

#     all_training_samples = int(6/7.0*Xdata.shape[0])
#     training_samples = all_training_samples
#     Xtrain = Xdata[0:training_samples]
#     S1train = S1data[0:training_samples]
#     S2train = S2data[0:training_samples]
#     ytrain = ydata[0:training_samples]

#     Xtest = Xdata[all_training_samples:]
#     S1test = S1data[all_training_samples:]
#     S2test = S2data[all_training_samples:]
#     ytest = ydata[all_training_samples:]
#     ytest = ytest.flatten()

#     sortinds = np.random.permutation(training_samples)
#     Xtrain = Xtrain[sortinds]
#     S1train = S1train[sortinds]
#     S2train = S2train[sortinds]
#     ytrain = ytrain[sortinds]
#     ytrain = ytrain.flatten()
#     return Xtrain, S1train, S2train, ytrain, Xtest, S1test, S2test, ytest

def process_gridworld_data(input, imsize):
    # run training from input matlab data file, and save test data prediction in output file
    # load data from Matlab file, including
    # im_data: flattened images
    # state_data: concatenated one-hot vectors for each state variable
    # state_xy_data: state variable (x,y position)
    # label_data: one-hot vector for action (state difference)
    # X: [size, size, 2], first channel encodes obstcle, second channel encodes goal position
    # S: [step, size, size, 1],  current position, have 10 steps for each X

    im_size=[imsize, imsize]
    steps = imsize + 2
    matlab_data = sio.loadmat(input)
    im_data = matlab_data["batch_im_data"]
    im_data = (im_data - 1)/255  # obstacles = 1, free zone = 0
    value_data = matlab_data["batch_value_data"]
    state1_data = matlab_data["state_x_data"]
    state2_data = matlab_data["state_y_data"]
    label_data = matlab_data["batch_label_data"]
    ydata = label_data.astype('int8')
    Xim_data = im_data.astype('float32')
    Xim_data = Xim_data.reshape(-1, im_size[0], im_size[1], 1)
    Xval_data = value_data.astype('float32')
    Xval_data = Xval_data.reshape(-1, im_size[0], im_size[1], 1)
    Xdata = np.concatenate((Xim_data, Xval_data), axis=3)
    Xdata_all = np.zeros([Xval_data.shape[0], steps, Xval_data.shape[1], Xval_data.shape[2], 2])

    S1data = state1_data.astype('int8')
    S2data = state2_data.astype('int8')
    Sdata = np.zeros([Xval_data.shape[0], steps, Xval_data.shape[1], Xval_data.shape[2], 1])
    for i in range(Sdata.shape[0]):
        for j in range(Sdata.shape[1]):
            Sdata[i, j, S1data[i,j], S2data[i,j]] = 20
            Xdata_all[i, j] = Xdata[i]
    Xdata = Xdata_all[:]


    all_training_samples = int(6/7.0*Xdata.shape[0])
    training_samples = all_training_samples
    Xtrain = Xdata[0:training_samples]
    Strain = Sdata[0:training_samples]
    ytrain = ydata[0:training_samples]

    Xtest = Xdata[all_training_samples:]
    Stest = Sdata[all_training_samples:]
    ytest = ydata[all_training_samples:]
    
    Xtest = Xtest.reshape(-1, im_size[0], im_size[1], 2)
    Stest = Stest.reshape(-1, im_size[0], im_size[1], 1)
    ytest = ytest.reshape(-1, 1)

    sortinds = np.random.permutation(training_samples)
    Xtrain = Xtrain[sortinds]
    Strain = Strain[sortinds]
    ytrain = ytrain[sortinds]
    Xtrain = Xtrain.reshape(-1, im_size[0], im_size[1], 2)
    Strain = Strain.reshape(-1, im_size[0], im_size[1], 1)
    ytrain = ytrain.reshape(-1, 1)

    return Xtrain, Strain, ytrain, Xtest, Stest, ytest

