"""
1 Packages
"""
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_utils import *
import BuildNN_StepByStep
import lr_utils

"""
2 Dataset
"""

train_x_orig,train_y,test_x_orig,test_y,classes = lr_utils.load_dataset()

# Explore the dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

#print ("Number of training examples: " + str(m_train))
#print ("Number of testing examples: " + str(m_test))
#print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
#print ("train_x_orig shape: " + str(train_x_orig.shape))
#print ("train_y shape: " + str(train_y.shape))
#print ("test_x_orig shape: " + str(test_x_orig.shape))
#print ("test_y shape: " + str(test_y.shape))

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],-1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0],-1).T

train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

#print ("train_x's shape: " + str(train_x.shape))
#print ("test_x's shape: " + str(test_x.shape))

"""
3 Architecture of model
"""

# 3.1 2-layer neural network
# 3.2 L-layer neural network
# 3.3 General methodology
"""
  a. Initialize parameters
  b. Loop for num_iterations
     b1. Forward propagation
     b2. Compute cost function
     b3. Backward propagation
     b4. Updata parameters(using parameters, and grads from backprop)
  c. Use trained parameters to predict labels
  """

"""
4 Two-layer neural network
"""

n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)


def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    np.random.seed(1)
    grads = {}
    costs = []
    m = X.shape[1]
    (n_x,n_h,n_y) = layers_dims

    parameters = BuildNN_StepByStep.initialize_parameters(n_x,n_h,n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(num_iterations):
        A1,cache1 = BuildNN_StepByStep.linear_activation_forward(X,W1,b1,'relu')
        A2,cache2 = BuildNN_StepByStep.linear_activation_forward(A1,W2,b2,'sigmoid')
        cost = BuildNN_StepByStep.compute_cost(A2,Y)

        dA2 = -(np.divide(Y,A2) - np.divide(1-Y,1-A2))

        dA1 , dW2 , db2 = BuildNN_StepByStep.linear_activation_backward(dA2,cache2,'sigmoid')
        dA0 , dW1 , db1 = BuildNN_StepByStep.linear_activation_backward(dA1,cache1,'relu')

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        parameters = BuildNN_StepByStep.update_parameters(parameters,grads,learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

#parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=False)

def predict(X,y,parameters):
    """

    :param X: training set
    :param y: label
    :param parameters: parameters of the training set
    :return:
    p - prediction
    """
    m = X.shape[1]
    L = len(parameters) //2
    p = np.zeros((1,m))

    probas,caches = BuildNN_StepByStep.L_model_forward(X,parameters)

    for i in range(0,probas.shape[1]):
        if probas[0,i]>0.5:
            p[0,i]=1
        else :
            p[0,i]=0

    print("Accuracy : "+str(float(np.sum(p==y))/m))

    return p

#predictions_train = predict(train_x, train_y, parameters) #训练集
#predictions_test = predict(test_x, test_y, parameters) #测试集


"""
5 L-layer Neural Network
"""

layers_dims = [12288,20,7,5,1]


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):  # lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []  # keep track of cost

    parameters = BuildNN_StepByStep.initialize_parameters_deep(layers_dims)

    for i in range(num_iterations):
        AL,caches = BuildNN_StepByStep.L_model_forward(X,parameters)
        cost = BuildNN_StepByStep.compute_cost(AL,Y)
        grads = BuildNN_StepByStep.L_model_backward(AL,Y,caches)
        parameters = BuildNN_StepByStep.update_parameters(parameters,grads,learning_rate)

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)

pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)































