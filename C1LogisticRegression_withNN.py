"""
1-Packages
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

"""
2-Overview of the problem set
"""

# Loading the data (cat/non-cat)
train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes = load_dataset()
  ## Added "_orig" because we are going to preprocess them.

# Example of a picture
index = 101
plt.imshow(train_set_x_orig[index])
print("y="+str(train_set_y[:,index])+",it's a '"+classes[np.squeeze(train_set_y[:,index])].decode("utf-8") + "'picture.")

# See the size of original data
m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]

 #print("Number of training examples: m_train = "+str(m_train))
 #print("Number of testing examples: m_test = "+str(m_test))
 #print("Height/Width of each image: num_px = "+str(num_px))
 #print("Each image is of size: ("+str(num_px)+", "+str(num_px)+", 3)")
 #print("train_set_x shape: "+str(train_set_x_orig.shape))
 #print("train_set_y.shape: "+str(train_set_y.shape))
 #print("test_set_x shape: "+str(test_set_x_orig.shape))
 #print("test_set_y shape: "+str(test_set_y.shape))

# Flatten the dataset
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

 #print("train_set_x_flatten shape: "+str(train_set_x_flatten.shape))
 #print("train_set_y.shape: "+str(train_set_y.shape))
 #print("test_set_x_flatten shape: "+str(test_set_x_flatten.shape))
 #print("test_set_y shape: "+str(test_set_y.shape))
 #print("sanity check after reshaping: "+str(train_set_x_flatten[0:5,0]))

# Standardize the dataset
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255



"""
3-General Architecture of the learning algorithm
  Steps:
  a. Initialize the parameters of the model
  b. Learn the parameters for the model by minimizing the cost
  c. Use the learned parameters to make predictions (on the test set)
  d. Analyse the results and conclude
"""

"""
4-Building the parts of our algorithm
"""
#4-1 Hepler functions
def sigmoid(z):
    """
    Compute the sigmoid of z
    :param z:
    :return:
    s -- sigmoid(z)
    """
    s = 1 / ( 1 + np.exp(-z) )
    return s
 #test
  #print("Sigmoid(0) = "+str(sigmoid(0)))
  #print("Sigmoid(9.2) = "+str(sigmoid(9.2)))

#4-2 Initializing parameters
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape(dim,1) for w and initialize b to 0.

    :param dim: size of the w vector we want
    :return:
    w -- initialized vector of shape(dim,1)
    b -- initialized scalar (corresponds to the bias)
    """
    b = 0
    w = np.zeros(shape=(dim,1))

    assert(w.shape == (dim,1))
    assert(isinstance(b,float) or isinstance(b,int))
    return w,b
 #test
  #dim = 2
  #w,b = initialize_with_zeros(dim)
  #print("w="+str(w))
  #print("b="+str(b))

#4-3 Forward and Backward propagation
def propagate(w,b,X,Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    :param w: weights, a numpy array of size(num_px*num_px*3,1)
    :param b: bias, a scalar
    :param X: data of size (num_px * num_px * 3,number of examples)
    :param Y: true "label" vector size(1,number of examples)
    :return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w,thus same shape as w
    db -- gradient of the loss with respect to b,thus same shape s b
    """
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    cost = (np.dot(Y,np.log(A).T)+np.dot(1-Y,np.log(1-A).T))*(-1/m)
    dz = A - Y
    dw = (np.dot(X,dz.T))/m
    db = (np.sum(dz))/m

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {
        "dw":dw,
        "db":db
    }
    return grads,cost

 #test
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
grads, cost = propagate(w, b, X, Y)
  #print ("dw = " + str(grads["dw"]))
  #print ("db = " + str(grads["db"]))
  #print ("cost = " + str(cost))

#4-4 Optimation
def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    :param w: weights
    :param b: bias
    :param X:
    :param Y:
    :param num_iterations:
    :param learning_rate:
    :param print_cost:
    :return:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs during the optization
    """
    costs = []
    for i in range(num_iterations):
        grads_,cost_ = propagate(w,b,X,Y)
        db = grads_["db"]
        dw = grads_["dw"]
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i%100 == 0:
            costs.append(cost_)
        if print_cost and i%100 ==0:
            print("Cost after iteration %i: %f" %(i,cost))
    params = {
        "w":w,
        "b":b
    }
    grads = {
        "dw":dw,
        "db":db
    }
    return params,grads,costs

 #test
  #params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

  #print ("w = " + str(params["w"]))
  #print ("b = " + str(params["b"]))
  #print ("dw = " + str(grads["dw"]))
  #print ("db = " + str(grads["db"]))


def predict(w,b,X):
    """
    Predict whether the labels is 0 or 1 using learned logistic regression parameters(w,b)

    :param w:
    :param b:
    :param X:
    :return:
    """
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)

    A = sigmoid(np.dot(w.T,X)+b)

    for i in range(A.shape[1]):
        if A[0,i]>0.5:
            Y_prediction[0,i] = 1
        else :
            Y_prediction[0,i] = 0

    assert(Y_prediction.shape == (1,m))
    return Y_prediction


"""
5- Merge all functions into a model
"""


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """
    w,b = initialize_with_zeros(X_train.shape[0])

    parameters,grads,costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = False)

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


"""
6- Further analysis
"""
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()









