x = tf.placeholder(tf.int64,name = 'x')
print(sess.run(2*x,feed_dict = {x:3}))

"""
1.1 - Linear Function
"""

def linear_function():
    np.random.seed(1)

    W = tf.constant(np.random.randn(4,3),name = 'W')
    X = tf.constant(np.random.randn(3,1),name = 'X')
    b = tf.constant(np.random.randn(4,1),name = 'b')

    Y = tf.add(tf.matmul(W,X),b)
    sess = tf.Session()
    result = sess.run(Y)
    sess.close()

    return result

"""
1.2 - Computing the sigmoid
"""

def sigmoid(z):
    x = tf.placeholder(tf.float32,name = 'x')
    y = tf.sigmoid(x)
    sess = tf.Session()
    result = sess.run(y,feed_dict = {x:z})
    sess.close()

    return result

# Steps:
  # Create placeholders
  # Specify the computation graph corresponding to operations you want to comupte
  # Create the session
  # Run the session,using a feed dictionary if necessary to specify placeholder variables' values

"""
1.3 - Computing the cost
"""

def cost(logits,labels):
    z = tf.placeholder(tf.float32,name = 'z')
    y = tf.placeholder(tf.float32,name = 'y')

    cost = tf.nn.sigmoid_cross_entropy_with_logits(z,y)

    sess = tf.Session()
    result = sess.run(cost,feed_dict = {z:logits,y:labels})
    sess.close()

    return result


"""
1.4 - Using One-Hot encodings
"""

def one_hot_matrix(labels,C):
    C = tf.constant(C,name = 'C')
    one_hot_matrix = tf.one_hot(indices = labels,depth = C,axis=0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()

    return one_hot

"""
1.5 - Initialize with zeros and ones
"""

def ones(shape):
    ones = tf.ones(shape)
    sess = tf.Session()
    ones = sess.run(ones)
    sess.close()

    return ones



"""
 
2 - Buidling your first neural network 
 
"""

# Loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0],-1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0],-1).T

# Normalizing image vectors
X_train = X_train_flatten / 255
X_test = X_test_flatten / 255

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig,6)
Y_test = convert_to_one_hot(Y_test_orig,6)

"""
2.1 - Create placeholders
"""

def create_placeholders(n_x,n_y):
    X = np.placeholder(tf.float32,[n_x,None],name = 'X')
    Y = np.placeholder(tf.float32,[n_y,None],name = 'Y')

    return X,Y

"""
2.2 - Initializing the parameters
"""

def initialize_parameters():
    W1 = tf.get_variable("W1",[25,12288],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1",[25,1],initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2",[12,25],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2",[12,1],initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3",[6,12],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3",[6,1],initializer = tf.zeros_initializer())

    parameters = {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2,
        "W3":W3,
        "b3":b3
    }

    return parameters

"""
2.3 - Forward propagation in tensorflow
"""

def forward_propagation(X,parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = tf.add(tf.matmul(W1,X),b1)
    A1 = tf.nn.relu(Z1)

    Z2 = tf.add(tf.matmul(W2,A1),b2)
    A2 = tf.nn.reulu(Z2)

    Z3 = tf.add(tf.matmul(W3,A2),b3)

    return Z3

"""
2.4 - Compute cost
"""

def compute_cost(Z3,Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = labels))

    return cost

"""
2.5 - Backward propagation & parameter update
"""

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
_ , c = sess.run([optimizer,cost],feed_dict={X:mini_batch_X,Y:mini_batch_Y})


"""
2.6 - Buidling the model
"""

def model(X_train,Y_train,X_test,Y_test,learning_rate = 0.0001,num_epochs = 1500, minibatch_size = 32, print_cost = True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x,m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    X, Y = create_placeholders(n_x,n_y)

    parameters = initializer_parameters()

    Z3 = forward_propagation(X,parameters)

    cost = compute_cost(Z3,Y)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    init = tf.global_variable_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m/minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train,Y_train,minibatch_size,seed)

            for minibatch in minibatches:
                (minibatch_X,minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer,cost],feed_dict = {X:minibatch_X,Y:minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches












