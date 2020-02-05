"""
Face Recognition for the Happy House

"""

from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *

"""
0 - Naive Face Verification
"""
# In Face Verification, you are given two images and you have to tell if they are of the same person.
# The simplest way to do this is to compare the two images pixel-by-pixel.
# If the distance between the raw images are less than a chosen threshold, it may be the same person.
# Of course, this algorithm perfomrs really poorly, since the pixel values change dramatically due to variations in lighting...

"""
1 - Encoding face images into a 128-dimensional vector
"""
# 1.1 - Using an ConvNet to compute encodings
FRmodel = faceRecoModel(input_shape = (3,96,96))

print("Total Params: ",FRmodel.count_params())

# 1.2 - The Triplet Loss
def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """
    anchor,positive,negative = y_pred[0],y_pred[1],y_pred[2]

    distance_1 = tf.reduce_sum(tf.square(tf.substract(anchor,postive)))
    distance_2 = tf.reduce_sum(tf.square(tf.substract(anchor,negative)))
    dis = tf.add(tf.substract(distance1,distance2),alpha)
6
    loss = tf.maximum(tf.reduce_mean(dis),0.0)

    return loss

"""
2 - Loading the trained model
"""

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

"""
3 - Applying the model
"""

# 3.1 Face Verification
database = {}
database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)


def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".

    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras

    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """
    encoding = img_to_encoding(image_path,model)

    dis = np.linalg.norm(encoding - database[identity])

    if dis < 0.7 :
        door_open = True
    else :
        door_open = False

    return dis,door_open


# 3.2 Face Recognition

def who_is_it(image_path, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.

    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras

    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding = img_to_encoding(image_path,model)

    ## Step 2: Find the closest encoding ##
    min_dist = 100

    for (name,db_enc) in database:
        dis = np.linalg.norm(encoding - db_enc)
        if dis < min_dist:
            min_dist = dis
            identity = name

    if min_dist > 0.7 :
        print("Not in the database.")
    else :
        print("it's "+ str(identity)+", the distance is "+str(min_dist))

    return min_dist,identity










