import numpy as np
import math

import tensorflow as tf
# from tensorflow import keras
from keras.initializations import normal, identity
from keras.layers import Input, Dense, merge
from keras.models import Sequential, Model
from keras.optimizers import Adam

import keras.backend as K

class ActorNetwork( object):
    def __init__( self, sess, state_dim, action_dim,
        batch_size, tau, lr, hidden1=500, hidden2=1000):

        self.sess = sess
        self.batch_size = batch_size
        self.tau = tau
        self.lr = lr

        K.set_session( sess)

        #Creating actor model
        print( "### DEBUG: Building actor model ###")
        self.model, self.weights, self.state = \
            self.create_network( state_dim, action_dim, hidden1, hidden2)
        #Create Target Actor Model
        self.target_model, self.target_weights, self.target_state = \
            self.create_network( state_dim, action_dim, hidden1, hidden2)
        print("### DEBUG: Actor Model built ###")

        ## REVIEW: Why ?
        self.action_gradient = tf.placeholder( tf.float32, [ None, action_dim])
        self.params_grad = tf.gradients( self.model.output, self.weights,
            - self.action_gradient)
        grads = zip( self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer( lr).apply_gradients( grads)
        self.sess.run( tf.initialize_all_variables())

    def train( self, states, action_grads):
        self.sess.run( self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train( self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()

        for i in range( len( actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] \
                + ( 1 - self.tau) * actor_target_weights[i]

        self.target_model.set_weights( actor_target_weights)

    def create_network( self, state_dim, action_dim, hidden1, hidden2):
        S = Input( shape=[state_dim])
        h1 = Dense( hidden1)(S)
        h2 = Dense( hidden2)(h1)
        steer = Dense( 1, activation="tanh",
            init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h2)
        accel = Dense( 1, activation="sigmoid",
            init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h2)
        V = merge([steer, accel], mode="concat")
        model = Model( input=S, output=V)

        return model, model.trainable_weights, S
