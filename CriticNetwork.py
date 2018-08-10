import numpy as np

import tensorflow as tf
# from tensorflow import keras
from keras.layers import Input, Dense, merge
from keras.models import Sequential, Model
from keras.initializations import normal
from keras.optimizers import Adam

import keras.backend as K

class CriticNetwork( object):
    def __init__( self, sess, state_dim, action_dim, batch_size, tau, lr,
        hidden1=500, hidden2=1000):

        self.sess = sess
        self.batch_size = batch_size
        self.tau = tau
        self.lr = lr

        K.set_session( sess)

        print( "### DEBUG: Building Critic network ###")
        self.model, self.action, self.state = self.create_network(
            state_dim, action_dim, hidden1, hidden2)
        self.target_model, self.target_action, self.target_state = \
            self.create_network( state_dim, action_dim, hidden1,
                hidden2)
        print( "### DEBUG: Critic network built ###")

        # DEBUG: Meaning: auto compute gradients ? Which method ?
        self.action_grads = tf.gradients( self.model.output, self.action)
        self.sess.run( tf.initialize_all_variables())

    def gradients( self, states, actions):
        return self.sess.run( self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train( self):
        # DEBUG: What form ? 1xN vector ? matrix
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()

        # DEBUG: Is this some kind of discrimination ?
        for i in range( len( critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] \
                + ( 1 - self.tau) * critic_target_weights[i]

        # Updating critic weights to
        self.target_model.set_weights( critic_target_weights)

    def create_network( self, state_dim, action_dim, hidden1, hidden2):
        S = Input( shape=[state_dim])
        A = Input( shape=[action_dim], name="action")
        w1 = Dense( hidden1, activation="relu")(S)
        a1 = Dense( hidden2, activation="linear")(A)
        h1 = Dense( hidden2, activation="linear")(w1)
        h2 = merge( [h1,a1], mode="sum")
        h3 = Dense( hidden2, activation="relu")(h2)
        V = Dense( action_dim, activation="linear")(h3)

        model = Model( input=[S,A], output=V)
        adam = Adam( lr=self.lr)
        model.compile( loss="mse", optimizer=adam)

        return model, A, S
