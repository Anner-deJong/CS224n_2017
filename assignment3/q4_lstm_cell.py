#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extra: LSTM
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

import tensorflow as tf
import numpy as np

logger = logging.getLogger("hw3.q3.1")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class LSTMCell(tf.contrib.rnn.LSTMCell): # adjust this for tensorflow V 1.0.1 (possibly earlier on as well)
    """Wrapper around our LSTM cell implementation that allows us to play
    nicely with TensorFlow.
    """
    def __init__(self, input_size, state_size):
        self.input_size = input_size
        self._state_size = state_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, output=None, scope=None):
        """Updates the state using the previous @state and @inputs.
        Remember the GRU equations are:
        # W and U are used somewhat confusingly between code and lecture nodes, take care!
        # I defined state as c, output as h (sizes are the same)

        i_t = sigmoid(x_t U_i + h_{t-1} W_i + b_i)
        f_t = sigmoid(x_t U_f + h_{t-1} W_f + b_f)
        o_t = sigmoid(x_t U_o + h_{t-1} W_o + b_o)
        c_~ = tanh   (x_t U_c + h_{t-1} W_c + b_c)

        c_t = f_t * c_t-1 + i_t * c_~
        h_t = o_t * tanh(c_t)

        TODO: In the code below, implement an GRU cell using @inputs
        (x_t above) and the state (h_{t-1} above).
            - Define W_r, U_r, b_r, W_z, U_z, b_z and W_o, U_o, b_o to
              be variables of the apporiate shape using the
              `tf.get_variable' functions.
            - Compute z, r, o and @new_state (h_t) defined above
        Tips:
            - Remember to initialize your matrices using the xavier
              initialization as before.
        Args:
            inputs: is the input vector of size [None, self.input_size]
            state: is the previous state vector of size [None, self.state_size]
            scope: is the name of the scope to be used when defining the variables inside.
        Returns:
            a pair of the output vector and the new state vector.
        """
        scope = scope or type(self).__name__
        if (output == None): output = state
        # It's always a good idea to scope variables in functions lest they
        # be defined elsewhere!
        with tf.variable_scope(scope):
            ### YOUR CODE HERE (~20-30 lines)

            U_i = tf.get_variable('U_i', initializer=tf.contrib.layers.xavier_initializer(), shape=(self.input_size, self.state_size))
            W_i = tf.get_variable('W_i', initializer=tf.contrib.layers.xavier_initializer(), shape=(self.state_size, self.state_size))
            b_i = tf.get_variable('b_i', initializer=tf.contrib.layers.xavier_initializer(), shape=(self.state_size))

            U_f = tf.get_variable('U_f', initializer=tf.contrib.layers.xavier_initializer(), shape=(self.input_size, self.state_size))
            W_f = tf.get_variable('W_f', initializer=tf.contrib.layers.xavier_initializer(), shape=(self.state_size, self.state_size))
            b_f = tf.get_variable('b_f', initializer=tf.contrib.layers.xavier_initializer(), shape=(self.state_size))

            U_o = tf.get_variable('U_o', initializer=tf.contrib.layers.xavier_initializer(), shape=(self.input_size, self.state_size))
            W_o = tf.get_variable('W_o', initializer=tf.contrib.layers.xavier_initializer(), shape=(self.state_size, self.state_size))
            b_o = tf.get_variable('b_o', initializer=tf.contrib.layers.xavier_initializer(), shape=(self.state_size))

            U_c = tf.get_variable('U_c', initializer=tf.contrib.layers.xavier_initializer(), shape=(self.input_size, self.state_size))
            W_c = tf.get_variable('W_c', initializer=tf.contrib.layers.xavier_initializer(), shape=(self.state_size, self.state_size))
            b_c = tf.get_variable('b_c', initializer=tf.contrib.layers.xavier_initializer(), shape=(self.state_size))

            i_t = tf.nn.sigmoid( tf.matmul(inputs, U_i) + tf.matmul(output, W_i) + b_i)
            f_t = tf.nn.sigmoid( tf.matmul(inputs, U_f) + tf.matmul(output, W_f) + b_f)
            o_t = tf.nn.sigmoid( tf.matmul(inputs, U_o) + tf.matmul(output, W_o) + b_o)
            c__ = tf.nn.tanh(    tf.matmul(inputs, U_c) + tf.matmul(output, W_c) + b_c)

            new_state  = tf.multiply(f_t, state) + tf.multiply(i_t, c__)
            new_output = tf.multiply(o_t, tf.nn.tanh(new_state))

            ### END YOUR CODE ###
        # For a GRU, the output and state are the same (N.B. this isn't true
        # for an LSTM, though we aren't using one of those in our
        # assignment)
        return new_output, new_state



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tests the GRU cell implemented as part of Q3 of Homework 3')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('test', help='')
    command_parser.set_defaults(func=do_test)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
