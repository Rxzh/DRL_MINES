'''
MIT License

Copyright (c) 2021 Junyoeb Baek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import GlorotNormal
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Input, Dense, Concatenate, Activation, BatchNormalization


class CriticNet():
	""" Critic Network for TD3
	"""
	def __init__(self, in_dim, out_dim, lr_, tau_, discount_factor):
		self.obs_dim = in_dim
		self.act_dim = out_dim
		self.lr = lr_; self.discount_factor=discount_factor;self.tau = tau_

		# initialize critic network and target
		self.network_1,self.network_2 = self.create_network(), self.create_network()
		self.target_network_1, self.target_network_2 = self.create_network(), self.create_network()

		self.optimizer1, self.optimizer2 = Adam(self.lr), Adam(self.lr)

		# copy the weights for initialization
		weights_ = self.network_1.get_weights(), self.network_2.get_weights()
		self.target_network_1.set_weights(weights_[0]); self.target_network_2.set_weights(weights_[1])

		self.critic_loss = None

	def create_network(self):
		""" Create a Critic Network Model using Keras
			as a Q-value approximator function
		"""
		# input layer(observations and actions)
		input_obs = Input(shape=self.obs_dim)
		input_act = Input(shape=(self.act_dim,))
		inputs = [input_obs,input_act]
		concat = Concatenate(axis=-1)(inputs)

		# hidden layer 1
		h1_ = Dense(24, kernel_initializer=GlorotNormal(), kernel_regularizer=l2(0.01))(concat)
		h1_b = BatchNormalization()(h1_)
		h1 = Activation('relu')(h1_b)

		# hidden_layer 2
		h2_ = Dense(16, kernel_initializer=GlorotNormal(), kernel_regularizer=l2(0.01))(h1)
		h2_b = BatchNormalization()(h2_)
		h2 = Activation('relu')(h2_b)

		# output layer(actions)
		output_ = Dense(1, kernel_initializer=GlorotNormal(), kernel_regularizer=l2(0.01))(h2)
		output_b = BatchNormalization()(output_)
		output = Activation('linear')(output_b)

		return Model(inputs,output)

	def train(self, obs, acts, target):
		"""Train Q-network for critic on sampled batch
		"""
		with tf.GradientTape() as tape1:
			q1_values = self.network_1([obs, acts], training=True)
			critic_loss_1 = tf.reduce_mean(tf.math.square(q1_values - target))
		critic_grad_1 = tape1.gradient(critic_loss_1, self.network_1.trainable_variables)  # compute critic gradient
		self.optimizer1.apply_gradients(zip(critic_grad_1, self.network_1.trainable_variables))
		
		with tf.GradientTape() as tape2:
			q2_values = self.network_2([obs, acts], training=True)
			critic_loss_2 = tf.reduce_mean(tf.math.square(q2_values - target))
		critic_grad_2 = tape2.gradient(critic_loss_2, self.network_2.trainable_variables)  # compute critic gradient
		
		self.optimizer2.apply_gradients(zip(critic_grad_2, self.network_2.trainable_variables))

		#tf.print("critic loss :",critic_loss_1,critic_loss_2)
		self.critic_loss = float(min(critic_loss_1,critic_loss_2))
		# tf.print("critic loss :",critic_loss_1)
		# self.critic_loss = float(critic_loss_1)

	def target_update(self):
		""" soft target update for training target critic network
		"""
		weights, weights_t = self.network_1.get_weights(), self.target_network_1.get_weights()
		for i in range(len(weights)):
			weights_t[i] = self.tau*weights[i] + (1-self.tau)*weights_t[i]
		self.target_network_1.set_weights(weights_t)

		weights, weights_t = self.network_2.get_weights(), self.target_network_2.get_weights()
		for i in range(len(weights)):
			weights_t[i] = self.tau*weights[i] + (1-self.tau)*weights_t[i]
		self.target_network_2.set_weights(weights_t)

	def save_network(self, path):
		self.network_1.save_weights(path + '_critic1.h5')
		self.target_network_1.save_weights(path + '_critic1_t.h5')
		self.network_2.save_weights(path + '_critic2.h5')
		self.target_network_2.save_weights(path + '_critic2_t.h5')

	def load_network(self, path):
		self.network_1.load_weights(path + '_critic1.h5')
		self.target_network_1.load_weights(path + '_critic1_t.h5')
		self.network_2.load_weights(path + '_critic2.h5')
		self.target_network_2.load_weights(path + '_critic2_t.h5')
