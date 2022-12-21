#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：computational_finance 
@File    ：deep_time_series.py
@IDE     ：PyCharm 
@Author  ：patrick
@Date    ：2022/4/16 10:24 
'''
import tensorflow as tf
import tensorflow_probability as tfp


class DeepAR(tf.keras.models.Model):
    """    DeepAR 模型     """
    def __init__(self, lstm_units):
        super().__init__()
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        self.dense_mu = tf.keras.layers.Dense(1)
        self.dense_sigma = tf.keras.layers.Dense(1, activation='softplus')

    def call(self, inputs, initial_state=None):
        outputs, state_h, state_c = self.lstm(inputs, initial_state=initial_state)

        mu = self.dense_mu(outputs)
        sigma = self.dense_sigma(outputs)
        state = [state_h, state_c]

        return [mu, sigma, state]


def log_gaussian_loss(mu, sigma, y_true):
    """
    Gaussian 损失函数
    """

from tensorflow.keras import Model
class LSTNet(Model):
    def __init__(self,dense_last_idx,sparse_fea_dim,
                 dense_hidden_units, dense_output_dim, dense_hidden_activation,
                 cross_layer_num, reg_w=1e-4, reg_b=1e-4,): #'sigmoid'
        super().__init__()

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        return output


if __name__ == '__main__':
    exit()
    
  
  