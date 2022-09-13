#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：computational_finance 
@File    ：lstnet.py
@IDE     ：PyCharm 
@Author  ：patrick
@Date    ：2022/4/17 2:45 
'''
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Conv1D, GRU, Dropout, Flatten, Activation,Layer
from tensorflow.keras.layers import concatenate, add, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from tensorflow.keras import Model
# class LSTNet( Model ): #object
#     def __init__(self,step,dim,hidCNN=100,hidRNN=100,
#                  hidSkip=10,CNN_kernel=6,skip=3,highway_window=3,dropout=0.2,
#                  activation='sigmoid',lr=0.01,clip=1,loss='mae' ):
#         super(LSTNet, self).__init__()
#         self.P = step
#         self.m = dim
#         # self.P    = window      #args.window     # default=24*7
#         self.hidC   = hidCNN   #'number of CNN hidden units'
#         self.hidR   = hidRNN  # args.hidRNN   #'number of RNN hidden units'
#         self.hidS   = hidSkip#args.hidSkip  #default=10
#         self.Ck     = CNN_kernel      #args.CNN_kernel #'the kernel size of the CNN layers'
#         self.skip   = skip          # default=24, help='period'
#         self.hw     = highway_window                  #default=3, help='The window size of the highway component')
#         self.dropout= dropout                    #default=0.2, help='dropout applied to layers (0 = no dropout)'
#         self.output_fun =  activation             #args.output_fun     # default='sigmoid'
#         self.lr     =  lr                              #0.01
#         self.loss   = loss#args.loss                #default='mae'
#         self.clip   = clip#args.clip                          #default=10., help='gradient clipping'
#         self.pt = int((self.P - self.Ck) / self.skip)
#         self.model = None
    # def build(self, input_shape):
    #     self.model  = make_model( input_shape[1],input_shape[2] )
    #     self.built = True  # 最后这句话一定要加上 #super(MyLayer, self).build(input_shape)  # 一定要在最后调用它

    # def call(self, inputs, **kwargs):
    #     ret = self.model( inputs )
    #     return ret

def  LSTNet_model(step,dim,hidCNN=100,hidRNN=100,
                hidSkip=10,CNN_kernel=6,skip=3,highway_window=3,dropout=0.2,
                activation='sigmoid',lr=0.01,clip=1,loss='mae',if_compile=True):
    P    = step
    m    = dim
    hidC = hidCNN
    Ck   = CNN_kernel
    hidR = hidRNN
    hidS = hidSkip
    hw   = highway_window
    output_fun = activation

    pt = int((step- CNN_kernel) / skip)
    x =Input( shape=(step, dim) ) # inputs
    # CNN
    c = Conv1D( hidC,  Ck, activation='relu')(x)
    c = Dropout( dropout)(c)
    # RNN
    r = GRU( hidR)(c)
    r = Lambda(  lambda k: K.reshape(k, (-1, hidR)))(r)
    r = Dropout( dropout)(r)
    # skip-RNN
    if skip > 0:
        # c: batch_size*steps*filters, steps=P-Ck
        s = Lambda(lambda k: k[:, int(-pt *  skip):, :])(c)
        s = Lambda(lambda k: K.reshape(k, (-1, pt,  skip,  hidC)))(s)
        s = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1, 3))  )(s)
        s = Lambda(lambda k: K.reshape(k, (-1,  pt, hidC)) )(s)
        s = GRU( hidS)( s )
        s = Lambda(lambda k: K.reshape(k, (-1, skip * hidS)))(s)
        s = Dropout(dropout)(s)
        r = concatenate([r, s])
    res = Dense(m)(r)
    # highway
    if hw > 0:
        z = Lambda(lambda k: k[:, -hw:, :])(x)
        z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
        z = Lambda(lambda k: K.reshape(k, (-1, hw)) )(z)
        z = Dense(1)(z)
        z = Lambda(lambda k: K.reshape(k, (-1, m)) )(z)
        res = add([res, z])

    if output_fun != 'no':
        res = Activation( output_fun)(res)
    model   = Model(inputs=x, outputs=res)
    if if_compile: model.compile(optimizer=Adam(lr= lr, clipnorm= clip), loss=loss  )
    return model

    # def get_config(self):
    #     """
    #     Returns the config of a the layer. This is used for saving and loading from a model
    #     :return: python dictionary with specs to rebuild layer """
    #     config = super(LSTNet, self).get_config()
    #     return config

class LSTNet_multi_inputs(Model):
    def __init__(self,input_dims ,window=24,hidCNN=100,hidRNN=100,
                 hidSkip=10,CNN_kernel=6,skip=24,highway_window=3,dropout=0.2,
                 output_fun='sigmoid',lr=0.01,loss='mae',clip=10,ps=3):
        super(LSTNet_multi_inputs, self).__init__()
        self.P      = window  # args.window     # default=24*7
        self.m      = input_dims  # dims
        self.hidC   = args.hidCNN  # 'number of CNN hidden units'
        self.hidR   = hidRNN  # args.hidRNN   #'number of RNN hidden units'
        self.hidS   = hidSkip  # args.hidSkip  #default=10
        self.Ck     = CNN_kernel  # args.CNN_kernel #'the kernel size of the CNN layers'
        self.skip   = skip  # default=24, help='period'
        # self.pt = int((self.P-self.Ck)/self.skip)
        #self.pt     = ps # default=3, help='number of skip (periods)'
        self.pt     = int((self.P - self.Ck) / self.skip)  #
        self.hw     = highway_window  # default=3, help='The window size of the highway component')
        self.dropout= dropout  # default=0.2, help='dropout applied to layers (0 = no dropout)'
        #self.output_fun =output_fun   # args.output_fun     # default='sigmoid'
        self.output = output_fun
        self.lr     = lr  # 0.01
        self.loss   = loss  # args.loss                #default='mae'
        self.clip   = clip  # args.clip                          #default=10., help='gradient clipping'

    def make_model(self): #*args, **kwargs
        # Input1: short-term time series
        #input1 = Input(shape=(self.P, self.m))
        # CNN
        x = Input(shape=(self.P, self.m))  # inputs
        conv1 = Conv1D(self.hidC, self.Ck, strides=1, activation='relu')  # for input1
        # It's a probelm that I can't find any way to use the same Conv1D layer to train the two inputs,
        # since input2's strides should be Ck, not 1 as input1
        conv2 = Conv1D(self.hidC, self.Ck, strides=self.Ck, activation='relu')  # for input2
        conv2.set_weights(conv1.get_weights())  # at least use same weight

        c1 = conv1(input1)
        c1 = Dropout(self.dropout)(c1)
        # RNN
        r1 = GRU(self.hidR)(c1)
        # r1 = Lambda(lambda k: K.reshape(k, (-1, self.hidR)))(r1)
        r1 = Dropout(self.dropout)(r1)

        # Input2: long-term time series with period
        input2 = Input(shape=(self.pt * self.Ck, self.m))
        # CNN
        c2 = conv2(input2)
        c2 = Dropout(self.dropout)(c2)
        # RNN
        r2 = GRU(self.hidS)(c2)
        # r2 = Lambda(lambda k: K.reshape(k, (-1, self.hidR)))(r2)
        r2 = Dropout(self.dropout)(r2)

        r = concatenate([r1, r2])
        res = Dense(self.m)(r)

        # highway
        if self.hw > 0:
            z = Lambda(lambda k: k[:, -self.hw:, :])(input1)
            z = Lambda(lambda k: K.permute_dimensions(k, (0, 2, 1)))(z)
            z = Lambda(lambda k: K.reshape(k, (-1, self.hw)))(z)
            z = Dense(1)(z)
            z = Lambda(lambda k: K.reshape(k, (-1, self.m)))(z)
            res = add([res, z])

        if self.output != 'no':
            res = Activation(self.output)(res)

        model = Model(inputs=[input1, input2], outputs=res)
        model.compile(optimizer=Adam(lr=self.lr, clipnorm=self.clip), loss=self.loss)
        return model


if __name__ == '__main__':
    exit()
    
  
  