#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：kaggle 
@File    ：nlp_huggingface.py
@IDE     ：PyCharm 
@Author  ：patrick
@Date    ：2022/9/22 19:07 
'''

import pandas as pd
import numpy as np
import os
import  tensorflow as tf
from    tensorflow.keras.layers import Dense, Input, Dropout
from    tensorflow.keras.optimizers import Adam
from    tensorflow.keras.models import Model
from    transformers import TFBertModel
import  transformers


def load_tokenizer( MODEL_PATH ):
    if 'deberta' in MODEL_PATH:
        deberta = load_debert_tokenizer(MODEL_PATH)
        return  deberta
    tokenizer = transformers.BertTokenizer.from_pretrained(MODEL_PATH)
    return  tokenizer

def load_bert( MODEL_PATH ):
    if 'deberta' in MODEL_PATH:
        deberta = load_deberta(MODEL_PATH ) ###
        return  deberta
    from transformers import TFBertModel
    model = TFBertModel.from_pretrained(MODEL_PATH)
    return  model


def load_debert_tokenizer( MODEL_PATH ):
    from transformers import  DebertaV2Tokenizer #AutoTokenizer, AutoModel, AutoConfig
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_PATH)
    return  tokenizer
def load_deberta( MODEL_PATH ):
    from transformers import  TFDebertaV2Model as TFBert#TFDebertaModel  #TFAutoModel #AutoModel #TFBertModel #BertModel  #,TFBertModel
    print( 'MODEL_PATH:',MODEL_PATH )
    model = TFBert.from_pretrained(MODEL_PATH ,trainable=True) # from_pretrained(MODEL_PATH)
    return  model

def get_bert_inputs(text_list, tokenizer,lenth,MAX_TEXT ):
    ids  = np.zeros( ( lenth, MAX_TEXT) )  #X_input_ids
    masks= np.zeros( ( lenth, MAX_TEXT) )  #X_attn_masks
    from tqdm.auto import tqdm
    for i, text in tqdm(enumerate( text_list ) ):
        tokenized_text = tokenizer.encode_plus(text,max_length=MAX_TEXT,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors='tf')
        ids[  i, :] = tokenized_text.input_ids
        masks[i, :] = tokenized_text.attention_mask
    return ids, masks


def trans_to_onehot(  src_list,nclasses=3 ): #src_list must be interger
    #df['target']      = df['target'].astype(int)
    labels = np.zeros((len(src_list), nclasses) )
    labels[ np.arange(len(src_list)), src_list ] = 1
    return labels

def create_dataset( X_input_ids, X_attn_masks, labels ,shuffle_len=10000,  batchsize=16):
    def DatasetMapFunction(input_ids, attn_masks,  labels):
        return {
                   'input_ids': input_ids,  # From tokenizer, not yet converted to features.
                   'attention_mask': attn_masks,  # From tokenizer, not yet converted to features.
               }, labels  # Manually created labels.
    # Tensorflow specific, need to combine tokenizer output and label to form training dataset.
    dataset = tf.data.Dataset.from_tensor_slices( ( X_input_ids, X_attn_masks, labels) )
    dataset = dataset.map(DatasetMapFunction)  # Convert to required format for tensorflow dataset.
    dataset = dataset.shuffle(shuffle_len).batch(batchsize, drop_remainder=True)
    return dataset

def create_deberta_dataset( X_input_ids, X_attn_masks, labels ,shuffle_len=10000,  batchsize=16):
    def DatasetMapFunction(input_ids, attn_masks,  labels):
        return (input_ids, attn_masks ),labels
        # return {
        #           'input_ids': input_ids,  # From tokenizer, not yet converted to features.
        #           'attention_mask': attn_masks,  # From tokenizer, not yet converted to features.
        #           'target': labels
        #        }  # Manually created labels.
    # Tensorflow specific, need to combine tokenizer output and label to form training dataset.
    dataset = tf.data.Dataset.from_tensor_slices( ( X_input_ids, X_attn_masks, labels) )
    dataset = dataset.map(DatasetMapFunction)  # Convert to required format for tensorflow dataset.
    dataset = dataset.shuffle(shuffle_len).batch(batchsize, drop_remainder=True)
    return dataset

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
def encode_categorical_cols(df, cols_to_encode):
    print('Encoding cols: {}'.format(cols_to_encode))
    for f in cols_to_encode:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df[f].values))
        df[f] = lbl.transform(list(df[f].values))
    return df

import keras.backend as K
def getPrecision(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # TP
    pred_P = K.sum(K.round(K.clip(y_pred, 0, 1)))
    # N = (-1) * K.sum(K.round(K.clip(y_true - K.ones_like(y_true), -1, 0)))  # N
    # TN = K.sum(K.round(K.clip((y_true - K.ones_like(y_true)) * (y_pred - K.ones_like(y_pred)), 0, 1)))  # TN
    # FP = N - TN
    # precision = TP / (TP + FP + K.epsilon())  # TT/P ; TP/P
    precision =  TP / ( pred_P + 0.00001 ) #K.epsilon()
    return precision

def getRecall(y_true, y_pred):
    TP = K.sum(K.round( K.clip(y_true * y_pred, 0, 1)) )  # TP
    P = K.sum(K.round(  K.clip(y_true, 0, 1)))
    #FN = P - TP  # FN=P-TP
    recall = TP / ( P + 0.00001 )  # TP/(TP+FN)  # TP + FN
    return recall

def get_f1score(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # TP
    pred_P = K.sum(K.round(K.clip(y_pred, 0, 1)))
    P = K.sum(K.round(K.clip(y_true, 0, 1)))
    precision = TP /( pred_P + K.epsilon() )
    recall    = TP /( P + K.epsilon()      )
    return  2*precision*recall/( precision + recall )

def build_bert_model(  MODEL_PATH,inter_dense=512,nclasses=3,metrics=['accuracy']):
    bertmodel = load_bert( MODEL_PATH )  # Load model.
    # Defines Bert layer. Input layer names must be the same as defined in dataset.
    input_ids  = tf.keras.layers.Input(shape=(MAX_TEXT,), name='input_ids', dtype='int32')  # Input layer.
    attn_masks = tf.keras.layers.Input(shape=(MAX_TEXT,), name='attention_mask', dtype='int32')  # Input layer.
    bert_embds = bertmodel.bert(input_ids, attention_mask=attn_masks)[1]
    intermediate_layer = tf.keras.layers.Dense( inter_dense , activation='relu', name='intermediate_layer')(bert_embds)

    output_layer = tf.keras.layers.Dense( nclasses, activation='softmax', name='output_layer')( intermediate_layer )
    feedback_model = tf.keras.Model(inputs=[input_ids, attn_masks ], outputs=output_layer)
    feedback_model.summary()
    feedback_model.compile(optimizer=Adam(learning_rate=1e-5, decay=1e-6), loss='categorical_crossentropy', metrics=metrics)
    return  feedback_model

def build_deberta_model(  MODEL_PATH,inter_dense=512,nclasses=3,metrics=['accuracy']):
    bertmodel = load_bert( MODEL_PATH )  # Load model.
    # Defines Bert layer. Input layer names must be the same as defined in dataset.
    input_ids  = tf.keras.layers.Input(shape=(MAX_TEXT,), name='input_ids', dtype='int32')  # Input layer.
    attn_masks = tf.keras.layers.Input(shape=(MAX_TEXT,), name='attention_mask', dtype='int32')  # Input layer.
    bert_embds = bertmodel( [input_ids,attn_masks] )[0][:,0,:]
    intermediate_layer = tf.keras.layers.Dense( inter_dense , activation='relu', name='intermediate_layer')(bert_embds)
    output_layer = tf.keras.layers.Dense( nclasses, activation='softmax', name='output_layer')( intermediate_layer )
    feedback_model= tf.keras.Model(inputs=[input_ids, attn_masks ], outputs=output_layer)
    feedback_model.summary()
    feedback_model.compile(optimizer=Adam(learning_rate=1e-5, decay=1e-6), loss='categorical_crossentropy', metrics=metrics)
    return  feedback_model

def process_input():
    return


if __name__ == '__main__':
    # from transformers import pipeline
    # classifier = pipeline("sentiment-analysis")  # 情感分析
    # q = classifier("I've been waiting for a HuggingFace course my whole life.")
    # print( q )
    # exit()
    bert = load_bert( "J:\\work\\input\\bert-base-chinese" )
    print(bert)


    exit()
    #load data


    exit()
    
  
  