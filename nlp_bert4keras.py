#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：computational_Finance 
@File    ：nlp_dl.py
@IDE     ：PyCharm 
@Author  ：patrick
@Date    ：2022/7/30 21:18 
'''
def textcnn(inputs,kernel_initializer):
    # 3,4,5
    ke_cnt = 32  #256
    cnn1 = keras.layers.Conv1D(ke_cnt, 3, strides=1, padding='same', activation='relu', kernel_initializer=kernel_initializer )(inputs) # shape=[batch_size,maxlen-2,256]
    cnn1 = keras.layers.GlobalMaxPooling1D()(cnn1) # shape=[batch_size,256]
    cnn2 = keras.layers.Conv1D(ke_cnt, 4,strides=1, padding='same',activation='relu', kernel_initializer=kernel_initializer)(inputs)
    cnn2 = keras.layers.GlobalMaxPooling1D()(cnn2)
    cnn3 = keras.layers.Conv1D(ke_cnt,5, strides=1, padding='same',  kernel_initializer=kernel_initializer )(inputs)
    cnn3 = keras.layers.GlobalMaxPooling1D()(cnn3)
    cnn = keras.layers.concatenate([cnn1,cnn2,cnn3],axis=-1)
    output = keras.layers.Dropout(0.2)(cnn)
    return output
import  os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from bert4keras.backend import keras,set_gelu
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam

def build_bert_model(config_path,checkpoint_path,class_nums):
    # bert模型的预加载
    bert = build_transformer_model(config_path=config_path,checkpoint_path=checkpoint_path,model='bert',return_keras_model=False)
    # bert的输入是[CLS] token1 token2 token3 ... [sep]
    # 要从输出里提取到CLS，bert的输出是768维的语义向量
    # 用Lambda函数抽取所有行的第一列，因为CLS在第一个位置，如果后面不再接textCNN的话，就可以直接拿CLS这个向量，后面接全连接层去做分类了
    cls_features        = keras.layers.Lambda(   lambda x:x[:,0],name='cls_token' )(bert.model.output) # shape=[batch_size,768]
    # 去掉CLS和SEP的所有token（第一列到倒数第二列），抽取所有token的embedding，可以看作是input经过embedding之后的结果
    # 其实就是一个embedding矩阵，将这个矩阵传给textCNN
    all_token_embedding = keras.layers.Lambda(lambda x:x[:,1:-1], name='all_token')(bert.model.output) # shape=[batch_size,maxlen-2,768]
    cnn_features        = textcnn(all_token_embedding,bert.initializer) # shape=[batch_size,cnn_output_dim]
    # 经过CNN提取特征后，将其和CLS特征进行拼接，然后输入全连接层进行分类
    concat_features     = keras.layers.concatenate([cls_features,cnn_features],axis=-1) # 在768那个维度拼接
    dense               = keras.layers.Dense( units=32,activation='relu', kernel_initializer=bert.initializer )(concat_features)
    output              = keras.layers.Dense(units=class_nums,activation='softmax', kernel_initializer=bert.initializer)(dense)
    model               = keras.models.Model(bert.model.input,output)
    print( model.summary() )
    return model

# class data_generator(DataGenerator):
#     '''    数据生成器      '''
#     def __iter__(self,random=False):
#         batch_token_ids,batch_segment_ids,batch_labels = [],[],[]
#         for is_end,(text,label) in self.sample(random):
#             token_ids,segment_ids = tokenizer.encode(text,maxlen=maxlen)
#             batch_token_ids.append(token_ids)
#             batch_segment_ids.append(segment_ids)
#             batch_labels.append([label])
#             if len(batch_token_ids) == self.batch_size or is_end:
#                 batch_token_ids = sequence_padding(batch_token_ids)
#                 batch_segment_ids = sequence_padding(batch_segment_ids)
#                 batch_labels = sequence_padding(batch_labels)
#                 yield [batch_token_ids,batch_segment_ids],batch_labels # [模型的输入]，标签
#                 batch_token_ids,batch_segment_ids,batch_labels = [],[],[] # 再次初始化
#
# # data_generator只是一种为了节约内存的数据方式
# class data_generator:
#     def __init__(self, data, batch_size=32, shuffle=True):
#         self.data           = data
#         self.batch_size     = batch_size
#         self.shuffle        = shuffle
#         self.steps          = len(self.data) // self.batch_size
#         if len(self.data) % self.batch_size != 0:
#             self.steps += 1
#     def __len__(self):
#         return self.steps
#
#     def __iter__(self):
#         while True:
#             idxs = list(range(len(self.data)))
#             if self.shuffle:
#                 np.random.shuffle(idxs)
#             X1, X2, Y = [], [], []
#             for i in idxs:
#                 d = self.data[i]
#                 text = d[0][:maxlen]
#                 x1, x2 = tokenizer.encode(first=text)
#                 y = d[1]
#                 X1.append(x1)
#                 X2.append(x2)
#                 Y.append([y])
#                 if len(X1) == self.batch_size or i == idxs[-1]:
#                     X1 = seq_padding(X1)
#                     X2 = seq_padding(X2)
#                     Y  = seq_padding(Y,max_len=1)
#                     yield [X1, X2], Y[:, 0, :]
#                     [X1, X2, Y] = [], [], []

from bert4keras.snippets import sequence_padding,DataGenerator
def seq_padding(X, padding=0,max_len=150 ):
    # 让每条文本的长度相同，用0填充
    L = [len(x) for x in X]
    ML = max( L )
    if ML > max_len: ML=max_len
    return np.array([   np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x[:ML] for x in X ])

from bert4keras.tokenizers import Tokenizer
dict_path       = 'J:\\work\\models\\chinese_L-12_H-768_A-12\\vocab.txt'
tokenizer = Tokenizer(dict_path)
maxlen = 20

class data_generator(DataGenerator):
    '''  数据生成器     '''
    def __iter__(self,random=False):
        print(  'data_generator batch size', self.batch_size )
        batch_token_ids,batch_segment_ids,batch_labels = [],[],[]
        for is_end,(text,label) in self.sample(random):
            token_ids,segment_ids = tokenizer.encode(text,maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids,maxlen)
                batch_segment_ids = sequence_padding(batch_segment_ids,maxlen)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids,batch_segment_ids],batch_labels # [模型的输入]，标签
                batch_token_ids,batch_segment_ids,batch_labels = [],[],[] # 再次初始化

# 计算top-k正确率,当预测值的前k个值中存在目标类别即认为预测正确
def acc_top2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

from bert4keras.backend import keras
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding,DataGenerator
from sklearn.metrics import classification_report
from bert4keras.optimizers import Adam
import numpy as np
# bert模型设置
def build_bert(nclass):
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)  # 加载预训练模型
    for l in bert_model.layers:
        l.trainable = True
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    x = bert_model([x1_in, x2_in])
    x = Lambda( lambda x: x[:, 0] )( x )  # 取出[CLS]对应的向量用来做分类
    p = Dense(  nclass, activation='softmax')(x)
    model = Model([x1_in, x2_in], p)  # # 用足够小的学习率
    model.compile( loss='categorical_crossentropy', optimizer=Adam(1e-5), metrics=['accuracy', acc_top2] )
    print(model.summary())
    return model

def load_data(filename, text_col='text',label_col='label'):
    '''  加载数据,    单条格式：（文本，标签id）'''
    df = pd.read_csv(filename,header=0)
    return df[ [text_col ,label_col  ] ].values

# from saddle.tf_utils import
if __name__ == '__main__':
    # config_path = './bert_weights/rbt3/bert_config_rbt3.json'
    # checkpoint_path = './bert_weights/rbt3/bert_model.ckpt'
    config_path     = 'J:\\work\\models\\chinese_L-12_H-768_A-12\\bert_config.json'
    checkpoint_path = 'J:\\work\\models\\chinese_L-12_H-768_A-12\\bert_model.ckpt'
    dict_path       = 'J:\\work\\models\\chinese_L-12_H-768_A-12\\vocab.txt'
    class_nums      =  2  # 根据实际任务修改类别数
    bert_model      = build_bert_model(  config_path, checkpoint_path, class_nums )
    print('build finished...')
    keras.utils.plot_model(bert_model, 'J:\\work\\models\\bert_textcnn_new.png', show_shapes=True )

    print('\n ===== predicting =====\n')
    token_ids, segment_ids = tokenizer.encode(u'语言模型')
    print( bert_model.predict([np.array([token_ids]), np.array([segment_ids])]) )

    ## compile   & train & fit_generator
    train_data = [[ '语 言 模 型',0]]*30 +  [['好 坏 模 型',1]]*40
    print( train_data )
    train_generator  = data_generator( train_data, batch_size=8 )
    bert_model.compile(loss ='sparse_categorical_crossentropy',
                       optimizer=Adam(5e-6), metrics=['accuracy'])
    earlystop = keras.callbacks.EarlyStopping(monitor='val_acc', patience=2, verbose=2, mode='max')
    best_model_filepath = 'best_bert4keras_model.weights'

    checkpoint = keras.callbacks.ModelCheckpoint(best_model_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # 传入迭代器进行训练
    bert_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=3,
        shuffle=True, callbacks=[checkpoint])
    bert_model.save_weights(  best_model_filepath  )

    test_pred = []
    for x, _ in train_generator:
        p = bert_model.predict(x).argmax(axis=1)
        test_pred.extend(p)
    print(test_pred)
    exit()
    
  
  