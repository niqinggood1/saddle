#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：computational_Finance 
@File    ：bert.py
@IDE     ：PyCharm 
@Author  ：patrick
@Date    ：2022/7/31 8:11 
'''
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
import numpy as np


#sample  get 50input , if

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []



if __name__ == '__main__':
    config_path = 'J:\\work\\models\\chinese_L-12_H-768_A-12\\bert_config.json'
    checkpoint_path = 'J:\\work\\models\\chinese_L-12_H-768_A-12\\bert_model.ckpt'
    dict_path = 'J:\\work\\models\\chinese_L-12_H-768_A-12\\vocab.txt'
    tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
    a = tokenizer.encode("吾儿莫慌")                                  ; print('a',a)
    b = tokenizer.tokens_to_ids(['[CLS]','吾','儿','莫','慌','[SEP]']); print('b',b)
    c = tokenizer.tokens_to_ids(['[CLS]','语','言','模','型','[SEP]']); print('c',c)
    d = tokenizer.tokenize("吾儿莫慌") ;                               print('d',d)

    exit()
    model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重

    # 编码测试
    token_ids, segment_ids = tokenizer.encode( u'语言模型' )
    print('\n ===== predicting =====\n')
    print(model.predict([np.array([token_ids]), np.array([segment_ids])]))
    #转换，开始训练



    exit()
    
  
  