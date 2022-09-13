import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense,Embedding,Dropout

class FM_layer(tf.keras.layers.Layer):
    def __init__(self, k, w_reg, v_reg):
        super().__init__()
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True,)
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.w_reg))
        self.v = self.add_weight(name='v', shape=(input_shape[-1], self.k),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.v_reg))
        self.built = True
    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        linear_part = tf.matmul(inputs, self.w) + self.w0   #shape:(batchsize, 1)
        inter_part1 = tf.pow( tf.matmul(inputs, self.v), 2)  #shape:(batchsize, self.k)
        inter_part2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2)) #shape:(batchsize, self.k)
        inter_part = 0.5*tf.reduce_sum(inter_part1 - inter_part2, axis=-1, keepdims=True) #shape:(batchsize, 1)

        output = linear_part + inter_part
        return tf.nn.sigmoid(output)

class FM(tf.keras.Model):
    def __init__(self, k, w_reg=1e-4, v_reg=1e-4):
        super().__init__()
        self.fm = FM_layer(k, w_reg, v_reg)

    def call(self, inputs, training=None, mask=None):
        output = self.fm(inputs)
        return output

from tensorflow.keras.layers import  Flatten
class Interaction_layer(Layer):
    '''# input shape:  [None, field, k]  # output shape: [None, field*(field-1)/2, k]  '''
    def __init__(self):
        super().__init__()

    def build(self, input_shape): # [None, field, k]
        self.cross_weight = self.add_weight(name='w', shape=(34, 1),   # 跟输入维度相同的向量input_shape[1]
                                             initializer=tf.random_normal_initializer(),regularizer=tf.keras.regularizers.l2(0.01),trainable=True)  # tf.keras.regularizers.l2(self.reg_w)
        self.cross_bias   = self.add_weight(name='b',   shape=(34, 1), # 跟输入维度相同的向量
                                              initializer=tf.zeros_initializer(),regularizer=tf.keras.regularizers.l2(0.01),#tf.keras.regularizers.l2(self.reg_b),
                                              trainable=True)
        self.dense_layer  = Dense( input_shape[1],kernel_regularizer=tf.keras.regularizers.l1(0.01)   )
        self.flatten = Flatten()
        self.built = True

    @tf.function
    def call(self, inputs, **kwargs): # [None, field, k]
        x0 = tf.expand_dims( inputs, axis=2)  # (None, dim, 1)
        # print('in Inter call x0:',x0)
        if K.ndim(x0) != 3:
            print('in Inter call x0:',x0)
            raise ValueError("in Interaction Layer Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(x0)))
        x = tf.matmul( x0, x0, transpose_b=True )
        x = self.flatten(x)             ;print('flatten x:',x )
        x = tf.expand_dims( x, axis=2)  ;print('expand_dims x:',x )
        x = tf.transpose(x, [0, 2, 1])  ;print('transpose x:',x )
        x = self.dense_layer(x)         ;print('dense_layer x:',x )
        x = tf.transpose(x, [0, 2, 1]) ;print('transpose x:',x )
        x = tf.squeeze(x, axis=2)  # (None, dim)
        return x
        # xl_w = tf.matmul( tf.transpose(x0, [0, 2, 1]), self.cross_weight  ) # (None, 1, 1)
        # # 再乘上x0，加上b、xl
        # xl = tf.matmul(x0, xl_w) +  self.cross_bias + x0 +1e-15 # (None, dim, 1)
        # output = tf.squeeze(xl, axis=2)  # (None, dim)
        # return output


        # x = self.flatten(x)
        # x = tf.matmul(  X0, Xi, transpose_b=True ) # list: k * [None, field_num[0], field_num[i]]

        #xl_w = tf.matmul(x0, x0,) # (None, 1, 1)
        # x = self.flatten(x)
        # out = tf.expand_dims(x, axis=2)   # (None, dim, 1)
        #print('x shape',x.shape, x)
        # print('out shape',out.shape, out)

        # for i in range(x0.shape[1]):
        #     for j in range(i+1, x0.shape[1]):
        #         # print('i,j',i,j,tf.multiply( x0[:, i], x0[:, j] ) )
        #         element_wise_product_list.append( tf.multiply( x0[:, i], x0[:, j] )  )  #[t, None, k]
        # element_wise_product = tf.transpose(tf.convert_to_tensor(element_wise_product_list), [1, 0, 2]) #[None, t, k]

        # print('element_wise_product_list:',element_wise_product_list)
        #element_wise_product =  tf.convert_to_tensor(element_wise_product_list)

import tensorflow.compat.v1 as tf
class Attention_layer(Layer):
    '''    # input shape:  [None, n, k]  ;     # output shape: [None, k]  '''
    def __init__(self):
        super().__init__()
        self.vector_dim  = 10
    def build(self, input_shape): # [None, field, k] input_shape[1]
        ndim = min(  self.vector_dim,input_shape[1] )
        self.attention_w = Dense(ndim, activation='relu',kernel_regularizer= tf.keras.regularizers.l1_l2(l1=0.03,l2=0.01)   )
        self.attention_h = Dense(1, activation=None)
        self.built = True
    def call(self, inputs, **kwargs): # [None, field, k]
        if K.ndim(inputs) == 2:
            inputs=tf.expand_dims(inputs,2)
        if K.ndim(inputs) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))
        x = self.attention_w(inputs)  # [None, field, field]
        x = self.attention_h(x)       # [None, field, 1]
        a_score = tf.nn.softmax(x)
        a_score = tf.transpose(a_score, [0, 2, 1]) # [None, 1, field]
        output  = tf.reshape( tf.matmul(a_score, inputs), shape=(-1, inputs.shape[2]))  # (None, k)
        return output

class AFM_layer(Layer):
    def __init__(self, dense_last_idx,sparse_fea_dim, mode='att'):
        super(AFM_layer, self).__init__()
        #self.dense_feature_columns, \
        self.dense_last_idx         = dense_last_idx
        self.sparse_fea_dim         = sparse_fea_dim
        self.sparse_fea_num         = len( sparse_fea_dim )
        self.mode = mode
        self.embed_layer = {"emb_"+str(i): Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                            for i, feat in enumerate(self.sparse_fea_dim)}
        self.interaction_layer1 = Interaction_layer()
        self.interaction_layer2 = Interaction_layer()
        self.attention_layer1   = Attention_layer()

        if self.mode=='att':
            self.attention_layer = Attention_layer()
        self.output_layer = Dense(1)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) == 1:
            inputs = tf.expand_dims( inputs, axis=2)
        if K.ndim(inputs) != 2:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))
        dense_inputs  = inputs[:,:self.dense_last_idx]
        sparse_inputs = inputs[:,self.dense_last_idx: self.dense_last_idx + self.sparse_fea_num]
        sparse_embed  = tf.concat([self.embed_layer['emb_'+str(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=1)
        #embed = tf.convert_to_tensor(embed)   #sparse_embed = tf.concat( embed , axis=1 )  # embed = tf.transpose(embed, [1, 0, 2])  #[None, 26，k]
        # Pair-wise Interaction
        x = tf.concat([ dense_inputs, sparse_embed ], axis=1 )
        x = tf.convert_to_tensor(x)
        x = self.interaction_layer1(x)
        x = self.interaction_layer2(x)
        x = tf.expand_dims( x, axis=2 )  # (None, dim, 1)
        if self.mode == 'avg':
            x = tf.reduce_mean(x, axis=1)  # (None, k)
        elif self.mode == 'max':
            x = tf.reduce_max(x, axis=1)  # (None, k)
        else:
            x = self.attention_layer(x)  # (None, k)
        output = tf.nn.sigmoid( self.output_layer(x) )
        return output

class AFM(Model):
    def __init__(self,dense_last_idx,sparse_fea_dim, mode):
        super().__init__()
        self.afm_layer = AFM_layer(dense_last_idx,sparse_fea_dim, mode)

    def call(self, inputs, training=None, mask=None):
        output = self.afm_layer(inputs)
        return output

class FFM_Layer(Layer):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4):
        super(FFM_Layer, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg
        self.feature_num = sum([feat['feat_onehot_dim'] for feat in self.sparse_feature_columns]) \
                           + len(self.dense_feature_columns)
        self.field_num = len(self.dense_feature_columns) + len(self.sparse_feature_columns)

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_num, 1),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.w_reg),
                                 trainable=True)
        self.v = self.add_weight(name='v',
                                 shape=(self.feature_num, self.field_num, self.k),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.v_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        dense_inputs = inputs[:, :13]
        sparse_inputs = inputs[:, 13:]
        # one-hot encoding
        x = tf.cast(dense_inputs, dtype=tf.float32)
        for i in range(sparse_inputs.shape[1]):
            x = tf.concat(
                [x, tf.one_hot( tf.cast(sparse_inputs[:, i], dtype=tf.int32),
                                depth=self.sparse_feature_columns[i]['feat_onehot_dim'])], axis=1)
        linear_part = self.w0 + tf.matmul(x, self.w)
        inter_part = 0
        # 每维特征先跟自己的 [field_num, k] 相乘得到Vij*X
        field_f = tf.tensordot(x, self.v, axes=1)  # [None, 2291] x [2291, 39, 8] = [None, 39, 8]
        # 域之间两两相乘，
        for i in range(self.field_num):
            for j in range(i+1, self.field_num):
                inter_part += tf.reduce_sum(
                    tf.multiply(field_f[:, i], field_f[:, j]), # [None, 8]
                    axis=1, keepdims=True
                )
        return linear_part + inter_part

class FFM(Model):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4):
        super(FFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.ffm = FFM_Layer(feature_columns, k, w_reg, v_reg)

    def call(self, inputs, **kwargs):
        output = self.ffm(inputs)
        output = tf.nn.sigmoid(output)
        return output

class DotProductAttention(Layer):
    def __init__(self, dropout=0.0):
        super(DotProductAttention, self).__init__()
        self._dropout = dropout
        self._masking_num = -2**32 + 1

    def call(self, inputs):
        # queries: [None, n, k]
        # keys:    [None, n, k]
        # values:  [None, n, k]
        queries, keys, values = inputs
        score = K.batch_dot(queries, tf.transpose(keys, [0, 2, 1]))  # [None, n, n]
        score = score/int(queries.shape[-1])**0.5   # 缩放
        score = K.softmax(score)                    # SoftMax
        score = K.dropout(score, self._dropout)     # dropout
        outputs = K.batch_dot(score, values)        # [None, n, k]
        return outputs

class MultiHeadAttention(Layer):
    def __init__(self, n_heads=4, head_dim=64, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self._n_heads = n_heads
        self._head_dim = head_dim
        self._dropout = dropout
        self._att_layer = DotProductAttention(dropout=self._dropout)

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self._weights_queries = self.add_weight(
            shape=(input_shape[0][-1], self._n_heads*self._head_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='weights_queries')
        self._weights_keys = self.add_weight(
            shape=(input_shape[1][-1], self._n_heads*self._head_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='weights_keys')
        self._weights_values = self.add_weight(
            shape=(input_shape[2][-1], self._n_heads*self._head_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='weights_values')

    def call(self, inputs):
        # queries: [None, n, k]
        # keys:    [None, n, k]
        # values:  [None, n, k]
        queries, keys, values = inputs
        if self._n_heads*self._head_dim != queries.shape[-1]:
            raise ValueError("n_head * head_dim not equal embedding dim {}".format(queries.shape[-1]))

        queries_linear = K.dot(queries, self._weights_queries)  # [None, n, k]
        keys_linear = K.dot(keys, self._weights_keys)           # [None, n, k]
        values_linear = K.dot(values, self._weights_values)     # [None, n, k]

        queries_multi_heads = tf.concat(tf.split(queries_linear, self._n_heads, axis=2), axis=0) # [None*n_head, n, k/n_head]
        keys_multi_heads = tf.concat(tf.split(keys_linear, self._n_heads, axis=2), axis=0)       # [None*n_head, n, k/n_head]
        values_multi_heads = tf.concat(tf.split(values_linear, self._n_heads, axis=2), axis=0)   # [None*n_head, n, k/n_head]

        att_out = self._att_layer([queries_multi_heads, keys_multi_heads, values_multi_heads])   # [None*n_head, n, k/n_head]
        outputs = tf.concat(tf.split(att_out, self._n_heads, axis=0), axis=2)    # [None, n, k]
        return outputs

class AutoInt(Model):
    def __init__(self, feature_columns, hidden_units, activation='relu',
                 dnn_dropout=0.0, n_heads=4, head_dim=64, att_dropout=0.1):
        super(AutoInt, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.dense_emb_layers = [Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                                 for feat in self.dense_feature_columns]
        self.sparse_emb_layers = [Embedding(feat['feat_onehot_dim'], feat['embed_dim'])
                                  for feat in self.sparse_feature_columns]
        self.dense_layer = Dense_layer(hidden_units, activation, dnn_dropout)
        self.multi_head_att = MultiHeadAttention(n_heads, head_dim, att_dropout)
        self.out_layer = Dense(1, activation=None)
        k = self.dense_feature_columns[0]['embed_dim']
        self.W_res = self.add_weight(name='W_res', shape=(k, k),
                                     trainable=True,
                                     initializer=tf.initializers.glorot_normal(),
                                     regularizer=tf.keras.regularizers.l1_l2(1e-5))

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
        # 值为1.0会使embedding报错
        dense_inputs = tf.where(tf.equal(dense_inputs, 1), 0.9999999, dense_inputs)
        dense_emb = [layer(dense_inputs[:, i]) for i, layer in enumerate(self.dense_emb_layers)]     # [13, None, k]
        sparse_emb = [layer(sparse_inputs[:, i]) for i, layer in enumerate(self.sparse_emb_layers)]  # [26, None, k]
        emb = tf.concat([tf.convert_to_tensor(dense_emb), tf.convert_to_tensor(sparse_emb)], axis=0) # [39, None, k]
        emb = tf.transpose(emb, [1, 0, 2])  # [None, 39, k]

        # DNN
        dnn_input = tf.reshape(emb, shape=(-1, emb.shape[1]*emb.shape[2])) # [None, 39*k]
        dnn_out = self.dense_layer(dnn_input)  # [None, out_dim]

        # AutoInt
        att_out = self.multi_head_att([emb, emb, emb]) # [None, 39, k]
        att_out_res = tf.matmul(emb, self.W_res)       # [None, 39, k]
        att_out = att_out + att_out_res
        att_out = tf.reshape(att_out, [-1, att_out.shape[1]*att_out.shape[2]]) # [None, 39*k]

        # output
        x = tf.concat([dnn_out, att_out], axis=-1)
        output = self.out_layer(x)
        return tf.nn.sigmoid(output)

