import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input, Dense,Embedding,Dropout


def sparseFeature(feat, feat_onehot_dim, embed_dim):
    return {'feat': feat, 'feat_onehot_dim': feat_onehot_dim, 'embed_dim': embed_dim}

def denseFeature(feat):
    return {'feat': feat}
class Dense_layer(Layer):
    def __init__(self, hidden_units, output_dim, hidden_layer_activation):
        super().__init__()
        self.hidden_layer = [Dense(x, activation=hidden_layer_activation,kernel_regularizer=tf.keras.regularizers.l2(0.01)
                                   ,bias_regularizer=tf.keras.regularizers.l2(0.01))
                             for x in hidden_units]
        self.dropout_layer= Dropout(0.3 )
        self.output_layer = Dense(output_dim, activation=None) #'sigmoid'
    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.hidden_layer:
            x = layer(x)
        x= self.dropout_layer(x)
        output = self.output_layer(x)
        return output

class Cross_layer(Layer):
    def __init__(self, layer_num, reg_w=1e-5, reg_b=1e-5):
        super().__init__()     #super的含义
        self.layer_num = layer_num
        self.reg_w = reg_w
        self.reg_b = reg_b
    def build(self, input_shape):
        #自动只执行一次build, 也可以在call里执行   # 每层对应不同的w,b
        self.cross_weight = [self.add_weight(name='w'+str(i), shape=(input_shape[1], 1),   # 跟输入维度相同的向量
                                             initializer=tf.random_normal_initializer(),regularizer=tf.keras.regularizers.l1_l2(l1=0.01,l2=0.01),trainable=True)  # tf.keras.regularizers.l2(self.reg_w)
                             for i in range(self.layer_num)]
        self.cross_bias   = [ self.add_weight(name='b'+str(i),   shape=(input_shape[1], 1), # 跟输入维度相同的向量
                                            initializer=tf.zeros_initializer(),regularizer=tf.keras.regularizers.l1_l2(l1=0.01,l2=0.01),#tf.keras.regularizers.l2(self.reg_b),
                                            trainable=True)
                            for i in range(self.layer_num)]			   #
        self.built = True  # 最后这句话一定要加上 #super(MyLayer, self).build(input_shape)  # 一定要在最后调用它
    def call(self, inputs, **kwargs): #前项传播
        x0 = tf.expand_dims(inputs, axis=2)  # (None, dim, 1)
        xl = x0  							 # (None, dim, 1)
        for i in range(self.layer_num):
            # 先乘后两项得到标量，便于计算
            xl_w = tf.matmul(tf.transpose(xl, [0, 2, 1]), self.cross_weight[i] ) # (None, 1, 1)
            # 再乘上x0，加上b、xl
            xl = tf.matmul(x0, xl_w) +  self.cross_bias[i] + xl +1e-15 # (None, dim, 1)

        output = tf.squeeze(xl, axis=2)  # (None, dim)
        #output = tf.concat([output, inputs], axis=1)
        return output

from tensorflow.keras import Model
class DCN(Model):
    def __init__(self,dense_last_idx,sparse_fea_dim,
                 dense_hidden_units, dense_output_dim, dense_hidden_activation,
                 cross_layer_num, reg_w=1e-4, reg_b=1e-4,last_activation=None): #'sigmoid'
        super().__init__()
        # self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.dense_last_idx = dense_last_idx
        self.sparse_fea_dim = sparse_fea_dim
        self.sparse_fea_num = len( sparse_fea_dim )
        self.embed_layers = {
            'embed_' + str(i): Embedding( feat['feat_onehot_dim'], feat['embed_dim'])
            for i, feat in enumerate( self.sparse_fea_dim ) }
        self.dense_layer = Dense_layer( dense_hidden_units, dense_output_dim, hidden_layer_activation=dense_hidden_activation )
        self.cross_layer = Cross_layer( cross_layer_num, reg_w=reg_w, reg_b=reg_b )
        #self.output_layer = Dense_layer([16,8,8], 1 , activation=activation)
        #self.hidden_layer = Dense_layer([128], 32, hidden_layer_activation='elu')
        self.output_layer = Dense( 1, activation=last_activation,
                                   kernel_regularizer=tf.keras.regularizers.l1_l2(0.001,0.02)  )
        print( 'in DCN embeding len:',len(self.embed_layers),'sparse_fea_num',self.sparse_fea_num,  self.embed_layers.keys() )
        #,        bias_regularizer=tf.keras.regularizers.l2(0.01)
    def call(self, inputs):
        dense_inputs  = inputs[:,:self.dense_last_idx]
        sparse_inputs = inputs[:, self.dense_last_idx: self.dense_last_idx + self.sparse_fea_num]

        # null_input    = tf.placeholder("float", shape=[None, 2,3])
        # sparse_inputs = tf.cond( self.sparse_fea_num,
        #        null_input            )
        #print( 'sparse_inputs,shape',sparse_inputs)
        # dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
        # embedding
        #emb = [ self.embed_layers[i](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        emb  = [ self.embed_layers['embed_' + str(i) ](sparse_inputs[:, i])
                                  for i in range( sparse_inputs.shape[1])  ]
        sparse_embed = tf.concat( emb , axis=1)

        #x = tf.cond(tf.equal(self.sparse_fea_num,0),lambda:dense_inputs,lambda:tf.concat([dense_inputs, sparse_embed], axis=1) )
        x = tf.concat([dense_inputs, sparse_embed], axis=1)  #  ( None,dense,x1 ) + ( None,sparse,x2 )=
        # Crossing layer
        cross_output = self.cross_layer(x)
        # Dense layer
        dnn_output = self.dense_layer(x)
        x = tf.concat([cross_output, dnn_output,x], axis=1)
        output =  self.output_layer(x) # output = tf.nn.sigmoid( self.output_layer(x) ) # 这里有改变，需要定位，需要论证是否可行
        return output

from .fm import Attention_layer
class ADCN(Model):
    def __init__(self,dense_last_idx,sparse_fea_dim,
                 dense_hidden_units, dense_output_dim, dense_hidden_activation,
                 cross_layer_num, reg_w=1e-4, reg_b=1e-4,): #'sigmoid'
        super().__init__()
        # self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.dense_last_idx = dense_last_idx
        self.sparse_fea_dim = sparse_fea_dim
        self.sparse_fea_num = len( sparse_fea_dim )
        self.embed_layers = {
            'embed_' + str(i): Embedding( feat['feat_onehot_dim'], feat['embed_dim'])
            for i, feat in enumerate(self.sparse_fea_dim)
        }
        self.dense_layer = Dense_layer(dense_hidden_units, dense_output_dim, hidden_layer_activation=dense_hidden_activation)
        self.cross_layer = Cross_layer(cross_layer_num, reg_w=reg_w, reg_b=reg_b)
        #self.output_layer = Dense_layer([16,8,8], 1 , activation=activation)
        #self.hidden_layer = Dense_layer([128], 32, hidden_layer_activation='elu')
        self.output_layer = Dense(1, activation=None,
                                  kernel_regularizer=tf.keras.regularizers.l1(0.01)
                                  )
        #,        bias_regularizer=tf.keras.regularizers.l2(0.01)
        self.attention_layer   = Attention_layer()

    def call(self, inputs):
        dense_inputs  = inputs[:, :self.dense_last_idx]
        sparse_inputs = inputs[:, self.dense_last_idx: self.dense_last_idx + self.sparse_fea_num]

        # null_input    = tf.placeholder("float", shape=[None, 2,3])
        # sparse_inputs = tf.cond( self.sparse_fea_num,
        #        null_input            )
        #print( 'sparse_inputs,shape',sparse_inputs)
        # dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]
        # embedding
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                                  for i in range(sparse_inputs.shape[1])], axis=1)
        #x = tf.cond(tf.equal(self.sparse_fea_num,0),lambda:dense_inputs,lambda:tf.concat([dense_inputs, sparse_embed], axis=1) )
        x = tf.concat([dense_inputs, sparse_embed], axis=1)
        # Crossing layer
        cross_output = self.cross_layer(x)
        # Dense layer
        dnn_output = self.dense_layer(x)
        x = tf.concat([cross_output, dnn_output,x], axis=1)
        x = self.attention_layer(x)
        output = tf.nn.sigmoid( self.output_layer(x) )
        return output

def sparseFeature(feat, feat_onehot_dim, embed_dim):
    return {'feat': feat, 'feat_onehot_dim': feat_onehot_dim, 'embed_dim': embed_dim}

def denseFeature(feat):
    return {'feat': feat}





if __name__ == '__main__':

    exit()
    # # # feature_columns=[稠密特征 + 稀疏矩阵特征 ]
    # # desne_list =[ {'feat':'f1'},{'feat':'f2'},{'feat':'f3'},{'feat':'f4'} ]
    # sparse_list=[ {'feat':'f5', 'feat_onehot_dim':8, 'embed_dim':10 },{'feat':'f6','feat_onehot_dim':8,'embed_dim': 10 } ]
    #
    # test_DCN=DCN( 8, sparse_fea_dim=sparse_list,
    #               dense_hidden_units=[256, 128, 64], dense_output_dim=4, activation='relu', cross_layer_num=10)