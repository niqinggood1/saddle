from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,GRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model

def create_lstm_model( lstmcells=4,densecells=1,steps=1,features=1 ):
    # create and fit the LSTM network
    model = Sequential()
    model.add( LSTM(  lstmcells, input_shape=(steps,features))  )
    model.add( Dense( densecells ) )
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def create_gru_model( cells=4,densecells=1,steps=1,features=1 ):
    # create and fit the LSTM network
    model = Sequential()
    model.add( GRU(cells, input_shape=(steps,features))  )
    model.add( Dense( densecells ) )
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

from .fm import Attention_layer
from attention import Attention
from tensorflow.keras import Input
from tensorflow.keras.models import load_model,Model
def create_attention_model( cells=4,densecells=1,steps=1,features=1 ):
    model_input = Input(shape=(steps, features))
    x = LSTM(64, return_sequences=True)(model_input)
    x = Attention(units=32)(x)
    #x = Attention_layer()(x)

    x = Dense(1)(x)
    model = Model(model_input, x)
    model.compile(loss='mae', optimizer='adam')
    return model

    # model = Sequential()
    # model.add( GRU(cells, input_shape=(steps,features))  )
    # model.add( Attention(units=4)  )
    # model.add( Dense( densecells ) )
    # model.compile(loss='mean_squared_error', optimizer='adam')


def model_fit(model,trainx,trainy,batch_size):
    model.fit(trainx, trainy, epochs=100, batch_size=batch_size, verbose=2)
    return

from tensorflow import keras
from tensorflow.keras.layers import Concatenate, Dense
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.models import Model

def cnn_model(time_varying_cols, n_steps_in, n_steps_out, used_cols,loss='mse' ,metric='mse',activation='linear' ):
    n_features = len(time_varying_cols)

    inputs_1 = keras.Input(shape=(n_steps_in, n_features))
    x1 = keras.layers.Conv1D(20, 5, padding='causal', dilation_rate=2)(inputs_1)
    x1 = keras.layers.Conv1D(20, 5, padding='causal', dilation_rate=4)(x1)
    x1 = keras.layers.Dense(1)(x1)
    x1 = Flatten()(x1)
    cnn_output_1 = Model(inputs=inputs_1, outputs=x1)
    # Inputting Number features
    inputs_2 = Input(shape=(len(used_cols),))
    # Merging inputs
    merge = Concatenate()([cnn_output_1.output, inputs_2])
    reg_dense = Dense(128)(merge)
    out = Dense(n_steps_out, activation=activation)(reg_dense)
    # Make a model
    model = Model([cnn_output_1.input, inputs_2], out)
    # optimizer learning rate
    opt = keras.optimizers.Adam(learning_rate=0.01)
    # Compile the model
    model.compile(loss=loss, optimizer=opt, metrics=[metric])
    model.summary()
    return model


def mirana_model(n_features,n_other, n_steps_in, n_steps_out,
                 loss ,metric,activation='linear' ):  #other_model_features,
    inputs_1 = keras.Input(shape=(n_steps_in, n_features))
    # x1 = LSTM(26, return_sequences=True)(inputs_1)
    x1 = keras.layers.Conv1D(20, 5, padding='causal', dilation_rate=2)( inputs_1 )
    x1 = keras.layers.Conv1D(20, 5, padding='causal', dilation_rate=4)( x1 )
    x1 = Dropout(0.1)(x1 )
    x1 = Flatten()(   x1 )
    x1 = Dense(1)( x1   )
    cnn_output_1 = Model(inputs=inputs_1, outputs=x1)

    x2           = LSTM(16, return_sequences=True)(inputs_1 )
    x2           = Attention(30)(x2)  ## x2 = Dropout(0.5)(x2)
    x2           = Dense(1)(x2)
    lstm_output2 = Model(inputs=inputs_1, outputs=x2)

    x3 = Dense(32, activation=None, kernel_regularizer=keras.regularizers.l2(0.01),
               bias_regularizer=keras.regularizers.l2(0.01))( inputs_1[:, -1, :] )
    x3 = Dropout(0.3)( x3 )
    x3 = Flatten()(x3)
    x3 = Dense(1, activation=None, kernel_regularizer=keras.regularizers.l2(0.01)
               , bias_regularizer=keras.regularizers.l2(0.01))(x3)
    dense_output3 = Model(inputs=inputs_1, outputs=x3)

    inputs_2 = Input( shape=(n_other,) )
    merge    = Concatenate()( [ cnn_output_1.output, inputs_1[:,-1,:],lstm_output2.output,inputs_2   ]  )  # lstm_output2.output, inputs_2   dense_output3.output,   casual dilateion cnn  ,inputs_2
    merge    = Dense(32, kernel_regularizer=keras.regularizers.l1(0.02) )(merge)
    out      = Dense(n_steps_out, activation=activation)(merge)
    model    = Model(inputs=[   cnn_output_1.input, inputs_2], outputs=out)  # Make a model

    opt = keras.optimizers.Adam(learning_rate=0.02)  # optimizer learning rate
    model.compile(loss=loss, optimizer=opt, metrics=[metric])  # Compile the model
    model.summary()
    keras.utils.plot_model(model, 'causal_dilated_convolution.png', show_shapes=True)
    return model


from tensorflow.keras.layers import Dense, Lambda, Dot, Activation, Concatenate, Layer,Dropout
def multi_temporal_model(n_features, n_steps_in, n_steps_out,
                 loss ,metric,activation='linear' ):  #other_model_features,
    inputs_1 = keras.Input(shape=(n_steps_in, n_features))
    # x1 = LSTM(26, return_sequences=True)(inputs_1)
    x1 = keras.layers.Conv1D(16, 4, padding='causal', dilation_rate=2)( inputs_1 )
    x1 = keras.layers.Conv1D(18, 3, padding='causal', dilation_rate=4)( x1 )
    x1 = Dropout(0.1)(x1 )
    x1 = Flatten()(   x1 )
    x1 = Dense(1)( x1   )
    cnn_output_1 = Model(inputs=inputs_1, outputs=x1)

    x2           = LSTM(5, return_sequences=True)(inputs_1 )
    x2           = Attention(10)(x2)  ## x2 = Dropout(0.5)(x2)
    x2           = Dense(1)(x2)
    lstm_output2 = Model(inputs=inputs_1, outputs=x2)

    x3 = Dense(32, activation=None, kernel_regularizer=keras.regularizers.l2(0.01),
               bias_regularizer=keras.regularizers.l2(0.01))( inputs_1[:, -1, :] )
    x3 = Dropout(0.3)( x3 )
    x3 = Flatten()(x3)
    x3 = Dense(1, activation=None, kernel_regularizer=keras.regularizers.l2(0.01)
               , bias_regularizer=keras.regularizers.l2(0.01))(x3)
    dense_output3 = Model(inputs=inputs_1, outputs=x3)

    # n_other  = 1
    # inputs_2 = Input( shape=(n_other,) )
    merge    = Concatenate()( [ cnn_output_1.output, inputs_1[:,-1,:],lstm_output2.output]  ) #   # lstm_output2.output, inputs_2   dense_output3.output,   casual dilateion cnn  ,inputs_2
    merge    = Dense(32, kernel_regularizer=keras.regularizers.l1(0.02) )(merge)
    out      = Dense(n_steps_out, activation=activation)(merge)
    model    = Model(inputs= cnn_output_1.input, outputs=out)  # Make a model

    opt = keras.optimizers.Adam(learning_rate=0.02)  # optimizer learning rate
    model.compile(loss=loss, optimizer=opt, metrics=[metric])  # Compile the model
    model.summary()
    keras.utils.plot_model(model, 'causal_dilated_convolution.png', show_shapes=True)
    return model



from tensorflow.keras.layers import LSTM
from tensorflow.keras import backend as K
debug_flag=1
class Attention(object if debug_flag else Layer):

    def __init__(self, units=128, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units = units

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        with K.name_scope(self.name if not debug_flag else 'attention'):
            self.attention_score_vec = Dense(input_dim, use_bias=False, name='attention_score_vec')
            self.h_t = Lambda(lambda x: x[:, -1, :], output_shape=(input_dim,), name='last_hidden_state')
            self.attention_score = Dot(axes=[1, 2], name='attention_score')
            self.attention_weight = Activation('softmax', name='attention_weight')
            self.context_vector = Dot(axes=[1, 1], name='context_vector')
            self.attention_output = Concatenate(name='attention_output')
            self.attention_vector = Dense(self.units, use_bias=False, activation='tanh', name='attention_vector')
        if not debug_flag:
            # debug: the call to build() is done in call().
            super(Attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def __call__(self, inputs, training=None, **kwargs):
        if debug_flag:
            return self.call(inputs, training, **kwargs)
        else:
            return super(Attention, self).__call__(inputs, training, **kwargs)

    # noinspection PyUnusedLocal
    def call(self, inputs, training=None, **kwargs):
        """
        Many-to-one attention mechanism for Keras.
        @param inputs: 3D tensor with shape (batch_size, time_steps, input_dim).
        @param training: not used in this layer.
        @return: 2D tensor with shape (batch_size, units)
        @author: felixhao28, philipperemy.
        """
        if debug_flag:
            self.build(inputs.shape)  # ->(batch_size,step,hidden_siz)
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part-- last dim
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = self.attention_score_vec(inputs)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        h_t = self.h_t(inputs)
        score = self.attention_score([h_t, score_first_part]) # 第一维，* 第二维
        attention_weights = self.attention_weight(score)     # （batch_size,step  ） is weights
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = self.context_vector([inputs, attention_weights]) #对inputs 进行权重调整，后面乘以前面
        pre_activation = self.attention_output([context_vector, h_t])     #横向，拼接
        attention_vector = self.attention_vector(pre_activation)          #128的输出
        return attention_vector

    def get_config(self):
        """
        Returns the config of a the layer. This is used for saving and loading from a model
        :return: python dictionary with specs to rebuild layer
        """
        config = super(Attention, self).get_config()
        config.update({'units': self.units})
        return config



def cnn_model(n_features, n_steps_in, n_steps_out,loss ,metric,activation='linear' ):
    inputs_1 = keras.Input(shape=(n_steps_in, n_features))
    x1 = keras.layers.Conv1D(40, 5, padding='causal', dilation_rate=2)(inputs_1)
    x1 = keras.layers.Conv1D(40, 5, padding='causal', dilation_rate=4)(x1)
    x1 = keras.layers.Dense(1)(x1)
    x1 = Flatten()(x1)
    x1 = Dense(1)(x1)
    cnn_output_1 = Model(inputs=inputs_1, outputs=x1)

    merge       = Concatenate()( [ cnn_output_1.output,  inputs_1[:,-1,:]  ]  )
    reg_dense   = Dense(32,kernel_regularizer=keras.regularizers.l2(0.02) )(merge)
    out         = Dense(  n_steps_out, activation=activation )( reg_dense )
    model       = Model( inputs=inputs_1, outputs=out )


def lstm_attention(n_steps_in, n_features,n_steps_out):
    inputs_1 = keras.Input(shape=(n_steps_in, n_features))
    x2          = LSTM(16, return_sequences=True)(inputs_1)
    x2          = Attention(30)(x2)   ## x2 = Dropout(0.5)(x2)
    x2          = Dense(n_steps_out)(x2 )
    lstm_output2= Model( inputs=inputs_1, outputs=x2 )
    return  lstm_output2

def dnn(n_steps_in, n_features,activation='linear' ):
    inputs_1 = keras.Input(shape=(n_steps_in, n_features))
    x3 = Dense(32, activation='elu',kernel_regularizer=keras.regularizers.l1(0.01)
                                   ,bias_regularizer=keras.regularizers.l1(0.01))(  )
    x3 = Dropout(0.3)(x3)
    x3 = Flatten()(x3)
    x3 = Dense(1,activation=activation,kernel_regularizer=keras.regularizers.l2(0.01)
                                   ,bias_regularizer=keras.regularizers.l2(0.01))(x3)
    dense_output3 = Model(inputs=inputs_1, outputs=x3)
    return  dense_output3


