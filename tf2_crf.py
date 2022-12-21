import tensorflow as tf
from tf2crf import CRF, ModelWithCRFLoss
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, Concatenate, Embedding
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from nlp_huggingface import load_bert
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow_addons.text.crf import crf_log_likelihood
from typing import Union

class CRF(tf.keras.layers.Layer):
    """
    Conditional Random Field layer (tf.keras)
    `CRF` can be used as the last layer in a network (as a classifier). Input shape (features)
    must be equal to the number of classes the CRF can predict (a linear layer is recommended).

    Args:
        units: Positive integer, dimensionality of the output space. If it is None, the last dim of input must be num_classes
        chain_initializer: the initialize method for transitions, default orthogonal.
        regularizer: Regularizer for crf transitions, can be 'l1', 'l2' or other tensorflow regularizers.
    Input shape:
        nD tensor with shape `(batch_size, sentence length, features)` or `(batch_size, sentence length, num_classes)`.
    Output shape:
        in training:
            viterbi_sequence: the predicted sequence tags with shape `(batch_size, sentence length)`
            inputs: the input tensor of the CRF layer with shape `(batch_size, sentence length, num_classes)`
            sequence_lengths: true sequence length of inputs with shape `(batch_size)`
            self.transitions: the internal transition parameters of CRF with shape `(num_classes, num_classes)`
        in predicting:
            viterbi_sequence: the predicted sequence tags with shape `(batch_size, sentence length)`

    Masking
        This layer supports keras masking for input data with a variable number
        of timesteps. To introduce masks to your data,
        use an embedding layer with the `mask_zero` parameter
        set to `True` or add a Masking Layer before this Layer
    """
    def __init__(self, units=None, chain_initializer="orthogonal", regularizer=None, **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.chain_initializer = tf.keras.initializers.get(chain_initializer)
        self.regularizer = regularizer
        self.transitions = None
        self.supports_masking = True
        self.mask = None
        self.accuracy_fn = tf.keras.metrics.Accuracy()
        self.units = units
        if units is not None:
            self.dense = tf.keras.layers.Dense(units)

    def get_config(self):
        config = super(CRF, self).get_config()
        config.update({
            "chain_initializer": "orthogonal"
        })
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3
        if self.units:
            units = self.units
        else:
            units = input_shape[-1]
        self.transitions = self.add_weight(
            name="transitions",
            shape=[units, units],
            initializer=self.chain_initializer,
            regularizer=self.regularizer
        )

    def call(self, inputs, mask=None, training=False):
        if mask is None:
            raw_input_shape = tf.slice(tf.shape(inputs), [0], [2])
            mask = tf.ones(raw_input_shape)
        sequence_lengths = K.sum(K.cast(mask, 'int32'), axis=-1)
        if self.units:
            inputs = self.dense(inputs)
        viterbi_sequence, _ = tfa.text.crf_decode(
            inputs, self.transitions, sequence_lengths
        )
        return viterbi_sequence, inputs, sequence_lengths, self.transitions

def unpack_data(data):
    if len(data) == 2:
        return data[0], data[1], None
    elif len(data) == 3:
        return data
    else:
        raise TypeError("Expected data to be a tuple of size 2 or 3.")


class ModelWithCRFLoss(tf.keras.Model):
    """
    Wrapper around the base model for custom training logic.
    Args:
        base_model: The model including the CRF layer
        sparse_target: if the y label is sparse or one-hot, default True
        metric: the metric for training, default 'accuracy'. Warning: Currently tensorflow metrics like AUC need the output and y_true to be one-hot to cauculate, they are not supported.
    """

    def __init__(self, base_model, sparse_target=True, metric: Union[str, object] = 'accuracy'):
        super().__init__()
        self.base_model = base_model
        self.sparse_target = sparse_target
        self.metric = metric
        if isinstance(metric, str):
            if metric == 'accuracy':
                self.metrics_fn = tf.keras.metrics.Accuracy(name='accuracy')
            else:
                raise ValueError('unknown metric name')
        else:
            self.metrics_fn = self.metric
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

    def call(self, inputs, training=False):
        if training:
            return self.base_model(inputs)
        else:
            return self.base_model(inputs)[0]

    def compute_loss(self, x, y, training=False):
        viterbi_sequence, potentials, sequence_length, chain_kernel = self(x, training=training)
        # we now add the CRF loss:
        crf_loss = -crf_log_likelihood(potentials, y, sequence_length, chain_kernel)[0]
        return viterbi_sequence, sequence_length, tf.reduce_mean(crf_loss)

    def train_step(self, data):
        x, y, sample_weight = unpack_data(data)
        # y : '(batch_size, seq_length)'
        if self.sparse_target:
            assert len(y.shape) == 2
        else:
            y = tf.argmax(y, axis=-1)
        with tf.GradientTape() as tape:
            viterbi_sequence, sequence_length, crf_loss = self.compute_loss(x, y, training=True)
            loss = crf_loss + tf.cast(tf.reduce_sum(self.losses), crf_loss.dtype)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.metrics_fn.update_state(y, viterbi_sequence, tf.sequence_mask(sequence_length, y.shape[1]))
        return {"loss": self.loss_tracker.result(), self.metrics_fn.name: self.metrics_fn.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.metrics_fn]

    def test_step(self, data):
        x, y, sample_weight = unpack_data(data)
        # y : '(batch_size, seq_length)'
        if self.sparse_target:
            assert len(y.shape) == 2
        else:
            y = tf.argmax(y, axis=-1)
        viterbi_sequence, sequence_length, crf_loss = self.compute_loss(x, y, training=True)
        loss = crf_loss + tf.cast(tf.reduce_sum(self.losses), crf_loss.dtype)
        self.loss_tracker.update_state(loss)
        self.metrics_fn.update_state(y, viterbi_sequence, tf.sequence_mask(sequence_length, y.shape[1]))
        return {"loss_val": self.loss_tracker.result(), f'val_{self.metrics_fn.name}': self.metrics_fn.result()}


class ModelWithCRFLossDSCLoss(tf.keras.Model):
    """
        Wrapper around the base model for custom training logic. And DSC loss to help improve the performance of NER task.
        Args:
            base_model: The model including the CRF layer
            sparse_target: if the y label is sparse or one-hot, default True
            metric: the metric for training, default 'accuracy'. Warning: Currently tensorflow metrics like AUC need the output and y_true to be one-hot to cauculate, they are not supported.
            alpha: parameter for DSC loss
        """

    def __init__(self, base_model, sparse_target=True, metric: Union[str, object] = 'accuracy', alpha=0.6):
        super().__init__()
        self.base_model = base_model
        self.sparse_target = sparse_target
        self.metric = metric
        if isinstance(metric, str):
            if metric == 'accuracy':
                self.metrics_fn = tf.keras.metrics.Accuracy(name='accuracy')
            else:
                raise ValueError('unknown metric name')
        else:
            self.metrics_fn = self.metric
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.alpha = alpha

    def call(self, inputs, training=False):
        if training:
            return self.base_model(inputs)
        else:
            return self.base_model(inputs)[0]

    def compute_loss(self, x, y, sample_weight, training=False):
        viterbi_sequence, potentials, sequence_length, chain_kernel = self(x, training=training)
        # we now add the CRF loss:
        crf_loss = -crf_log_likelihood(potentials, y, sequence_length, chain_kernel)[0]
        ds_loss = compute_dsc_loss(potentials, y, self.alpha)
        if sample_weight is not None:
            crf_loss = crf_loss * sample_weight[0]
            ds_loss = ds_loss * sample_weight[1]
        return viterbi_sequence, sequence_length, tf.reduce_mean(crf_loss), tf.reduce_mean(ds_loss)

    def train_step(self, data):
        x, y, sample_weight = unpack_data(data)
        # y : '(batch_size, seq_length)'
        if self.sparse_target:
            assert len(y.shape) == 2
        else:
            y = tf.argmax(y, axis=-1)

        with tf.GradientTape() as tape:
            viterbi_sequence, sequence_length, crf_loss, ds_loss = self.compute_loss(x, y, sample_weight, training=True)
            loss = crf_loss + ds_loss + tf.cast(tf.reduce_sum(self.losses), crf_loss.dtype)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.metrics_fn.update_state(y, viterbi_sequence, tf.sequence_mask(sequence_length, y.shape[1]))
        return {"loss": self.loss_tracker.result(), self.metrics_fn.name: self.metrics_fn.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.metrics_fn]

    def test_step(self, data):
        x, y, sample_weight = unpack_data(data)
        if self.sparse_target:
            assert len(y.shape) == 2
        else:
            y = tf.argmax(y, axis=-1)
        viterbi_sequence, sequence_length, crf_loss, ds_loss = self.compute_loss(x, y, sample_weight, training=True)
        loss = crf_loss + ds_loss + tf.cast(tf.reduce_sum(self.losses), crf_loss.dtype)
        self.loss_tracker.update_state(loss)
        self.metrics_fn.update_state(y, viterbi_sequence, tf.sequence_mask(sequence_length, y.shape[1]))
        return {"loss_val": self.loss_tracker.result(), f'val_{self.metrics_fn.name}': self.metrics_fn.result()}

def compute_dsc_loss(y_pred, y_true, alpha=0.6):
    y_pred = K.reshape(K.softmax(y_pred), (-1, y_pred.shape[2]))
    y = K.expand_dims(K.flatten(y_true), axis=1)
    probs = tf.gather_nd(y_pred, y, batch_dims=1)
    pos = K.pow(1 - probs, alpha) * probs
    dsc_loss = 1 - (2 * pos + 1) / (pos + 2)
    return dsc_loss

# from keras_contrib.losses import crf_loss
# from keras_contrib.metrics import crf_accuracy
def bert_bilstm_crf( MODEL_PATH,inter_dense=256, nclasses=20,MAX_TEXT=512 ):
    bertmodel   = load_bert(  MODEL_PATH )
    input_ids   = tf.keras.layers.Input(  shape=(MAX_TEXT,), name='input_ids', dtype='int32'      )  # Input layer.
    attn_masks  = tf.keras.layers.Input( shape=(MAX_TEXT,), name='attention_mask', dtype='int32' )  # Input layer.
    bert_output = bertmodel.bert( input_ids, attention_mask=attn_masks)
    last_hidden_state = bert_output[0]
    x = Bidirectional( LSTM(nclasses, return_sequences=True,activation='relu' ))( last_hidden_state )
    crf = CRF( nclasses, name ="crf_output",dtype='float32') # sparse_target=True,nclasses, name="crf_output"
    output_layer= crf(x)
    base_model = Model( inputs=[input_ids, attn_masks ], outputs=output_layer ) #[out1, crf_output]
    model = ModelWithCRFLoss( base_model, sparse_target=True )
    model.compile(optimizer='adam')

    #model.compile( optimizer='adam',loss=crf.loss_function, metrics=[crf.accuracy] ) #[crf.accuracy]
    ###模型有两个loss,categorical_crossentropy和crf.loss_function
    # model.compile(optimizer='adam',
    #               loss={ 'crf_output': crf.loss},  #'out1': 'categorical_crossentropy', { 'crf_output': crf.loss}
    #               loss_weights={'crf_output': 1},           #{'out1': 1, 'crf_output': 1},
    #               metrics=["acc"])
    plot_model(model, to_file="model.png")
    #model.summary()
    return model

if __name__ == '__main__':
    MODEL_PATH  =  '../input/bert-base-chinese' #'./bert-base-chinese'
    model       =   bert_bilstm_crf(MODEL_PATH, inter_dense=256, nclasses=3,  MAX_TEXT=512)

    ## ###任务一，10分类的文本分类任务
    # out1 = GlobalMaxPool1D()(x)
    # out1 = Dense(64, activation='relu')(out1)
    # out1 = Dropout(0.5)(out1)
    # out1 = Dense(10, activation='softmax', name="out1")(out1)

    ###任务二，实体识别任务 sparse_target=True,