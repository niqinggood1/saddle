
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#########LR model
class Linefit(tf.keras.Model):
    """类继承方式搭建神经网络
    参数: tf.keras.Model: Model父类
    """
    def __init__(self, hidden_units, output_dim, hidden_layer_activation):
        # 继承
        super(Linefit, self).__init__()
        # 隐藏层-1
        self.layer1 = layers.Dense(10, activation=tf.nn.relu, name="layer1")
        # 隐藏层-2
        self.layer2 = layers.Dense(15, activation=tf.nn.relu, name="layer2")
        # 输出层
        self.outputs = layers.Dense(5, activation=tf.nn.softmax, name="outputs")
    def call(self, inputs):
        """实例回调接口，类似重载()
        参数:self: 对象
        inputs: 输入数据
        output: 输出层张量
        """
        layer1 = self.layer1(inputs)
        layer2 = self.layer2(layer1)
        return self.outputs(layer2)

########## mnist_layer
def mnist_layer():
    inputs = keras.Input(shape=(784,))
    # img_inputs = keras.Input(shape=(32, 32, 3))
    # 隐藏层-1
    dense = layers.Dense(64, activation="relu")
    x = dense(inputs)
    # 隐藏层-2
    x = layers.Dense(64, activation="relu")(x)
    # 输出层
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
    model.summary()
    keras.utils.plot_model(model, "mnist_model.png", show_shapes=True)
    return model


def line_fit_sequential(out_dim=5,activation="softmax"):
    """Sequential内置序列化搭建网络结构
    output:    model: 网络类实例
    """
    model = tf.keras.Sequential([
        # 隐藏层-1
        layers.Dense(10, activation="relu", input_shape=(1,), name="layer1"),
        # 隐藏层-2
        layers.Dense(15, activation="relu", name="layer2"),
        # 输出层
        layers.Dense(out_dim, activation=activation, name="outputs")
    ])
    model.compile()
    # 展示网络结构
    model.summary()
    # 绘制网络流程图
    keras.utils.plot_model(model, "./line-fit-seq.png", show_shapes=True)
    return model


def cnn_sequential_add():
    """Sequential序列外置搭建卷积神经网络
    返回:    model: 类实例 """
    model = tf.keras.Sequential()
    # 卷积层-1
    model.add(layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=(28, 28, 3), name="conv-1"))
    # 最大池化层-1
    model.add(layers.MaxPooling2D((2, 2), name="max-pooling-1"))
    # 卷积层-2
    model.add(layers.Conv2D(64, (3, 3), activation=tf.nn.relu, name="conv-2"))
    # 最大池化层-2
    model.add(layers.MaxPooling2D((2, 2), name="max-pooling-2"))
    # 卷积层-3
    model.add(layers.Conv2D(64, (3, 3), activation=tf.nn.relu, name="conv-3"))
    model.add(layers.Flatten())
    # 全连接层-1
    model.add(layers.Dense(64, activation=tf.nn.relu, name="fullc-1"))
    # softmax层
    model.add(layers.Dense(10, activation=tf.nn.softmax, name="softmax"))
    # 展示网络结构
    model.summary()
    # 绘制网络流程图
    keras.utils.plot_model(model, "./images/cnn-sequential-add.png", show_shapes=True)
    return model

def lenet_cnn():
    """搭建LeNet-5卷积神经网络
    output:    model 类实例"""
    # 实例化
    model = tf.keras.Sequential(name="LeNet-5")
    # 卷积层-1
    model.add(layers.Conv2D(6, (5, 5), activation="relu", input_shape=(32, 32, 1), name="conv-1"))
    # 最大池化层-1
    model.add(layers.MaxPooling2D((2,2), name="max-pooling-1"))
    # 卷积层-2
    model.add(layers.Conv2D(16, (5, 5), activation="relu", name="conv-2"))
    # 最大池化层-2
    model.add(layers.MaxPooling2D((2,2), name="max-pooling-2"))
    model.add(layers.Flatten())
    # 全连接层-1
    model.add(layers.Dense(120, activation="relu", name="fullc-1"))
    # 全连接层-2
    model.add(layers.Dense(84, activation="relu", name="fullc-2"))
    # softmax层
    model.add(layers.Dense(10, activation="softmax", name="softmax"))
    # 展示网络结构
    model.summary()
    # 绘制网络流程
    keras.utils.plot_model(model, "./images/lenet-5.png", show_shapes=True)
    return model

def vggnet_cnn():
    """搭建LeNet-5卷积神经网络
    output:     model: 类实例
    """
    # 实例化
    model = tf.keras.Sequential(name="VGGNet")
    # 卷积组-1
    model.add( layers.Conv2D(64, (3, 3),padding="same",activation="relu",input_shape=(224, 224, 3),name="conv1-1") )
    model.add( layers.Conv2D(64, (3, 3),padding="same",activation="relu",name="conv1-2") )
    # 最大池化层-1
    model.add( layers.MaxPooling2D((2,2), name="max-pooling-1")    )
    # 卷积组-2
    model.add( layers.Conv2D(128, (3, 3),padding="same",activation="relu",name="conv2-1")  )
    model.add( layers.Conv2D(128, (3, 3),padding="same",activation="relu",name="conv2-2")  )
    # 最大池化层-2
    model.add( layers.MaxPooling2D((2,2), name="max-pooling-2") )
    # 卷积组-3
    model.add( layers.Conv2D(256, (3, 3),padding="same",activation="relu",name="conv3-1"))
    model.add( layers.Conv2D(256, (3, 3),padding="same",activation="relu",name="conv3-2"))
    model.add( layers.Conv2D(256, (3, 3),padding="same",activation="relu",name="conv3-3"))
    # 最大池化层-3
    model.add( layers.MaxPooling2D((2,2),name="max-pooling-3") )
    # 卷积组-4
    model.add( layers.Conv2D(512, (3, 3),padding="same",activation="relu",name="conv4-1") )
    model.add( layers.Conv2D(512, (3, 3),padding="same",activation="relu",name="conv4-2") )
    model.add( layers.Conv2D(512, (3, 3),padding="same",activation="relu",name="conv4-3") )
    # 最大池化层-4
    model.add( layers.MaxPooling2D((2,2),name="max-pooling-4") )
    # 卷积组-5
    model.add( layers.Conv2D(512, (3, 3),padding="same",activation="relu",name="conv5-1") )
    model.add( layers.Conv2D(512, (3, 3),padding="same",activation="relu",name="conv5-2")   )
    model.add( layers.Conv2D(512, (3, 3),padding="same",activation="relu",name="conv5-3")  )
    # 最大池化层-5
    model.add( layers.MaxPooling2D((2,2),name="max-pooling-5") )
    # 数据拉伸
    model.add(layers.Flatten())
    # 全连接层-1
    model.add(  layers.Dense(4096, activation="relu", name="fullc-1") )
    # 全连接层-2
    model.add( layers.Dense(4096,activation="relu", name="fullc-2")  )
    # 全连接层-2
    model.add( layers.Dense(4096, activation="relu", name="fullc-3")  )
    # softmax层
    model.add( layers.Dense(1000, activation="softmax", name="softmax")  )
    # 展示网络结构
    model.summary()
    # 绘制网络流程
    keras.utils.plot_model( model,"./images/vggnet.png", show_shapes=True )
    return model



def create_tf_log(log_path = "./logs/mat"):
    summary_writer = tf.summary.create_file_writer(log_path)
    return summary_writer

def trac_tf_log(summary_writer,name='mat',log_path=''):
    # 追踪图结构,还不太会用
    # 保存图结构
    with summary_writer.as_default():
        tf.summary.trace_export(name=name, step=0, profiler_outdir=log_path)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
def use_gpu():
    gpus =tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # 设置GPU内存按需分配
    config = ConfigProto()
    config.gpu_options.allow_growth = True

def use_cpu():
    # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
    print( cpus)
    import os
    # os.environ["CUDA_DEVICE_ORDER"]  = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'   #"-1" '-1' #'-1'
    return

def gen_mnist_datas():
    """生成数据
    参数:      无
    返回:
        inputs: 训练图像
        outputs: 训练标签
        eval_images: 测试图像 """
    # 读取MNIST数据集
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    # 获取前1000个图像数据
    train_labels = train_labels[:1000]
    # 获取前1000个评估使用图像
    eval_images  = train_images[:1000]
    # 调整图像数据维度，供训练使用
    train_images = train_images[:1000].reshape(-1, 784)/255.0
    return train_images, train_labels, eval_images


def callback_only_params(model_path):
    """保存模型回调函数
    参数:
        model_path: 模型文件路径
    返回:
        ckpt_callback: 回调函数
    """
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        verbose=1,
        save_weights_only=True,
        save_freq='epoch'
    )
    return ckpt_callback

def tb_callback(model_path):
    """保存Tensorboard日志回调函数
    参数:
        model_path: 模型文件路径
    返回:
        tensorboard_callback: 回调函数
    """
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=model_path,
        histogram_freq=1)
    return tensorboard_callback

def train_model_save(model, inputs, outputs, model_path, log_path):
    """训练神经网络
    参数:
        model: 神经网络实例
        inputs: 输入数据
        outputs: 输出数据
        model_path: 模型文件路径
        log_path: 日志文件路径
    """
    # 回调函数
    ckpt_callback = callback_only_params(model_path)
    tensorboard_callback = tb_callback(log_path)  # tensorboard回调
    # 保存参数
    model.save_weights(model_path.format(epoch=0))
    # 训练模型，并使用最新模型参数
    history = model.fit(
        inputs,
        outputs,
        epochs=20,
        callbacks=[ckpt_callback], #, tensorboard_callback
        verbose=0
    )

    #绘制图像
    #plot_history(history)

def load_prediction(model, model_path, inputs):
    """神经网络预测
    参数:
        model: 神经网络实例
        model_path: 模型文件路径
        inputs: 输入数据
    返回:
        pres: 预测值
    """
    # 载入模型
    load_model(model, model_path)
    # 预测值
    pres = model.predict(inputs)
    # print("prediction:{}".format(pres))
    # 返回预测值
    return pres

def plot_mnist_prediction(model, model_path, inputs, evals):
    import matplotlib.pyplot as plt
    """可视化预测结果
    参数:
        model: 神经网络实例
        inputs: 输入数据
        outputs: 输出数据
        model_path: 模型文件路径
    返回:
        无
    """
    # 预测值
    pres = load_prediction(model, model_path, inputs)
    pres = tf.math.argmax(pres, 1)
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.subplots_adjust(wspace=0.5, hspace=0.8)
        plt.imshow(evals[i], cmap=plt.cm.binary)
        plt.title("预测值:{}".format(pres[i]), fontproperties=font)
    plt.savefig("./images/nn-pre.png", format="png", dpi=300)
    plt.show()

def display_model_structure(model, structure_path):
    """展示神经网络结构
    参数:
        model: 神经网络对象
        nn_structure_path: 神经网络结构保存路径
    返回:
        无
    """
    model.summary()
    keras.utils.plot_model(model, structure_path, show_shapes=True)



def create_mnist_cnn_model():
    """使用keras新建神经网络
    参数:       无
    返回:       model: 神经网络实例 """
    model = tf.keras.Sequential(name="MNIST-CNN")
    # 卷积层-1
    model.add(layers.Conv2D(32, (3,3), padding="same", activation=tf.nn.relu,input_shape=(28,28,1),name="conv-1") )
    # 最大池化层-1
    model.add(layers.MaxPooling2D((2,2),name="max-pooling-1" )  )
    # 卷积层-2
    model.add(layers.Conv2D(64, (3,3),padding="same", activation=tf.nn.relu, name="conv-2")    )
    # 最大池化层-2
    model.add( layers.MaxPooling2D(   (2,2),  name="max-pooling-2"  ) )
    # 全连接层-1
    model.add(layers.Flatten(name="fullc-1"))
    # 全连接层-2
    model.add(    layers.Dense(512,  activation=tf.nn.relu, name="fullc-2") )
    # 全连接层-3
    model.add( layers.Dense(10,   activation=tf.nn.softmax,   name="fullc-3") )
    # 配置损失计算及优化器
    compile_model(model)
    return model

def compile_model(model,loss='mae',learning_rate=0.01,  metrics=["mse"],pic_sav_path='xxxxx.png'):
    """神经网络参数配置 参数:      model: 神经网络实例  返回:      无"""
    ## dcnmodel.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate,clipnorm=1.0 ),
        loss=loss, #tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics=metrics  )
    #model.build()

    return model

def summary_plot_model(model,pic_sav_path='xxxx.png'):
    #after fit or build
    model.summary()
    keras.utils.plot_model(model, pic_sav_path, show_shapes=True)
    return

    # opt = keras.optimizers.Adam(learning_rate=0.02)  # optimizer learning rate
    # model.compile(loss=loss, optimizer=opt, metrics=[metric])  # Compile the model



def create_plate_model():
    """使用keras新建神经网络
    参数:         无
    返回:         model: 神经网络实例"""
    # 输入层
    inputs = tf.keras.Input(shape=(72, 272, 3), name="inputs")
    # 卷积层-1
    layer1 = layers.Conv2D( 32,    (3,3),   activation=tf.nn.relu,name="conv-1")(inputs)
    # 卷积层-2
    layer2 = layers.Conv2D( 32,  (3,3),  activation=tf.nn.relu,name="conv-2")(layer1)
    # 最大池化层-1
    layer3 = layers.MaxPooling2D((2,2), name="max-pooling-1")(layer2)
    # 卷积层-3
    layer4 = layers.Conv2D(  64, (3,3), activation=tf.nn.relu, name="conv-3")(layer3)
    # 卷积层-4
    layer5 = layers.Conv2D(64,(3,3),  activation=tf.nn.relu, name="conv-4")(layer4)
    # 最大池化层-2
    layer6 = layers.MaxPooling2D(  (2,2),  name="max-pooling-2")(layer5)
    # 卷积层-5
    layer7 = layers.Conv2D(   128,   (3,3),activation=tf.nn.relu,  name="conv-5")(layer6)
    # 卷积层-6
    layer8 = layers.Conv2D(   128,   (3,3),  activation=tf.nn.relu, name="conv-6")(layer7)
    # 全连接层-1
    layer9 = layers.MaxPooling2D((2,2), name="max-pooling-3")(layer8)
    layer10 = layers.Flatten(name="fullc-1")(layer9)
    # 输出，全连接层-21~27
    outputs = [ layers.Dense(65, activation=tf.nn.softmax, name="fullc-2{}".format(i+1))(layer10) for i in range(7)]
    # 模型实例化
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="PLR-CNN")
    # 配置优化器和损失函数
    compile_model(model)
    return model

###########chat machine
def tokenize(datas):
    """数据集处理为向量和字典
    参数:   datas: 数据集列表
    返回:  voc_li: 数据集向量; tokenizer: 数据集字典  """
    # 数据序列化为向量实例化
    tokenizer = keras.preprocessing.text.Tokenizer(filters="")
    tokenizer.fit_on_texts(datas)
    # 数据系列化为向量
    voc_li = tokenizer.texts_to_sequences(datas)
    # 数据向量填充
    voc_li = keras.preprocessing.sequence.pad_sequences( voc_li, padding="post")
    # 返回数据
    return voc_li, tokenizer

def train_model_server(model, inputs, outputs, model_path):
    """训练神经网络
    参数:
        model: 神经网络实例
        inputs: 输入数据
        outputs: 输出数据
        model_path: 模型文件路径
    返回:
        无
    """
    # 保存参数
    # 训练模型，并使用最新模型参数
    history = model.fit(
        inputs,
        outputs,
        epochs=300,
        verbose=0
    )
    # 保存Tensorflow Serving使用的pb模型
    tf.keras.models.save_model(
        model,
        model_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )

######## save_model,   load_model
def save_model( model,path="attention_lstm.h5"):
    model.save(path)
    return

def save_weights( model,path="attention_lstm.h5" ):
    model.save_weights(path)
    return

from tensorflow.keras.models import load_model
from attention import Attention

def load_attention():
    model = load_model("attention_lstm.h5" , custom_objects={'Attention': Attention}  )
    return model

def load_model_weights(model, model_path):
    """载入模型
    参数: model: 神经网络实例,  model_path: 模型文件路径  返回: 无  """
    # latest = tf.train.latest_checkpoint(model_path)  # 检查最新模型
    # print("latest:{}".format(model_path))
    model.load_weights(model_path)
    return model

def load_model(path):
    from  tensorflow.keras.models import load_model
    model= load_model(path)
    return model


def load_model2(path):
    # predict and evl
    from  tensorflow.keras.models import load_model
    from causal_dilated_cnn_net import Attention_layer #
    from attention import Attention
    model= load_model(path,custom_objects = {'Attention':Attention,
                                             'attention_layer': Attention_layer,
                                             'Attention_layer': Attention_layer} )
    return model


if __name__ == "__main__":
    lenet_cnn()
    model1 = cnn_sequential_add( )