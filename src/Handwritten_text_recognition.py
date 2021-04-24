import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import parser
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
from DataLoaderIAM import DataLoaderIAM

img_dir = "C:/Users/avazb/Desktop/dir"
dl = DataLoaderIAM(img_dir, (128, 32), 32)
label = dl.toSparse(dl.words)
train_data = []
train_label = []

for i in range(len(dl.samples)):
    train_data.append(dl.samples[i].preprocessed_img)
    train_label.append(label[i])

train_data = np.array(train_data)
train_label = np.array(train_label)

imgSize = (128, 32, 1)

initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None)
img_input = keras.layers.Input(shape = imgSize)
labels = keras.layers.Input(name="label", shape=(32), dtype="float32")

# Creating CNN layers
x = keras.layers.Conv2D(filters=32, kernel_size=(5,5), activation = "relu", strides=(1,1), padding = "SAME", kernel_initializer = initializer, name = "CNN1")(img_input)
x = keras.layers.MaxPool2D(2, padding = "VALID", name = "MaxPool1")(x)
x = keras.layers.Conv2D(filters=64, kernel_size=(5,5), activation = "relu", strides = (1,1), padding = "SAME", kernel_initializer = initializer, name = "CNN2")(x)
x = keras.layers.MaxPool2D(2, padding = "VALID", name = "MaxPool2")(x)
x = keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation = "relu", strides = (1,1), padding = "SAME", kernel_initializer = initializer, name = "CNN3")(x)
x = keras.layers.MaxPool2D(pool_size = (1,2), padding = "VALID", name = "MaxPool3")(x)
x = keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation = "relu", strides = (1,1), padding = "SAME", kernel_initializer = initializer, name = "CNN4")(x)
x = keras.layers.MaxPool2D(pool_size = (1,2), padding = "VALID", name = "MaxPool4")(x)
x = keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation = "relu", strides = (1,1), padding = "SAME", kernel_initializer = initializer, name = "CNN5")(x)
cnnout4d = keras.layers.MaxPool2D(pool_size = (1,2), padding = "VALID", name = "MaxPool5")(x)

cnnin3d = tf.squeeze(cnnout4d, axis = [2])

#creating RNN leayers
rnn_cells = [tf.keras.layers.LSTMCell(256) for _ in range(2)]
stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells, name = "StackedRNNcell")
lstm_layer = tf.keras.layers.RNN(stacked_lstm, return_sequences = True, name = "RNNCell")
biderctional_lstm = keras.layers.Bidirectional(lstm_layer, merge_mode='concat')
x = biderctional_lstm(cnnin3d)
x = tf.expand_dims(x, 2)
x = keras.layers.Conv2D(80, kernel_size = (1,1), dilation_rate = (1,1), padding = "SAME", kernel_initializer = initializer)(x)
x = tf.squeeze(x, axis=[2])


#creating CTC loss
class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        print(y_true)
        print(y_pred)
        #loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        loss = tf.compat.v1.nn.ctc_loss_v2(tf.cast(y_true, dtype='int64'), y_pred, input_length, label_length)
        tf.print(loss)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred

print(labels)
output = CTCLayer(name="ctc_loss")(labels, x)

model = keras.Model([img_input, labels], outputs = output)
model.compile(optimizer = "adam")
print(train_data.shape)
print(train_label.shape)
model.fit([train_data, train_label])
#parser.add_argument('--dump', help='dump output of NN to CSV file(s)', action='store_true')
#args = parser.parse_args()