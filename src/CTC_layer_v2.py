import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input

class CTCLayer_v2(layers.Layer):
   def __init__(self, name = None):
       super().__init__(name = name)
       self.loss_fn = keras.backend.ctc_batch_cost
       self.the_labels = Input(name='the_labels', shape=32, dtype='float32')
       self.input_length = Input(name='input_length', shape=[1], dtype='int64')
       self.label_length = Input(name='label_length', shape=[1], dtype='int64')


   def ctc_Layer_func(self, args):
      y_pred, labels, input_length, label_length = args

      return self.loss_fn(labels, y_pred, input_length, label_length)