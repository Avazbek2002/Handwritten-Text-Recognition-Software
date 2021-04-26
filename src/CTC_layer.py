import tensorflow as tf
from tensorflow import keras

#creating CTC loss
class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64", name = "input_length")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64", name = "label_length")
        print(label_length)
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        print(label_length)
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        #loss = tf.compat.v1.nn.ctc_loss_v2(tf.cast(y_true, dtype='int64'), y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred