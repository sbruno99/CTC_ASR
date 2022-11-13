import keras
from keras import layers
import numpy as np
import tensorflow as tf



trainfeatures = np.load("traindatacoefs.npy")
trainlabels = np.load("traindatalabs.npy")

print(trainfeatures.shape)
print(trainlabels.shape)


class CERMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the Character Error Rate
    """
    def __init__(self, name='CER_metric', **kwargs):
        super(CERMetric, self).__init__(name=name, **kwargs)
        self.cer_accumulator = self.add_weight(name="total_cer", initializer="zeros")
        self.counter = self.add_weight(name="cer_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        input_shape = tf.keras.backend.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0]) * tf.keras.backend.cast(input_shape[1], 'float32')

        decode, log = tf.keras.backend.ctc_decode(y_pred,
                                    input_length,
                                    greedy=True)

        decode = tf.keras.backend.ctc_label_dense_to_sparse(decode[0], tf.keras.backend.cast(input_length, 'int32'))
        y_true_sparse = tf.keras.backend.ctc_label_dense_to_sparse(y_true, tf.keras.backend.cast(input_length, 'int32'))

        decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))

        distance = tf.edit_distance(tf.keras.backend.cast(decode,'int32'), y_true_sparse, normalize=True)

        self.cer_accumulator.assign_add(tf.reduce_sum(distance))
        self.counter.assign_add(len(y_true))

    def result(self):
        return tf.math.divide_no_nan(self.cer_accumulator, self.counter)

    def reset_states(self):
        self.cer_accumulator.assign(0.0)
        self.counter.assign(0.0)


def CTCLoss(y_true, y_pred):
    print(y_true.shape)
    print(y_pred.shape)
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    print(input_length)
    print(label_length)
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


model = keras.Sequential()
model.add(layers.Masking(input_shape=(778, 26)))
model.add(layers.Bidirectional(layers.LSTM(return_sequences=True, units=96)))
model.add(layers.TimeDistributed(layers.Dense(units=27, activation="softmax")))

model.compile(optimizer='adam', loss=CTCLoss)

model.summary()


model.fit(x=trainfeatures, y=trainlabels, batch_size=32, epochs=120, validation_split=0.05)

model.summary()

model.save('ctc_model.h5')



