# -*- coding: utf-8 -*-
"""
Original Implementation from: https://github.com/Hironsan/keras-crf-layer

Adapted tensorflow implementation by:
@author: mwahdan
"""


import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.contrib.crf import crf_decode


class CRFLayer(Layer):

    def __init__(self, transition_params=None, **kwargs):
        super(CRFLayer, self).__init__(**kwargs)
        self.transition_params = transition_params
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=2)]
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape[0]) == 3

        return input_shape[0]

    def build(self, input_shape):
        """Creates the layer weights.

        Args:
            input_shape (list(tuple, tuple)): [(batch_size, n_steps, n_classes), (batch_size, 1)]
        """
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 3
        assert len(input_shape[1]) == 2
        n_steps = input_shape[0][1].value
        n_classes = input_shape[0][2].value
        assert n_steps is None or n_steps >= 2

        self.transition_params = self.add_weight(shape=(n_classes, n_classes),
                                                 initializer='uniform',
                                                 name='transition')
        self.input_spec = [InputSpec(dtype=K.floatx(), shape=(None, n_steps, n_classes)),
                           InputSpec(dtype='int32', shape=(None, 1))]
        self.built = True

    def viterbi_decode(self, potentials, sequence_length):
        """Decode the highest scoring sequence of tags in TensorFlow.

        This is a function for tensor.

        Args:
            potentials: A [batch_size, max_seq_len, num_tags] tensor, matrix of unary potentials.
            sequence_length: A [batch_size] tensor, containing sequence lengths.

        Returns:
            decode_tags: A [batch_size, max_seq_len] tensor, with dtype tf.int32.
                         Contains the highest scoring tag indicies.
        """
        decode_tags, best_score = crf_decode(potentials, self.transition_params, sequence_length)

        return decode_tags

    def call(self, inputs, mask=None, **kwargs):
        inputs, sequence_lengths = inputs
#        self.sequence_lengths = K.flatten(sequence_lengths)
        self.sequence_lengths = tf.reshape(sequence_lengths, [-1])
        y_pred = self.viterbi_decode(inputs, self.sequence_lengths)
        nb_classes = self.input_spec[0].shape[2]
#        y_pred_one_hot = K.one_hot(y_pred, nb_classes)
        y_pred_one_hot = tf.one_hot(y_pred, nb_classes)

        return K.in_train_phase(inputs, y_pred_one_hot)

    def loss(self, y_true, y_pred):
        """Computes the log-likelihood of tag sequences in a CRF.

        Args:
            y_true : A (batch_size, n_steps, n_classes) tensor.
            y_pred : A (batch_size, n_steps, n_classes) tensor.

        Returns:
            loss: A scalar containing the log-likelihood of the given sequence of tag indices.
        """
#        y_true = K.cast(K.argmax(y_true, axis=-1), dtype='int32')
        y_true = tf.cast(tf.argmax(y_true, axis=-1), dtype='int32')
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            y_pred, y_true, self.sequence_lengths, self.transition_params)
        loss = tf.reduce_mean(-log_likelihood)

        return loss

    def get_config(self):
        config = {
            'transition_params': K.eval(self.transition_params),
        }
        base_config = super(CRFLayer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


def create_custom_objects():
    """Returns the custom objects, needed for loading a persisted model."""
    instanceHolder = {'instance': None}

    class ClassWrapper(CRFLayer):
        def __init__(self, *args, **kwargs):
            instanceHolder['instance'] = self
            super(ClassWrapper, self).__init__(*args, **kwargs)

    def loss(*args):
        method = getattr(instanceHolder['instance'], 'loss')
        return method(*args)

    return {'CRFLayer': ClassWrapper, 'loss': loss}
