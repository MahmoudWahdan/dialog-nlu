# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

import tensorflow as tf
import tensorflow.python.keras.backend as K
import tensorflow_hub as hub


class AlbertLayer(tf.keras.layers.Layer):
    
    def __init__(self, fine_tune=True, pooling='first',
        albert_path="https://tfhub.dev/google/albert_base/1",
        **kwargs,):
        self.fine_tune = fine_tune
#        self.trainable = True
        self.output_size = 768
        self.pooling = pooling
        self.albert_path = albert_path
        if self.pooling not in ['first', 'mean']:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        super(AlbertLayer, self).__init__(**kwargs)
        

    def build(self, input_shape):
        self.albert = hub.Module(
            self.albert_path, trainable=self.fine_tune, name=f"{self.name}_module"
        )
        
        if self.fine_tune:
            # Remove unused layers
            trainable_vars = self.albert.variables
            if self.pooling == 'first':
                trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
                trainable_layers = ["pooler/dense"]
    
            elif self.pooling == 'mean':
                trainable_vars = [var for var in trainable_vars
                    if not "/cls/" in var.name and not "/pooler/" in var.name
                ]
                trainable_layers = []
            else:
                raise NameError(
                    f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
                )
    
            # Select how many layers to fine tune
            trainable_layers.append("encoder/transformer/group_0")
    
            # Update trainable vars to contain only the specified layers
            trainable_vars = [
                var
                for var in trainable_vars
                if any([l in var.name for l in trainable_layers])
            ]
    
            # Add to trainable weights
            for var in trainable_vars:
                self._trainable_weights.append(var)
    
            for var in self.albert.variables:
                if var not in self._trainable_weights:
                    self._non_trainable_weights.append(var)
        
        super(AlbertLayer, self).build(input_shape)

    
    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids, valid_positions = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.albert(inputs=bert_inputs, signature='tokens', as_dict=True)
        return result['pooled_output'], result['sequence_output']
    

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_size)
    
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'fine_tune': self.fine_tune,
#            'trainable': self.trainable,
            'output_size': self.output_size,
            'pooling': self.pooling,
            'albert_path': self.albert_path,
        })
        return config
