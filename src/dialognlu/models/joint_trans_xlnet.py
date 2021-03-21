# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Multiply, TimeDistributed, Lambda, GlobalAveragePooling1D
from .callbacks import F1Metrics
import numpy as np
import os
import json

from .base_joint_trans import (BaseJointTransformerModel,
                               TfliteBaseJointTransformer4inputsModel,
                               TfliteBaseJointTransformerModel)


class JointTransXlnetModel(BaseJointTransformerModel):

    def __init__(self, config, trans_model=None, is_load=False):
        super(JointTransXlnetModel, self).__init__(config, trans_model, is_load)
        

    def build_model(self):
        in_id = Input(shape=(self.max_length), name='input_word_ids', dtype=tf.int32)
        in_mask = Input(shape=(self.max_length), name='input_mask', dtype=tf.int32)
        in_segment = Input(shape=(self.max_length), name='input_type_ids', dtype=tf.int32)
        in_valid_positions = Input(shape=(self.max_length, self.slots_num), name='valid_positions')
        bert_inputs = {'input_ids': in_id, 'token_type_ids': in_segment, 'attention_mask': in_mask}
        inputs = [in_id, in_mask, in_segment] + [in_valid_positions]

        bert_sequence_output_all = self.trans_model(bert_inputs)
        bert_sequence_output = bert_sequence_output_all[0]
        bert_pooled_output = GlobalAveragePooling1D()(bert_sequence_output)        
        
        intents_fc = Dense(self.intents_num, activation='softmax', name='intent_classifier')(bert_pooled_output)
        
        slots_output = TimeDistributed(Dense(self.slots_num, activation='softmax'))(bert_sequence_output)
        slots_output = Multiply(name='slots_tagger')([slots_output, in_valid_positions])
        
        self.model = Model(inputs=inputs, outputs=[slots_output, intents_fc])
        
        
    def prepare_valid_positions(self, in_valid_positions):
        in_valid_positions = np.expand_dims(in_valid_positions, axis=2)
        in_valid_positions = np.tile(in_valid_positions, (1, 1, self.slots_num))
        return in_valid_positions
    
    
    def save(self, model_path):
        self.save_to_path(model_path, 'model')
    
    
    @staticmethod
    def load(load_folder_path):
        return BaseJointTransformerModel.load_model_by_class(JointTransXlnetModel, load_folder_path, 'model')


class TfliteJointTransXlnetModel(TfliteBaseJointTransformer4inputsModel):

    def __init__(self, config):
        super(TfliteJointTransXlnetModel, self).__init__(config)

    @staticmethod
    def load(path):
        return TfliteBaseJointTransformerModel.load_model_by_class(TfliteJointTransXlnetModel, path)
