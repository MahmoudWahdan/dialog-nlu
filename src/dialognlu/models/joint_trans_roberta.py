# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Multiply, TimeDistributed
from tensorflow.keras.models import Model

from .base_joint_trans import (BaseJointTransformerModel,
                               TfliteBaseJointTransformer4inputsModel,
                               TfliteBaseJointTransformerModel)


class JointTransRobertaModel(BaseJointTransformerModel):

    def __init__(self, config, trans_model=None, is_load=False):
        super(JointTransRobertaModel, self).__init__(config, trans_model, is_load)
        

    def build_model(self):
        in_id = Input(shape=(self.max_length), name='input_word_ids', dtype=tf.int32)
        in_mask = Input(shape=(self.max_length), name='input_mask', dtype=tf.int32)
        in_segment = Input(shape=(self.max_length), name='input_type_ids', dtype=tf.int32)
        in_valid_positions = Input(shape=(self.max_length, self.slots_num), name='valid_positions')
        bert_inputs = [in_id, in_mask, in_segment]
        inputs = bert_inputs + [in_valid_positions]
        

        bert_sequence_output, bert_pooled_output = self.trans_model(bert_inputs)
        
        intents_fc = Dense(self.intents_num, activation='softmax', name='intent_classifier')(bert_pooled_output)
        
        slots_output = TimeDistributed(Dense(self.slots_num, activation='softmax'))(bert_sequence_output)
        slots_output = Multiply(name='slots_tagger')([slots_output, in_valid_positions])
        
        self.model = Model(inputs=inputs, outputs=[slots_output, intents_fc])
            

    def save(self, model_path):
        self.save_to_path(model_path, 'joint_roberta_model.h5')
    
        
    @staticmethod
    def load(load_folder_path):
        return BaseJointTransformerModel.load_model_by_class(JointTransRobertaModel, load_folder_path, 'joint_roberta_model.h5')


class TfliteJointTransRobertaModel(TfliteBaseJointTransformer4inputsModel):

    def __init__(self, config):
        super(TfliteJointTransRobertaModel, self).__init__(config)

    @staticmethod
    def load(path):
        return TfliteBaseJointTransformerModel.load_model_by_class(TfliteJointTransRobertaModel, path)
