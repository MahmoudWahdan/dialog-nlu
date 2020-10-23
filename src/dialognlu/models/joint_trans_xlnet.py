# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Multiply, TimeDistributed, Lambda, GlobalAveragePooling1D
from .base_joint_trans import BaseJointTransformerModel
import numpy as np
import os
import json


class JointTransXlnetModel(BaseJointTransformerModel):

    def __init__(self, config, trans_model=None, is_load=False):
        super(JointTransXlnetModel, self).__init__(config, trans_model, is_load)
#        self.slots_num = config.get('slots_num')
#        self.intents_num = config.get('intents_num')
#        self.pretrained_model_name_or_path = config.get('pretrained_model_name_or_path')
#        self.cache_dir = config.get('cache_dir', None)
#        self.from_pt = config.get('from_pt', False)
#        self.num_bert_fine_tune_layers = config.get('num_bert_fine_tune_layers', 10)
#        
#        self.model_params = config
#        
#        if not is_load:
#            self.trans_model = trans_model
#            self.build_model()
#            self.compile_model()
        
        
#    def compile_model(self):
#        # Instead of `using categorical_crossentropy`, 
#        # we use `sparse_categorical_crossentropy`, which does expect integer targets.
#        
#        optimizer = tf.keras.optimizers.Adam(lr=5e-5)#0.001)
#
#        losses = {
#        	'slots_tagger': 'sparse_categorical_crossentropy',
#        	'intent_classifier': 'sparse_categorical_crossentropy',
#        }
#        loss_weights = {'slots_tagger': 3.0, 'intent_classifier': 1.0}
#        metrics = {'intent_classifier': 'acc'}
#        self.model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)
#        self.model.summary()
        

    def build_model(self):
        in_id = Input(shape=(None,), name='input_word_ids', dtype=tf.int32)
        in_mask = Input(shape=(None,), name='input_mask', dtype=tf.int32)
        in_segment = Input(shape=(None,), name='input_type_ids', dtype=tf.int32)
        in_valid_positions = Input(shape=(None, self.slots_num), name='valid_positions')
        bert_inputs = [in_id, in_mask, in_segment]
        inputs = bert_inputs + [in_valid_positions]
        

#        bert_sequence_output, bert_pooled_output = self.trans_model(bert_inputs)
        bert_sequence_output_all = self.trans_model(bert_inputs)
        bert_sequence_output = bert_sequence_output_all[0]
#        bert_pooled_output = Lambda(function=lambda x: tf.keras.backend.mean(x, axis=1))(bert_sequence_output)
        bert_pooled_output = GlobalAveragePooling1D()(bert_sequence_output)
        
        
        intents_fc = Dense(self.intents_num, activation='softmax', name='intent_classifier')(bert_pooled_output)
        
        slots_output = TimeDistributed(Dense(self.slots_num, activation='softmax'))(bert_sequence_output)
        slots_output = Multiply(name='slots_tagger')([slots_output, in_valid_positions])
        
        self.model = Model(inputs=inputs, outputs=[slots_output, intents_fc])

        
    def fit(self, X, Y, validation_data=None, epochs=5, batch_size=32):
        """
        X: batch of [input_ids, input_mask, segment_ids, valid_positions]
        """
        X = (X[0], X[1], X[2], self.prepare_valid_positions(X[3]))
        if validation_data is not None:
            X_val, Y_val = validation_data
            validation_data = ((X_val[0], X_val[1], X_val[2], self.prepare_valid_positions(X_val[3])), Y_val)
        
        history = self.model.fit(X, Y, validation_data=validation_data, 
                                 epochs=epochs, batch_size=batch_size)
        self.visualize_metric(history.history, 'slots_tagger_loss')
        self.visualize_metric(history.history, 'intent_classifier_loss')
        self.visualize_metric(history.history, 'loss')
        self.visualize_metric(history.history, 'intent_classifier_acc')
        
        
    def prepare_valid_positions(self, in_valid_positions):
        in_valid_positions = np.expand_dims(in_valid_positions, axis=2)
        in_valid_positions = np.tile(in_valid_positions, (1, 1, self.slots_num))
        return in_valid_positions
    
                
        
    def predict_slots_intent(self, x, slots_vectorizer, intent_vectorizer, remove_start_end=True,
                             include_intent_prob=False):
        valid_positions = x[3]
        x = (x[0], x[1], x[2], self.prepare_valid_positions(valid_positions))
        y_slots, y_intent = self.predict(x)
        slots = slots_vectorizer.inverse_transform(y_slots, valid_positions)
        if remove_start_end:
            slots = [x[1:-1] for x in slots]
            
        if not include_intent_prob:
            intents = np.array([intent_vectorizer.inverse_transform([np.argmax(i)])[0] for i in y_intent])
        else:
            intents = np.array([(intent_vectorizer.inverse_transform([np.argmax(i)])[0], round(float(np.max(i)), 4)) for i in y_intent])
        return slots, intents
    

#    def save(self, model_path):
#        self.model_params["class"] = self.__class__.__name__
#        with open(os.path.join(model_path, 'params.json'), 'w') as json_file:
#            json.dump(self.model_params, json_file)
#        self.model.save(os.path.join(model_path, 'joint_xlnet_model.h5'))
#        
#        
#    def load(load_folder_path):
#        with open(os.path.join(load_folder_path, 'params.json'), 'r') as json_file:
#            model_params = json.load(json_file)
#                        
#        new_model = JointTransXlnetModel(model_params, trans_model=None, is_load=True)
#        new_model.model = tf.keras.models.load_model(os.path.join(load_folder_path,'joint_xlnet_model.h5'))
#        return new_model
    
    def save(self, model_path):
        self.save_to_path(model_path, 'joint_xlnet_model.h5')
    
        
    def load(load_folder_path):
        return BaseJointTransformerModel.load_model_by_class(JointTransXlnetModel, load_folder_path, 'joint_xlnet_model.h5')
