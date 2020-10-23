# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

from .joint_bert import JointBertModel
from ..layers.crf_layer import CRFLayer
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import tensorflow_hub as hub
import numpy as np
import os
import json


class JointBertCRFModel(JointBertModel):
    

    def __init__(self, slots_num, intents_num, bert_hub_path, num_bert_fine_tune_layers=10,
                 is_bert=True):
        super(JointBertCRFModel, self).__init__(slots_num, intents_num, bert_hub_path, 
             num_bert_fine_tune_layers, is_bert)
        
        
    def compile_model(self):
        # Instead of `using categorical_crossentropy`, 
        # we use `sparse_categorical_crossentropy`, which does expect integer targets.
        
        optimizer = tf.keras.optimizers.Adam(lr=5e-5)#0.001)

        losses = {
        	'slots_tagger': self.crf.loss,
        	'intent_classifier': 'sparse_categorical_crossentropy',
        }
        loss_weights = {'slots_tagger': 3.0, 'intent_classifier': 1.0}
        metrics = {'intent_classifier': 'acc'}
        self.model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)
        self.model.summary()
        

    def build_model(self):
        in_id = Input(shape=(None,), name='input_word_ids', dtype=tf.int32)
        in_mask = Input(shape=(None,), name='input_mask', dtype=tf.int32)
        in_segment = Input(shape=(None,), name='input_type_ids', dtype=tf.int32)
        # in_valid_positions = Input(shape=(None, self.slots_num), name='valid_positions')
        sequence_lengths = Input(shape=(1), dtype='int32', name='sequence_lengths')
        
        bert_inputs = [in_id, in_mask, in_segment]
        inputs = bert_inputs + [sequence_lengths]# [in_valid_positions, sequence_lengths]
        
        if self.is_bert:
            name = 'BertLayer'
        else:
            name = 'AlbertLayer'
        bert_pooled_output, bert_sequence_output = hub.KerasLayer(self.bert_hub_path,
                              trainable=True, name=name)(bert_inputs)
        
        intents_fc = Dense(self.intents_num, activation='softmax', name='intent_classifier')(bert_pooled_output)
        
        self.crf = CRFLayer(name='slots_tagger')
        slots_output = self.crf(inputs=[bert_sequence_output, sequence_lengths])
        
        self.model = Model(inputs=inputs, outputs=[slots_output, intents_fc])

        
    def fit(self, X, Y, validation_data=None, epochs=5, batch_size=32, id2label=None):
        # X["valid_positions"] = self.prepare_valid_positions(X["valid_positions"])
        # if validation_data is not None:
        #     X_val, Y_val = validation_data
        #     X_val["valid_positions"] = self.prepare_valid_positions(X_val["valid_positions"])
        #     validation_data = (X_val, Y_val)
        
        history = self.model.fit(X, Y, validation_data=validation_data, 
                                 epochs=epochs, batch_size=batch_size)
        self.visualize_metric(history.history, 'slots_tagger_loss')
        self.visualize_metric(history.history, 'intent_classifier_loss')
        self.visualize_metric(history.history, 'loss')
        self.visualize_metric(history.history, 'intent_classifier_acc')
        
        
    def predict_slots_intent(self, x, slots_vectorizer, intent_vectorizer, remove_start_end=True):
        valid_positions = x["valid_positions"]
        # x["valid_positions"] = self.prepare_valid_positions(valid_positions)
        y_slots, y_intent = self.predict(x)
        slots = slots_vectorizer.inverse_transform(y_slots, valid_positions)
        if remove_start_end:
            slots = [x[1:-1] for x in slots]
            
        intents = np.array([intent_vectorizer.inverse_transform([np.argmax(y_intent[i])])[0] for i in range(y_intent.shape[0])])
        return slots, intents
    

    def save(self, model_path):
        with open(os.path.join(model_path, 'params.json'), 'w') as json_file:
            json.dump(self.model_params, json_file)
        self.model.save(os.path.join(model_path, 'joint_bert_crf_model.h5'))
        

    @staticmethod    
    def load(load_folder_path):
        with open(os.path.join(load_folder_path, 'params.json'), 'r') as json_file:
            model_params = json.load(json_file)
            
        slots_num = model_params['slots_num'] 
        intents_num = model_params['intents_num']
        bert_hub_path = model_params['bert_hub_path']
        num_bert_fine_tune_layers = model_params['num_bert_fine_tune_layers']
        is_bert = model_params['is_bert']
            
        new_model = JointBertCRFModel(slots_num, intents_num, bert_hub_path, num_bert_fine_tune_layers, is_bert)
        new_model.model.load_weights(os.path.join(load_folder_path,'joint_bert_crf_model.h5'))
        return new_model