# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense
from models.joint_bert import JointBertModel
from layers.bert_layer import BertLayer
from layers.albert_layer import AlbertLayer
from layers.crf_layer import CRFLayer
import numpy as np
import os
import json


class JointBertCRFModel(JointBertModel):
    

    def __init__(self, slots_num, intents_num, bert_hub_path, sess, num_bert_fine_tune_layers=10,
                 is_bert=True):
        super(JointBertCRFModel, self).__init__(slots_num, intents_num, bert_hub_path, sess, 
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
        in_id = Input(shape=(None,), name='input_ids')
        in_mask = Input(shape=(None,), name='input_masks')
        in_segment = Input(shape=(None,), name='segment_ids')
        in_valid_positions = Input(shape=(None, self.slots_num), name='valid_positions')
        sequence_lengths = Input(shape=(1), dtype='int32', name='sequence_lengths')
        
        bert_inputs = [in_id, in_mask, in_segment, in_valid_positions]
        
        if self.is_bert:
            bert_pooled_output, bert_sequence_output = BertLayer(
                n_fine_tune_layers=self.num_bert_fine_tune_layers,
                bert_path=self.bert_hub_path,
                pooling='mean', name='BertLayer')(bert_inputs)
        else:
            bert_pooled_output, bert_sequence_output = AlbertLayer(
                fine_tune=True if self.num_bert_fine_tune_layers > 0 else False,
                albert_path=self.bert_hub_path,
                pooling='mean', name='AlbertLayer')(bert_inputs)
        
        intents_fc = Dense(self.intents_num, activation='softmax', name='intent_classifier')(bert_pooled_output)
        
        self.crf = CRFLayer(name='slots_tagger')
        slots_output = self.crf(inputs=[bert_sequence_output, sequence_lengths])
        
        self.model = Model(inputs=bert_inputs + [sequence_lengths], outputs=[slots_output, intents_fc])

        
    def fit(self, X, Y, validation_data=None, epochs=5, batch_size=32):
        """
        X: batch of [input_ids, input_mask, segment_ids, valid_positions]
        """
        X = (X[0], X[1], X[2], self.prepare_valid_positions(X[3]), X[4])
        if validation_data is not None:
            X_val, Y_val = validation_data
            validation_data = ((X_val[0], X_val[1], X_val[2], 
                                self.prepare_valid_positions(X_val[3]), X_val[4]), Y_val)
        
        history = self.model.fit(X, Y, validation_data=validation_data, 
                                 epochs=epochs, batch_size=batch_size)
        self.visualize_metric(history.history, 'slots_tagger_loss')
        self.visualize_metric(history.history, 'intent_classifier_loss')
        self.visualize_metric(history.history, 'loss')
        self.visualize_metric(history.history, 'intent_classifier_acc')
        
        
    def predict_slots_intent(self, x, slots_vectorizer, intent_vectorizer, remove_start_end=True):
        valid_positions = x[3]
        x = (x[0], x[1], x[2], self.prepare_valid_positions(valid_positions), x[4])
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
        
        
    def load(load_folder_path, sess):
        with open(os.path.join(load_folder_path, 'params.json'), 'r') as json_file:
            model_params = json.load(json_file)
            
        slots_num = model_params['slots_num'] 
        intents_num = model_params['intents_num']
        bert_hub_path = model_params['bert_hub_path']
        num_bert_fine_tune_layers = model_params['num_bert_fine_tune_layers']
        is_bert = model_params['is_bert']
            
        new_model = JointBertCRFModel(slots_num, intents_num, bert_hub_path, sess, num_bert_fine_tune_layers, is_bert)
        new_model.model.load_weights(os.path.join(load_folder_path,'joint_bert_crf_model.h5'))
        return new_model