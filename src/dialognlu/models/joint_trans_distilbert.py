# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""
    
from .base_joint_trans import BaseJointTransformerModel, TfliteBaseJointTransformerModel
from .callbacks import F1Metrics
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Multiply, TimeDistributed, Lambda, GlobalAveragePooling1D
import numpy as np


class JointTransDistilBertModel(BaseJointTransformerModel):

    def __init__(self, config, trans_model=None, is_load=False):
        super(JointTransDistilBertModel, self).__init__(config, trans_model, is_load)
        

    def build_model(self):
        in_id = Input(shape=(self.max_length), name='input_word_ids', dtype=tf.int32)
        in_mask = Input(shape=(self.max_length), name='input_mask', dtype=tf.int32)
        in_valid_positions = Input(shape=(self.max_length, self.slots_num), name='valid_positions')
        bert_inputs = [in_id, in_mask]
        inputs = bert_inputs + [in_valid_positions]
        
        bert_sequence_output = self.trans_model(bert_inputs)[0]
#        bert_pooled_output = Lambda(function=lambda x: tf.keras.backend.mean(x, axis=1))(bert_sequence_output)
        bert_pooled_output = GlobalAveragePooling1D()(bert_sequence_output)
        
        intents_fc = Dense(self.intents_num, activation='softmax', name='intent_classifier')(bert_pooled_output)
        
        slots_output = TimeDistributed(Dense(self.slots_num, activation='softmax'))(bert_sequence_output)
        slots_output = Multiply(name='slots_tagger')([slots_output, in_valid_positions])
        
        self.model = Model(inputs=inputs, outputs=[slots_output, intents_fc])

        
    def fit(self, X, Y, validation_data=None, epochs=5, batch_size=32, id2label=None):
        """
        X: batch of [input_ids, input_mask, segment_ids, valid_positions]
        """
        X["valid_positions"] = self.prepare_valid_positions(X["valid_positions"])
        if validation_data is not None:
            X_val, Y_val = validation_data
            X_val["valid_positions"] = self.prepare_valid_positions(X_val["valid_positions"])
            validation_data = (X_val, Y_val)
        
        callbacks = [F1Metrics(id2label, validation_data=validation_data)]
        history = self.model.fit(X, Y, validation_data=validation_data, 
                                 epochs=epochs, batch_size=batch_size,
                                 callbacks=callbacks)
        self.visualize_metric(history.history, 'slots_tagger_loss')
        self.visualize_metric(history.history, 'intent_classifier_loss')
        self.visualize_metric(history.history, 'loss')
        self.visualize_metric(history.history, 'intent_classifier_acc')
                
        
    def predict_slots_intent(self, x, slots_vectorizer, intent_vectorizer, remove_start_end=True,
                             include_intent_prob=False):
        valid_positions = x["valid_positions"]
        x["valid_positions"] = self.prepare_valid_positions(valid_positions)
        y_slots, y_intent = self.predict(x)
        slots = slots_vectorizer.inverse_transform(y_slots, valid_positions)
        if remove_start_end:
            slots = [x[1:-1] for x in slots]
            
        if not include_intent_prob:
            intents = np.array([intent_vectorizer.inverse_transform([np.argmax(i)])[0] for i in y_intent])
        else:
            intents = np.array([(intent_vectorizer.inverse_transform([np.argmax(i)])[0], round(float(np.max(i)), 4)) for i in y_intent])
        return slots, intents
    
    
    def save(self, model_path):
        self.save_to_path(model_path, 'joint_distilbert_model.h5')
    
        
    @staticmethod
    def load(load_folder_path):
        return BaseJointTransformerModel.load_model_by_class(JointTransDistilBertModel, load_folder_path, 'joint_distilbert_model.h5')



class TfliteJointTransDistilBertModel(TfliteBaseJointTransformerModel):

    def __init__(self, config):
        super(TfliteJointTransDistilBertModel, self).__init__(config)

    def predict(self, inputs):
        self.interpreter.set_tensor(self.interpreter.get_input_details()[0]["index"], inputs.get("input_word_ids").astype(np.int32))
        self.interpreter.set_tensor(self.interpreter.get_input_details()[1]["index"], inputs.get("input_mask").astype(np.int32))
        self.interpreter.set_tensor(self.interpreter.get_input_details()[2]["index"], inputs.get("valid_positions").astype(np.float32))
        output_index_0 = self.interpreter.get_output_details()[0]["index"]
        output_index_1 = self.interpreter.get_output_details()[1]["index"]
        self.interpreter.invoke()
        intent = self.interpreter.get_tensor(output_index_0)
        slots = self.interpreter.get_tensor(output_index_1)
        return slots, intent

    @staticmethod
    def load(path):
        return TfliteBaseJointTransformerModel.load_model_by_class(TfliteJointTransDistilBertModel, path)