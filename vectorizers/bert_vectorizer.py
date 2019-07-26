# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from bert.tokenization import FullTokenizer


class BERTVectorizer:
    
    def __init__(self, sess,
                 bert_model_hub_path='https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1'):
        self.sess = sess
        self.bert_model_hub_path = bert_model_hub_path
        self.create_tokenizer_from_hub_module()
    
        
    def create_tokenizer_from_hub_module(self):
        """Get the vocab file and casing info from the Hub module."""
        bert_module =  hub.Module(self.bert_model_hub_path)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        vocab_file, do_lower_case = self.sess.run(
            [
                tokenization_info["vocab_file"],
                tokenization_info["do_lower_case"],
            ]
        )
    
        self.tokenizer = FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    
    
    def tokenize(self, text: str):
        words = text.split() # whitespace tokenizer
        tokens = []
        valid_positions = []
        for i, word in enumerate(words):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)
        return tokens, valid_positions
    
    
    def transform(self, text_arr):
        input_ids = []
        input_mask = []
        segment_ids = []
        valid_positions = []
        for text in text_arr:
            ids, mask, seg_ids, valid_pos = self.__vectorize(text)
            input_ids.append(ids)
            input_mask.append(mask)
            segment_ids.append(seg_ids)
            valid_positions.append(valid_pos)

        sequence_lengths = np.array([len(i) for i in input_ids])            
        input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids, padding='post')
        input_mask = tf.keras.preprocessing.sequence.pad_sequences(input_mask, padding='post')
        segment_ids = tf.keras.preprocessing.sequence.pad_sequences(segment_ids, padding='post')
        valid_positions = tf.keras.preprocessing.sequence.pad_sequences(valid_positions, padding='post')
        return input_ids, input_mask, segment_ids, valid_positions, sequence_lengths
    
    
    def __vectorize(self, text: str):
        tokens, valid_positions = self.tokenize(text)
        # insert "[CLS]"
        tokens.insert(0, '[CLS]')
        valid_positions.insert(0, 1)
        # insert "[SEP]"
        tokens.append('[SEP]')
        valid_positions.append(1)
        
        segment_ids = [0] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)
        
        return input_ids, input_mask, segment_ids, valid_positions