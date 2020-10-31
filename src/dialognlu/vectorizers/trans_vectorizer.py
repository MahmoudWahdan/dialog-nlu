# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

import tensorflow as tf
from transformers import AutoTokenizer
import numpy as np


class TransVectorizer:
    
    def __init__(self, pretrained_model_name_or_path, max_length=None, cache_dir=None):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, cache_dir=cache_dir)
        self.tokenizer_type = self.tokenizer.__class__.__name__
        self.valid_start = None
        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token
        if self.tokenizer_type == 'BertTokenizer':
            self.not_valid_start = '##'
        elif self.tokenizer_type == 'DistilBertTokenizer':
            self.not_valid_start = '##'
        elif self.tokenizer_type == 'AlbertTokenizer':
            self.valid_start = '▁'
        elif self.tokenizer_type == 'XLNetTokenizer':
            self.valid_start = '▁'
        elif self.tokenizer_type == 'RobertaTokenizer':
            self.valid_start = 'Ġ'
        else:
            raise Exception('%s is not supported tokenizer' % self.tokenizer_type)
            
            
#    def tokenize(self, text: str):
#        tokens = self.tokenizer.tokenize(text)
#        valid_positions = []
#        for i, token in enumerate(tokens):
#            if self.valid_start is not None:
#                # TODO: 
#                raise NotImplementedError('To be implemented')
#            elif self.not_valid_start is not None:
#                if token.startswith(self.not_valid_start):
#                    valid_positions.append(0)
#                else:
#                    valid_positions.append(1)
#            else:
#                raise Exception('either valid_start or not_valid_start should be not None')
#        return tokens, valid_positions
            
            
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
        input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids, padding='post', maxlen=self.max_length)
        input_mask = tf.keras.preprocessing.sequence.pad_sequences(input_mask, padding='post', maxlen=self.max_length)
        segment_ids = tf.keras.preprocessing.sequence.pad_sequences(segment_ids, padding='post', maxlen=self.max_length)
        valid_positions = tf.keras.preprocessing.sequence.pad_sequences(valid_positions, padding='post', maxlen=self.max_length)
        result = {"input_word_ids": input_ids, "input_mask": input_mask, "input_type_ids": segment_ids,
                    "valid_positions": valid_positions, "sequence_lengths": sequence_lengths}
        # set new max_length if None
        if self.max_length is None:
            self.max_length = input_ids.shape[1]
        return result
    
    
    def __vectorize(self, text: str):
        tokens, valid_positions = self.tokenize(text)
        # insert cls token
        tokens.insert(0, self.cls_token)
        valid_positions.insert(0, 1)
        # insert sep token
        tokens.append(self.sep_token)
        valid_positions.append(1)
        
        segment_ids = [0] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)
        
        return input_ids, input_mask, segment_ids, valid_positions
        
        
#    def vectorize(self, text):
#        input_ids = []
#        attention_mask = []
#        token_type_ids = []
#        valid_positions = []
#        for t in text:
#            d = self.tokenizer.encode_plus(t,  add_special_tokens=True)#, pad_to_max_length=True, max_length=self.max_length, return_length=True)
#            _id = d['input_ids']
#            input_ids.append(_id)
#            attention_mask.append(d['attention_mask'])
#            token_type_ids.append(d['token_type_ids'])
#            valid = [1] * len(_id)
#            valid[0] = valid[-1] = 0
#            valid_positions.append(valid)
#            
#        sequence_lengths = np.array([len(i) for i in input_ids]) 
#        input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids, padding='post')
#        attention_mask = tf.keras.preprocessing.sequence.pad_sequences(attention_mask, padding='post')
#        token_type_ids = tf.keras.preprocessing.sequence.pad_sequences(token_type_ids, padding='post')
#        valid_positions = tf.keras.preprocessing.sequence.pad_sequences(valid_positions, padding='post')
#        return input_ids, attention_mask, token_type_ids, valid_positions, sequence_lengths