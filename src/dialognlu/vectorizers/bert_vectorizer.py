# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


class BERTVectorizer:
    
    def __init__(self, is_bert, bert_model_hub_path):
        self.is_bert = is_bert
        self.bert_model_hub_path = bert_model_hub_path
        self.create_tokenizer_from_hub_module(is_bert=is_bert)
    
        
    def create_tokenizer_from_hub_module(self, is_bert):
        """Get the vocab file and casing info from the Hub module."""
        # bert_module =  hub.Module(self.bert_model_hub_path)
        module_layer = hub.KerasLayer(self.bert_model_hub_path,
                              trainable=False)
        
        if is_bert:
            from .tokenization import FullTokenizer
            vocab_file = module_layer.resolved_object.vocab_file.asset_path.numpy()
            do_lower_case = module_layer.resolved_object.do_lower_case.numpy()
            self.tokenizer = FullTokenizer(vocab_file, do_lower_case)
        else:
            sp_model_file = module_layer.resolved_object.sp_model_file.asset_path.numpy()
            
            # commented and used the below instead because of lower case problem
            # from vectorizers.tokenization import FullSentencePieceTokenizer
            # self.tokenizer = FullSentencePieceTokenizer(sp_model_file)
            from .albert_tokenization import FullTokenizer
            self.tokenizer = FullTokenizer(vocab_file=sp_model_file, 
                                        do_lower_case=True,
                                        spm_model_file=sp_model_file)
        
        del module_layer
    
    
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
        result = {"input_word_ids": input_ids, "input_mask": input_mask, "input_type_ids": segment_ids,
                    "valid_positions": valid_positions, "sequence_lengths": sequence_lengths}
        return result
    
    
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