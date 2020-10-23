# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle


class TagsVectorizer:
    
    def __init__(self):
        pass
    
    def tokenize(self, tags_str_arr):
        return [s.split() for s in tags_str_arr]
    
    def fit(self, tags_str_arr):
        self.label_encoder = LabelEncoder()
        data = ['<PAD>'] + [item for sublist in self.tokenize(tags_str_arr) for item in sublist]
        self.label_encoder.fit(data)
    
    def transform(self, tags_str_arr, valid_positions):
        seq_length = valid_positions.shape[1]
        data = self.tokenize(tags_str_arr)
        data = [self.label_encoder.transform(['O'] + x + ['O']).astype(np.int32) for x in data]

        output = np.zeros((len(data), seq_length))
        for i in range(len(data)):
            idx = 0
            for j in range(seq_length):
                if valid_positions[i][j] == 1:
                    output[i][j] = data[i][idx]
                    idx += 1
        return output
    
    def inverse_transform(self, model_output_3d, valid_positions):
        seq_length = valid_positions.shape[1]
        slots = np.argmax(model_output_3d, axis=-1)
        slots = [self.label_encoder.inverse_transform(y) for y in slots]
        output = []
        for i in range(len(slots)):
            y = []
            for j in range(seq_length):
                if valid_positions[i][j] == 1:
                    y.append(str(slots[i][j]))
            output.append(y)
        return output
    
    @staticmethod
    def load(path):
        vectorizer = TagsVectorizer()
        with open(path, 'rb') as handle:
            vectorizer.label_encoder = pickle.load(handle)
        return vectorizer
    
    def save(self, path):
        with open(path, 'wb') as handle:
            pickle.dump(self.label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
