# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

from readers.goo_format_reader import Reader
from vectorizers.bert_vectorizer import BERTVectorizer
from vectorizers.tags_vectorizer import TagsVectorizer
from models.joint_bert import JointBertModel

from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import pickle
import tensorflow as tf


tf.compat.v1.random.set_random_seed(7)


sess = tf.compat.v1.Session()

save_folder_path = 'saved_models/joint_bert_model'
epochs = 5
batch_size = 64
#max_seq_length = 52 # 50 + start + end
bert_model_hub_path = 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1'

train_text_arr, train_tags_arr, train_intents = Reader.read('data/atis/train')
val_text_arr, val_tags_arr, val_intents = Reader.read('data/atis/valid')



bert_vectorizer = BERTVectorizer(sess, bert_model_hub_path)
train_input_ids, train_input_mask, train_segment_ids, train_valid_positions = bert_vectorizer.transform(train_text_arr)
val_input_ids, val_input_mask, val_segment_ids, val_valid_positions = bert_vectorizer.transform(val_text_arr)


tags_vectorizer = TagsVectorizer()
tags_vectorizer.fit(train_tags_arr)
train_tags = tags_vectorizer.transform(train_tags_arr, train_valid_positions)
val_tags = tags_vectorizer.transform(val_tags_arr, val_valid_positions)
slots_num = len(tags_vectorizer.label_encoder.classes_)


intents_label_encoder = LabelEncoder()
train_intents = intents_label_encoder.fit_transform(train_intents).astype(np.int32)
val_intents = intents_label_encoder.transform(val_intents).astype(np.int32)
intents_num = len(intents_label_encoder.classes_)


model = JointBertModel(slots_num, intents_num, sess, num_bert_fine_tune_layers=10)

model.fit([train_input_ids, train_input_mask, train_segment_ids, train_valid_positions], [train_tags, train_intents],
          validation_data=([val_input_ids, val_input_mask, val_segment_ids, val_valid_positions], [val_tags, val_intents]),
          epochs=epochs, batch_size=batch_size)


### saving
print('Saving ..')
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)
    print('Folder `%s` created' % save_folder_path)
model.save(save_folder_path)
with open(os.path.join(save_folder_path, 'tags_vectorizer.pkl'), 'wb') as handle:
    pickle.dump(tags_vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(save_folder_path, 'intents_label_encoder.pkl'), 'wb') as handle:
    pickle.dump(intents_label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)


tf.compat.v1.reset_default_graph()