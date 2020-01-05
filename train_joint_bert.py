# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

from readers.goo_format_reader import Reader
from vectorizers.bert_vectorizer import BERTVectorizer
from vectorizers.tags_vectorizer import TagsVectorizer
from models.joint_bert import JointBertModel

import argparse
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import pickle
import tensorflow as tf


# read command-line parameters
parser = argparse.ArgumentParser('Training the Joint BERT NLU model')
parser.add_argument('--train', '-t', help = 'Path to training data in Goo et al format', type = str, required = True)
parser.add_argument('--val', '-v', help = 'Path to validation data in Goo et al format', type = str, required = True)
parser.add_argument('--save', '-s', help = 'Folder path to save the trained model', type = str, required = True)
parser.add_argument('--epochs', '-e', help = 'Number of epochs', type = int, default = 5, required = False)
parser.add_argument('--batch', '-bs', help = 'Batch size', type = int, default = 64, required = False)
parser.add_argument('--type', '-tp', help = 'bert   or    albert', type = str, default = 'bert', required = False)


VALID_TYPES = ['bert', 'albert']

args = parser.parse_args()
train_data_folder_path = args.train
val_data_folder_path = args.val
save_folder_path = args.save
epochs = args.epochs
batch_size = args.batch
type_ = args.type


tf.compat.v1.random.set_random_seed(7)


sess = tf.compat.v1.Session()

if type_ == 'bert':
    bert_model_hub_path = 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1'
    is_bert = True
elif type_ == 'albert':
    bert_model_hub_path = 'https://tfhub.dev/google/albert_base/1'
    is_bert = False
else:
    raise ValueError('type must be one of these values: %s' % str(VALID_TYPES))

print('read data ...')
train_text_arr, train_tags_arr, train_intents = Reader.read(train_data_folder_path)
val_text_arr, val_tags_arr, val_intents = Reader.read(val_data_folder_path)

print('vectorize data ...')
bert_vectorizer = BERTVectorizer(sess, is_bert, bert_model_hub_path)
train_input_ids, train_input_mask, train_segment_ids, train_valid_positions, train_sequence_lengths = bert_vectorizer.transform(train_text_arr)
val_input_ids, val_input_mask, val_segment_ids, val_valid_positions, val_sequence_lengths = bert_vectorizer.transform(val_text_arr)


print('vectorize tags ...')
tags_vectorizer = TagsVectorizer()
tags_vectorizer.fit(train_tags_arr)
train_tags = tags_vectorizer.transform(train_tags_arr, train_valid_positions)
val_tags = tags_vectorizer.transform(val_tags_arr, val_valid_positions)
slots_num = len(tags_vectorizer.label_encoder.classes_)


print('encode labels ...')
intents_label_encoder = LabelEncoder()
train_intents = intents_label_encoder.fit_transform(train_intents).astype(np.int32)
val_intents = intents_label_encoder.transform(val_intents).astype(np.int32)
intents_num = len(intents_label_encoder.classes_)


model = JointBertModel(slots_num, intents_num, bert_model_hub_path, sess, 
                       num_bert_fine_tune_layers=10, is_bert=is_bert)

print('training model ...')
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