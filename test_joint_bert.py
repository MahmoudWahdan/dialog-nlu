# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

from readers.goo_format_reader import Reader
from vectorizers.bert_vectorizer import BERTVectorizer
from models.joint_bert import JointBertModel
from utils import flatten

import os
import pickle
import tensorflow as tf
from sklearn import metrics


sess = tf.compat.v1.Session()

batch_size = 128

bert_model_hub_path = 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1'
bert_vectorizer = BERTVectorizer(sess, bert_model_hub_path)

load_folder_path = 'saved_models/joint_bert_model'
# loading models
print('Loading models ...')
if not os.path.exists(load_folder_path):
    print('Folder `%s` not exist' % load_folder_path)

with open(os.path.join(load_folder_path, 'tags_vectorizer.pkl'), 'rb') as handle:
    tags_vectorizer = pickle.load(handle)
    slots_num = len(tags_vectorizer.label_encoder.classes_)
with open(os.path.join(load_folder_path, 'intents_label_encoder.pkl'), 'rb') as handle:
    intents_label_encoder = pickle.load(handle)
    intents_num = len(intents_label_encoder.classes_)
    
model = JointBertModel.load(load_folder_path, sess)


train_text_arr, train_tags_arr, train_intents = Reader.read('data/atis/train')
val_text_arr, val_tags_arr, val_intents = Reader.read('data/atis/valid')
test_text_arr, test_tags_arr, test_intents = Reader.read('data/atis/test')


train_input_ids, train_input_mask, train_segment_ids, train_valid_positions = bert_vectorizer.transform(train_text_arr)
val_input_ids, val_input_mask, val_segment_ids, val_valid_positions = bert_vectorizer.transform(val_text_arr)
test_input_ids, test_input_mask, test_segment_ids, test_valid_positions = bert_vectorizer.transform(test_text_arr)


def get_results(input_ids, input_mask, segment_ids, valid_positions, tags_arr, 
                intents, tags_vectorizer, intents_label_encoder):
    predicted_tags, predicted_intents = model.predict_slots_intent(
            [input_ids, input_mask, segment_ids, valid_positions], 
            tags_vectorizer, intents_label_encoder, remove_start_end=True)
    gold_tags = [x.split() for x in tags_arr]
    #print(metrics.classification_report(flatten(gold_tags), flatten(predicted_tags), digits=3))
    f1_score = metrics.f1_score(flatten(gold_tags), flatten(predicted_tags), average='micro')
    acc = metrics.accuracy_score(intents, predicted_intents)
    return f1_score, acc


print('==== Training Dataset Evaluation ====')
f1_score, acc = get_results(train_input_ids, train_input_mask, train_segment_ids, train_valid_positions,
                            train_tags_arr, train_intents, tags_vectorizer, intents_label_encoder)
print('Slot f1_score = %f' % f1_score)
print('Intent accuracy = %f' % acc)

print('==== Validation Dataset Evaluation ====')
f1_score, acc = get_results(val_input_ids, val_input_mask, val_segment_ids, val_valid_positions,
                            val_tags_arr, val_intents, tags_vectorizer, intents_label_encoder)
print('Slot f1_score = %f' % f1_score)
print('Intent accuracy = %f' % acc)

print('==== Testing Dataset Evaluation ====')
f1_score, acc = get_results(test_input_ids, test_input_mask, test_segment_ids, test_valid_positions,
                            test_tags_arr, test_intents, tags_vectorizer, intents_label_encoder)
print('Slot f1_score = %f' % f1_score)
print('Intent accuracy = %f' % acc)


tf.compat.v1.reset_default_graph()