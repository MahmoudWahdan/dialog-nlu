# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 15:12:18 2020

@author: mwahdan
"""

from flask import Flask, jsonify, request
from models.trans_auto_model import load_joint_trans_model
from vectorizers.trans_vectorizer import TransVectorizer
from vectorizers.bert_vectorizer import BERTVectorizer
from models.joint_bert import JointBertModel
from utils import convert_to_slots
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
import pickle
import argparse
import os


# Create app
app = Flask(__name__)

def initialize(type_, load_folder_path):

    global intents_label_encoder
    global model
    global bert_vectorizer
    if type_ in {'bert', 'albert'}:
        if type_ == 'bert':
            bert_model_hub_path = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
            is_bert = True
        else:
            bert_model_hub_path = "https://tfhub.dev/tensorflow/albert_en_base/1"
            is_bert = False
        
        bert_vectorizer = BERTVectorizer(is_bert, bert_model_hub_path)
        # loading models
        print('Loading models ...')
        if not os.path.exists(load_folder_path):
            print('Folder `%s` not exist' % load_folder_path)
        
        global slots_num
        global tags_vectorizer
        with open(os.path.join(load_folder_path, 'tags_vectorizer.pkl'), 'rb') as handle:
            tags_vectorizer = pickle.load(handle)
            slots_num = len(tags_vectorizer.label_encoder.classes_)
        global intents_num
        with open(os.path.join(load_folder_path, 'intents_label_encoder.pkl'), 'rb') as handle:
            intents_label_encoder = pickle.load(handle)
            intents_num = len(intents_label_encoder.classes_)
        
        model = JointBertModel.load(load_folder_path)
    elif type_ == 'trans':
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
            
        # loading joint trans model
        model = load_joint_trans_model(load_folder_path)

        # loading trans vectorizer
        pretrained_model_name_or_path = model.model_params['pretrained_model_name_or_path']
        cache_dir = model.model_params['cache_dir']
        
        bert_vectorizer = TransVectorizer(pretrained_model_name_or_path, max_length=None, cache_dir=cache_dir)
    else:
        raise ValueError('type must be one of these values: %s' % str(VALID_TYPES))


    

@app.route('/', methods=['GET', 'POST'])
def hello():
    return 'hello from NLU service'


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    input_json = request.json
    utterance = input_json["utterance"]
    tokens = utterance.split()
    print(utterance)
    input_ids, input_mask, segment_ids, valid_positions, data_sequence_lengths = bert_vectorizer.transform([utterance])
    predicted_tags, predicted_intents = model.predict_slots_intent(
            [input_ids, input_mask, segment_ids, valid_positions], 
            tags_vectorizer, intents_label_encoder, remove_start_end=True,
            include_intent_prob=True)
    
    slots = convert_to_slots(predicted_tags[0])
    slots = [{"slot": slot, "start": start, "end": end, "value": ' '.join(tokens[start:end + 1])} for slot, start, end in slots]
    
    response = {
            "intent": {
                    "name": predicted_intents[0][0].strip(),
                    "confidence": predicted_intents[0][1]
                    },
            "slots": slots
            }

    return jsonify(response)


if __name__ == '__main__':
    VALID_TYPES = ['bert', 'albert', 'trans']
    
    # read command-line parameters
    parser = argparse.ArgumentParser('Running Joint BERT / ALBERT NLU model basic service')
    parser.add_argument('--model', '-m', help = 'Path to joint BERT / ALBERT NLU model', type = str, required = True)
    parser.add_argument('--type', '-tp', help = '[bert or albert] for tensorflow-hub based models, trans for HuggingFace transformers based models', type = str, default = 'bert', required = False)
    
    args = parser.parse_args()
    load_folder_path = args.model
    type_ = args.type
    
    print(('Starting the Server'))
    initialize(type_, load_folder_path)
    # Run app
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)