# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

from .models.trans_auto_model import create_joint_trans_model, load_joint_trans_model
from .models.joint_bert import JointBertModel
from .models.joint_bert_crf import JointBertCRFModel
from .vectorizers.trans_vectorizer import TransVectorizer
from .vectorizers.bert_vectorizer import BERTVectorizer
from .vectorizers.tags_vectorizer import TagsVectorizer
from .readers.dataset import NluDataset
from .utils.data_utils import flatten, convert_to_slots
from .utils.tf_utils import convert_to_tflite_model
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from seqeval.metrics import classification_report, f1_score
import numpy as np
import pickle
import json
import os



class NLU:

    def __init__(self):
        pass

class JointNLU(NLU):

    def __init__(self):
        super(JointNLU, self).__init__()
        self.model = None
        self.text_vectorizer = None
        self.tags_vectorizer = None
        self.intents_label_encoder = None
        self.config = {}

    def init_text_vectorizer(self):
        raise NotImplementedError

    def init_model(self):
        raise NotImplementedError

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            print('Folder `{}` created'.format(path))
        self.model.save(path)
        # with open(os.path.join(path, 'tags_vectorizer.pkl'), 'wb') as handle:
        #     pickle.dump(self.tags_vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.tags_vectorizer.save(os.path.join(path, 'tags_vectorizer.pkl'))
        with open(os.path.join(path, 'intents_label_encoder.pkl'), 'wb') as handle:
            pickle.dump(self.intents_label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path, 'config.json'), 'w') as json_file:
            json.dump(self.config, json_file)

    @staticmethod
    def load_pickles(path, claz):
        if not os.path.exists(path):
            raise Exception('Folder `{}` not exist'.format(path))
        new_instance = claz()
        # loading tags vectorizer
        new_instance.tags_vectorizer = TagsVectorizer.load(os.path.join(path, 'tags_vectorizer.pkl'))
        # with open(os.path.join(path, 'tags_vectorizer.pkl'), 'rb') as handle:
        #     new_instance.tags_vectorizer = pickle.load(handle)
        # loading intents encoder
        with open(os.path.join(path, 'intents_label_encoder.pkl'), 'rb') as handle:
            new_instance.intents_label_encoder = pickle.load(handle)
        # loading config
        with open(os.path.join(path, 'config.json'), 'r') as json_file:
            new_instance.config = json.load(json_file)
        new_instance.init_text_vectorizer()
        return new_instance

    def train(self, train_dataset: NluDataset, val_dataset: NluDataset=None, epochs=3, batch_size=64):
        if self.text_vectorizer is None:
            self.init_text_vectorizer()
        print('Vectorizing training text ...')
        # train_input_ids, train_input_mask, train_segment_ids, train_valid_positions, train_sequence_lengths = self.text_vectorizer.transform(train_dataset.text)
        train_data = self.text_vectorizer.transform(train_dataset.text)
        # get max if not exist
        max_length = self.config.get("max_length", None)
        if max_length is None:
            max_length = self.text_vectorizer.max_length
            self.config["max_length"] = max_length
        train_valid_positions = train_data["valid_positions"]
        if self.tags_vectorizer is None:
            print('Fitting tags encoder ...')
            self.tags_vectorizer = TagsVectorizer()
            self.tags_vectorizer.fit(train_dataset.tags)
            slots_num = len(self.tags_vectorizer.label_encoder.classes_)
            self.config["slots_num"] = slots_num
        if self.intents_label_encoder is None:
            print('Fitting intent encoder ...')
            self.intents_label_encoder = LabelEncoder()
            self.intents_label_encoder.fit(train_dataset.intents)
            intents_num = len(self.intents_label_encoder.classes_)
            self.config["intents_num"] = intents_num
        print('Encoding training tags ...')
        train_tags = self.tags_vectorizer.transform(train_dataset.tags, train_valid_positions)
        print('Encoding training intents ...')
        train_intents = self.intents_label_encoder.transform(train_dataset.intents).astype(np.int32)
        
        id2label = {i:v for i, v in enumerate(self.tags_vectorizer.label_encoder.classes_)}

        if val_dataset is not None:
            print('Vectorizing validation text ...')
            # val_input_ids, val_input_mask, val_segment_ids, val_valid_positions, val_sequence_lengths = self.text_vectorizer.transform(val_dataset.text)
            val_data = self.text_vectorizer.transform(val_dataset.text)
            val_valid_positions = val_data["valid_positions"]
            print('Encoding validation tags ...')
            val_tags = self.tags_vectorizer.transform(val_dataset.tags, val_valid_positions)
            print('Encoding validation intents ...')
            val_intents = self.intents_label_encoder.transform(val_dataset.intents).astype(np.int32)

        if self.model is None:
            self.init_model()

        print('Training model ...')
        self.model.fit(train_data, [train_tags, train_intents], validation_data=(val_data, [val_tags, val_intents]),
                epochs=epochs, batch_size=batch_size, id2label=id2label)

    def predict(self, utterance: str):
        tokens = utterance.split()
        x = self.text_vectorizer.transform([utterance])
        predicted_tags, predicted_intents = self.model.predict_slots_intent(x, self.tags_vectorizer, 
                    self.intents_label_encoder, remove_start_end=True, include_intent_prob=True)
        slots = convert_to_slots(predicted_tags[0])
        slots = [{"slot": slot, "start": start, "end": end, "value": ' '.join(tokens[start:end + 1])} for slot, start, end in slots]
        response = {
            "intent": {
                    "name": predicted_intents[0][0].strip(),
                    "confidence": predicted_intents[0][1]
                    },
            "slots": slots
        }
        return response

    def evaluate(self, dataset: NluDataset):
        print('Vectorizing validation text ...')
        X = self.text_vectorizer.transform(dataset.text)
        tags, intents = dataset.tags, dataset.intents

        predicted_tags, predicted_intents = self.model.predict_slots_intent(X, 
                self.tags_vectorizer, self.intents_label_encoder, remove_start_end=True)
        gold_tags = [x.split() for x in tags]
                
        #print(metrics.classification_report(flatten(gold_tags), flatten(predicted_tags), digits=3))
        token_f1_score = metrics.f1_score(flatten(gold_tags), flatten(predicted_tags), average='micro')
        acc = metrics.accuracy_score(intents, predicted_intents)
            
        report = classification_report(gold_tags, predicted_tags, digits=4)
        tag_f1_score = f1_score(gold_tags, predicted_tags, average='micro')
        
        return token_f1_score, tag_f1_score, report, acc


class TransformerNLU(JointNLU):

    def __init__(self):
        super(TransformerNLU, self).__init__()

    @staticmethod
    def from_config(config: dict):
        new_instance = TransformerNLU()
        config["nlu_class"] = "TransformerNLU"
        new_instance.config = config
        new_instance.init_text_vectorizer()
        return new_instance

    def init_text_vectorizer(self):
        pretrained_model_name_or_path = self.config["pretrained_model_name_or_path"]
        cache_dir = self.config["cache_dir"]
        max_length = self.config.get("max_length", None) # get max_length or None. If None, it will be computed internally
        self.text_vectorizer = TransVectorizer(pretrained_model_name_or_path, max_length, cache_dir)

    def init_model(self):
        self.model = create_joint_trans_model(self.config)  

    @staticmethod
    def load(path, quantized=False, num_process=4):
        new_instance = JointNLU.load_pickles(path, TransformerNLU)
        new_instance.model = load_joint_trans_model(path, quantized, num_process)
        return new_instance

    def save(self, path, save_tflite=False, conversion_mode="hybrid_quantization"):
        super(TransformerNLU, self).save(path)
        if save_tflite:
            convert_to_tflite_model(self.model.model, os.path.join(path, "model.tflite"), conversion_mode=conversion_mode)


class BertNLU(JointNLU):

    def __init__(self):
        super(BertNLU, self).__init__()
        self.VALID_TYPES = {'bert', 'albert'}

    @staticmethod
    def from_config(config: dict):
        new_instance = BertNLU()
        config["nlu_class"] = "BertNLU"
        new_instance.config = config
        new_instance.init_text_vectorizer()
        return new_instance

    def init_text_vectorizer(self):
        model_type = self.config.get("model_type", "bert")
        if model_type == 'bert':
            bert_model_hub_path = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
            is_bert = True
        elif model_type == 'albert':
            bert_model_hub_path = "https://tfhub.dev/tensorflow/albert_en_base/1"
            is_bert = False
        else:
            raise ValueError('type must be one of these values: %s' % str(self.VALID_TYPES))
        self.config["is_bert"] = is_bert
        self.config["bert_model_hub_path"] = bert_model_hub_path
        self.text_vectorizer = BERTVectorizer(is_bert, bert_model_hub_path)

    def init_model(self):
        bert_model_hub_path = self.config["bert_model_hub_path"]
        is_bert = self.config["is_bert"]
        slots_num = self.config["slots_num"]
        intents_num = self.config["intents_num"]
        self.model = JointBertModel(slots_num, intents_num, bert_model_hub_path, 
                        num_bert_fine_tune_layers=10, is_bert=is_bert)

    @staticmethod
    def load(path):
        new_instance = JointNLU.load_pickles(path, BertNLU)
        new_instance.model = JointBertModel.load(path)
        return new_instance



class BertCrfNLU(JointNLU):

    def __init__(self):
        super(BertCrfNLU, self).__init__()
        self.VALID_TYPES = {'bert', 'albert'}

    @staticmethod
    def from_config(config: dict):
        new_instance = BertCrfNLU()
        config["nlu_class"] = "BertCrfNLU"
        new_instance.config = config
        new_instance.init_text_vectorizer()
        return new_instance

    def init_text_vectorizer(self):
        model_type = self.config.get("model_type", "bert")
        if model_type == 'bert':
            bert_model_hub_path = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
            is_bert = True
        elif model_type == 'albert':
            bert_model_hub_path = "https://tfhub.dev/tensorflow/albert_en_base/1"
            is_bert = False
        else:
            raise ValueError('type must be one of these values: %s' % str(self.VALID_TYPES))
        self.config["is_bert"] = is_bert
        self.config["bert_model_hub_path"] = bert_model_hub_path
        self.text_vectorizer = BERTVectorizer(is_bert, bert_model_hub_path)

    def init_model(self):
        bert_model_hub_path = self.config["bert_model_hub_path"]
        is_bert = self.config["is_bert"]
        slots_num = self.config["slots_num"]
        intents_num = self.config["intents_num"]
        self.model = JointBertCRFModel(slots_num, intents_num, bert_model_hub_path, 
                        num_bert_fine_tune_layers=10, is_bert=is_bert)

    @staticmethod
    def load(path):
        new_instance = JointNLU.load_pickles(path, BertNLU)
        new_instance.model = JointBertModel.load(path)
        return new_instance