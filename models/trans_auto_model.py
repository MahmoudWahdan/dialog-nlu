# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 09:19:55 2020

@author: mwahdan
"""

from transformers import TFAutoModel
from models.joint_trans_bert import JointTransBertModel
from models.joint_trans_distilbert import JointTransDistilBertModel
import json
import os


TYPE_2_JOINT_MODEL = {
    'TFBertModel': JointTransBertModel,
    'TFDistilBertModel': JointTransDistilBertModel
}


LOAD_CLASS_NAME_2_MODEL = {
    'JointTransDistilBertModel': JointTransDistilBertModel,
    'JointTransBertModel': JointTransBertModel
}


def get_transformer_model(pretrained_model_name_or_path, cache_dir, from_pt):
    trans_model = TFAutoModel.from_pretrained(pretrained_model_name_or_path, 
                                             cache_dir=cache_dir, from_pt=from_pt)
    trans_type = trans_model.__class__.__name__
    model = None
    if trans_type == 'TFBertModel':
        model = trans_model.bert
    elif trans_type == 'TFDistilBertModel':
        model = trans_model.distilbert
    else:
        raise Exception("%s is not supported yet!" % trans_type)
    return model, trans_type


def create_joint_trans_model(config):
    pretrained_model_name_or_path = config.get('pretrained_model_name_or_path')
    cache_dir = config.get('cache_dir', None)
    from_pt = config.get('from_pt', False)
    model, trans_type = get_transformer_model(pretrained_model_name_or_path, cache_dir, from_pt)
    if trans_type not in TYPE_2_JOINT_MODEL:
        raise Exception("%s is not supported yet!" % trans_type)
    joint_model = TYPE_2_JOINT_MODEL[trans_type](config, model)
    return joint_model


def load_joint_trans_model(load_folder_path):
    with open(os.path.join(load_folder_path, 'params.json'), 'r') as json_file:
        model_params = json.load(json_file)
    clazz = model_params['class']
    if clazz not in LOAD_CLASS_NAME_2_MODEL:
        raise Exception('%s not supported')
    model = LOAD_CLASS_NAME_2_MODEL[clazz].load(load_folder_path)
    return model