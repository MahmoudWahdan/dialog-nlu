# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

from .nlu_components import TransformerNLU, BertNLU, BertCrfNLU
import json
import os


NAME_TO_CLASS = {
    "TransformerNLU": TransformerNLU,
    "BertNLU": BertNLU,
    "BertCrfNLU": BertCrfNLU
}


class AutoNLU:

    def __init__(self):
        pass

    @staticmethod
    def load(path):
        with open(os.path.join(path, 'config.json'), 'r') as json_file:
            config = json.load(json_file)
        class_name = config.get("nlu_class")
        if class_name not in NAME_TO_CLASS:
            raise Exception("{} is not supported in AutoNLU".format(class_name))
        print("Loading {} ...".format(class_name))
        clazz = NAME_TO_CLASS[class_name]
        return clazz.load(path)
