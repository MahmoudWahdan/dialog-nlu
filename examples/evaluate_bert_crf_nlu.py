# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

from dialognlu import BertCrfNLU
from dialognlu.readers.goo_format_reader import Reader


model_path = "../saved_models/joint_bert_crf_model"

print("Loading model ...")
nlu = BertCrfNLU.load(model_path)

print("Loading dataset ...")
test_path = "../data/snips/test"
test_dataset = Reader.read(test_path)

print("Evaluating model ...")
token_f1_score, tag_f1_score, report, acc = nlu.evaluate(test_dataset)

print('Slot Classification Report:', report)
print('Slot token f1_score = %f' % token_f1_score)
print('Slot tag f1_score = %f' % tag_f1_score)
print('Intent accuracy = %f' % acc)