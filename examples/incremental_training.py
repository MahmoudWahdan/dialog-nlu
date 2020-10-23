# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

from dialognlu import AutoNLU
from dialognlu.readers.goo_format_reader import Reader


model_path = "../saved_models/joint_distilbert_model"
# model_path = "../saved_models/joint_bert_model"
epochs = 1
batch_size = 64


print("Loading model ...")
nlu = AutoNLU.load(model_path)

print("Loading dataset ...")
train_path = "../data/snips/train"
val_path = "../data/snips/valid"
test_path = "../data/snips/test"

train_dataset = Reader.read(train_path)
val_dataset = Reader.read(val_path)
test_dataset = Reader.read(test_path)

print("Evaluating model ...")
token_f1_score, tag_f1_score, report, acc = nlu.evaluate(test_dataset)

print('Slot Classification Report:', report)
print('Slot token f1_score = %f' % token_f1_score)
print('Slot tag f1_score = %f' % tag_f1_score)
print('Intent accuracy = %f' % acc)

print("Continue training ...")
nlu.train(train_dataset, val_dataset, epochs, batch_size)

print("Saving ...")
nlu.save(model_path)
print("Done")

print("Evaluating model again! ...")
token_f1_score, tag_f1_score, report, acc = nlu.evaluate(test_dataset)

print('Slot Classification Report:', report)
print('Slot token f1_score = %f' % token_f1_score)
print('Slot tag f1_score = %f' % tag_f1_score)
print('Intent accuracy = %f' % acc)