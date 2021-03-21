# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

# diasable the GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from dialognlu import TransformerNLU
from dialognlu.readers.goo_format_reader import Reader
import time


num_process = 2


# model_path = "../saved_models/joint_distilbert_model"
# model_path = "../saved_models/joint_trans_bert_model"
# model_path = "../saved_models/joint_trans_albert_model"
# model_path = "../saved_models/joint_trans_roberta_model"
model_path = "../saved_models/joint_trans_xlnet_model"


print("Loading model ...")
nlu = TransformerNLU.load(model_path, quantized=True, num_process=num_process)

print("Loading dataset ...")
test_path = "../data/snips/test"
test_dataset = Reader.read(test_path)

print("Evaluating model ...")
t1 = time.time()
token_f1_score, tag_f1_score, report, acc = nlu.evaluate(test_dataset)
t2 = time.time()

print('Slot Classification Report:', report)
print('Slot token f1_score = %f' % token_f1_score)
print('Slot tag f1_score = %f' % tag_f1_score)
print('Intent accuracy = %f' % acc)

print("Using %d processes took %f seconds" % (num_process, t2 - t1))