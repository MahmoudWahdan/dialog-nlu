# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

# diasable the GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from dialognlu import TransformerNLU


# model_path = "../saved_models/joint_distilbert_model"
# model_path = "../saved_models/joint_trans_bert_model"
# model_path = "../saved_models/joint_trans_albert_model"
model_path = "../saved_models/joint_trans_roberta_model"

print("Loading model ...")
nlu = TransformerNLU.load(model_path, quantized=True, num_process=1)

print("Prediction ...")
utterance = "add sabrina salerno to the grime instrumentals playlist"
print ("utterance: {}".format(utterance))
result = nlu.predict(utterance)
print ("result: {}".format(result))