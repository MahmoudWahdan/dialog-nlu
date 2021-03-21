# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

from dialognlu import TransformerNLU
from dialognlu.utils.tf_utils import convert_to_tflite_model

# model_path = "../saved_models/joint_distilbert_model"
# model_path = "../saved_models/joint_trans_bert_model"
model_path = "../saved_models/joint_trans_xlnet_model"

print("Loading model ...")
nlu = TransformerNLU.load(model_path)

save_file_path = model_path + "/model.tflite"
convert_to_tflite_model(nlu.model.model, save_file_path, conversion_mode="normal")
print("Done")