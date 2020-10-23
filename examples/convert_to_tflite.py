# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

from dialognlu import TransformerNLU
from dialognlu.utils.tf_utils import convert_to_tflite_model

model_path = "../saved_models/joint_distilbert_model"

print("Loading model ...")
nlu = TransformerNLU.load(model_path)

save_file_path = "../saved_models/joint_distilbert_model/model.tflite"
convert_to_tflite_model(nlu.model.model, save_file_path, conversion_mode="fp16_quantization")