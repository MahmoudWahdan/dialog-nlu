# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

from dialognlu import TransformerNLU
from dialognlu.readers.goo_format_reader import Reader


train_path = "../data/snips/train"
val_path = "../data/snips/valid"

train_dataset = Reader.read(train_path)
val_dataset = Reader.read(val_path)


pretrained_model_name_or_path = "xlnet-base-cased"
save_path = "../saved_models/joint_trans_xlnet_model"

# pretrained_model_name_or_path = "roberta-base"
# save_path = "../saved_models/joint_trans_roberta_model"

# pretrained_model_name_or_path = "albert-base-v1"
# save_path = "../saved_models/joint_trans_albert_model"

# pretrained_model_name_or_path = "bert-base-uncased"
# save_path = "../saved_models/joint_trans_bert_model"

# pretrained_model_name_or_path = "distilbert-base-uncased"
# save_path = "../saved_models/joint_distilbert_model"

epochs = 5
batch_size = 32 # 64


config = {
    "cache_dir": "/media/mwahdan/Data/transformers",
    "pretrained_model_name_or_path": pretrained_model_name_or_path,
    "from_pt": False,
    "num_bert_fine_tune_layers": 10,
    "intent_loss_weight": 1.0,#0.2,
    "slots_loss_weight": 3.0,#2.0,

    "max_length": 64, # You can set max_length (recommended) or leave it and it will be computed automatically based on longest training example
}


nlu = TransformerNLU.from_config(config)
nlu.train(train_dataset, val_dataset, epochs, batch_size)

print("Saving ...")
nlu.save(save_path, save_tflite=True, conversion_mode="normal")
print("Done")