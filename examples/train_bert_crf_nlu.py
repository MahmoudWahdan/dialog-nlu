# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

from dialognlu import BertCrfNLU
from dialognlu.readers.goo_format_reader import Reader


train_path = "../data/snips/train"
val_path = "../data/snips/valid"

train_dataset = Reader.read(train_path)
val_dataset = Reader.read(val_path)

save_path = "../saved_models/joint_bert_crf_model"
epochs = 1 #3
batch_size = 32#64

config = {
    "model_type": "bert"
}

nlu = BertCrfNLU.from_config(config)
nlu.train(train_dataset, val_dataset, epochs, batch_size)

print("Saving ...")
nlu.save(save_path)
print("Done")