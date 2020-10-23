# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

from dialognlu import TransformerNLU
from dialognlu.readers.goo_format_reader import Reader

k = 4

train_path = "../data/snips/train"
val_path = "../data/snips/valid"

train_dataset = Reader.read(train_path)
val_dataset = Reader.read(val_path)


# pretrained_model_name_or_path = "xlnet-base-cased"
# save_path = "../saved_models/joint_trans_xlnet_model_{}_layers_pruning".format(k)

pretrained_model_name_or_path = "roberta-base"
save_path = "../saved_models/joint_trans_roberta_model_{}_layers_pruning".format(k)

# pretrained_model_name_or_path = "bert-base-uncased"
# save_path = "../saved_models/joint_trans_bert_model_{}_layers_pruning".format(k)

# pretrained_model_name_or_path = "distilbert-base-uncased"
# save_path = "../saved_models/joint_distilbert_model_{}_layers_pruning".format(k)

epochs = 1 #3
batch_size = 64


#     {
#        "layer_pruning": {
#            "strategy": "top",
#            "k": 2#6#2
#        }
#        "layer_pruning": {
#            "strategy": "custom",
#            "layers_indexes": [2, 3]
#        }
#     }


config = {
    "cache_dir": "/media/mwahdan/Data/transformers",
    "pretrained_model_name_or_path": pretrained_model_name_or_path,
    "from_pt": False,
    "num_bert_fine_tune_layers": 10,
    "intent_loss_weight": 1.0,#0.2,
    "slots_loss_weight": 3.0,#2.0,

    "layer_pruning": {
        "strategy": "top",
        "k": k
    }
}


nlu = TransformerNLU.from_config(config)
nlu.train(train_dataset, val_dataset, epochs, batch_size)

print("Saving ...")
nlu.save(save_path)
print("Done")