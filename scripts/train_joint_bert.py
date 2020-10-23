# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

from dialognlu import BertNLU
from dialognlu.readers.goo_format_reader import Reader


import argparse
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf


# read command-line parameters
parser = argparse.ArgumentParser('Training the Joint BERT NLU model')
parser.add_argument('--train', '-t', help = 'Path to training data in Goo et al format', type = str, required = True)
parser.add_argument('--val', '-v', help = 'Path to validation data in Goo et al format', type = str, required = True)
parser.add_argument('--save', '-s', help = 'Folder path to save the trained model', type = str, required = True)
parser.add_argument('--epochs', '-e', help = 'Number of epochs', type = int, default = 5, required = False)
parser.add_argument('--batch', '-bs', help = 'Batch size', type = int, default = 64, required = False)
parser.add_argument('--type', '-tp', help = 'bert   or    albert', type = str, default = 'bert', required = False)
parser.add_argument('--model', '-m', help = 'Path to joint BERT / ALBERT NLU model for incremental training', type = str, required = False)


VALID_TYPES = ['bert', 'albert']

args = parser.parse_args()
train_data_folder_path = args.train
val_data_folder_path = args.val
save_folder_path = args.save
epochs = args.epochs
batch_size = args.batch
type_ = args.type
start_model_folder_path = args.model


if type_ not in {'bert', 'albert'}:
    raise ValueError('type must be one of these values: %s' % str(VALID_TYPES))


print('Reading data ...')
train_dataset = Reader.read(train_data_folder_path)
val_dataset = Reader.read(val_data_folder_path)


if start_model_folder_path is None:
    config = {
        "model_type": type_
    }
    nlu = BertNLU.from_config(config)
else:
    nlu = BertNLU.load(start_model_folder_path)

print("Training ...")
nlu.train(train_dataset, val_dataset, epochs, batch_size)

print("Saving ...")
nlu.save(save_folder_path)
print("Done")

tf.compat.v1.reset_default_graph()