# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

from dialognlu import TransformerNLU
from dialognlu.readers.goo_format_reader import Reader
from dialognlu.utils.io_utils import str2bool
import tensorflow as tf
import argparse


# read command-line parameters
parser = argparse.ArgumentParser('Training the Joint Transformer NLU model')
parser.add_argument('--train', '-t', help = 'Path to training data in Goo et al format', type = str, required = True)
parser.add_argument('--val', '-v', help = 'Path to validation data in Goo et al format', type = str, required = True)
parser.add_argument('--save', '-s', help = 'Folder path to save the trained model', type = str, required = True)
parser.add_argument('--epochs', '-e', help = 'Number of epochs', type = int, default = 5, required = False)
parser.add_argument('--batch', '-bs', help = 'Batch size', type = int, default = 64, required = False)
parser.add_argument('--model', '-m', help = 'Path to joint trans NLU model for incremental training', type = str, required = False)
parser.add_argument('--trans', '-tr', help = 'Pretrained transformer model name or path. Is optional. Either --model OR --trans should be provided'
                    , type = str, required = False)
parser.add_argument('--from_pt', '-pt', help = 'Whether the --trans (if provided) is from pytorch or not', 
                    type=str2bool, nargs='?', const=True, default=False, required=False)
parser.add_argument('--cache_dir', '-c', help = 'The cache_dir for transformers library. Is optional', type = str, required = False, default = None)



args = parser.parse_args()
train_data_folder_path = args.train
val_data_folder_path = args.val
save_folder_path = args.save
epochs = args.epochs
batch_size = args.batch
start_model_folder_path = args.model
pretrained_model_name_or_path = args.trans
from_pt = args.from_pt
cache_dir = args.cache_dir
if start_model_folder_path is None and pretrained_model_name_or_path is None:
    raise argparse.ArgumentTypeError("Either --model OR --trans should be provided")

print('Reading data ...')
train_dataset = Reader.read(train_data_folder_path)
val_dataset = Reader.read(val_data_folder_path)


if start_model_folder_path is None:
    config = {
        "cache_dir": cache_dir,
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "from_pt": from_pt,
        "num_bert_fine_tune_layers": 10,
        "intent_loss_weight": 1.0,
        "slots_loss_weight": 3.0,
    }
    nlu = TransformerNLU.from_config(config)
else:
    nlu = TransformerNLU.load(start_model_folder_path)

print("Training ...")
nlu.train(train_dataset, val_dataset, epochs, batch_size)

print("Saving ...")
nlu.save(save_folder_path)
print("Done")

tf.compat.v1.reset_default_graph()