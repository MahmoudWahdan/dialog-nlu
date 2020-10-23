# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

from dialognlu import AutoNLU
from dialognlu.readers.goo_format_reader import Reader
import argparse

# read command-line parameters
parser = argparse.ArgumentParser('Evaluating the Joint Transformer NLU model')
parser.add_argument('--model', '-m', help = 'Path to joint BERT / ALBERT NLU model', type = str, required = True)
parser.add_argument('--data', '-d', help = 'Path to data in Goo et al format', type = str, required = True)
parser.add_argument('--batch', '-bs', help = 'Batch size', type = int, default = 128, required = False)

args = parser.parse_args()
model_path = args.model
data_folder_path = args.data
batch_size = args.batch


print("Loading model ...")
nlu = AutoNLU.load(model_path)

print("Loading dataset ...")
test_dataset = Reader.read(data_folder_path)

print("Evaluating model ...")
token_f1_score, tag_f1_score, report, acc = nlu.evaluate(test_dataset)

print('Slot Classification Report:', report)
print('Slot token f1_score = %f' % token_f1_score)
print('Slot tag f1_score = %f' % tag_f1_score)
print('Intent accuracy = %f' % acc)