# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

from tensorflow.keras.callbacks import Callback
from seqeval.metrics import f1_score, classification_report
import numpy as np


"""
Original Implementation: https://github.com/chakki-works/seqeval/blob/master/seqeval/callbacks.py
adapted implementation by @mwahdan
"""
class F1Metrics(Callback):

    def __init__(self, id2label, pad_value=0, validation_data=None, digits=None):
        """
        Args:
            id2label (dict): id to label mapping.
            (e.g. {1: 'B-LOC', 2: 'I-LOC'})
            pad_value (int): padding value.
            digits (int or None): number of digits in printed classification report
              (use None to print only F1 score without a report).
        """
        super(F1Metrics, self).__init__()
        self.id2label = id2label
        self.pad_value = pad_value
        self.validation_data = validation_data
        self.digits = digits

    def convert_idx_to_name(self, y, array_indexes):
        """Convert label index to name.

        Args:
            y (np.ndarray): label index 2d array.
            array_indexes (list): list of valid index arrays for each row.

        Returns:
            y: label name list.
        """
        y = [[self.id2label[idx] for idx in row[row_indexes]] for
             row, row_indexes in zip(y, array_indexes)]
        return y

    def predict(self, X, y):
        """Predict sequences.

        Args:
            X (np.ndarray): input data.
            y (np.ndarray): tags.

        Returns:
            y_true: true sequences.
            y_pred: predicted sequences.
        """
        y_pred, _ = self.model.predict_on_batch(X)

        y_true = y
        # reduce dimension.
        y_pred = np.argmax(y_pred, -1)

        non_pad_indexes = [np.nonzero(y_true_row != self.pad_value)[0] for y_true_row in y_true]

        y_true = self.convert_idx_to_name(y_true, non_pad_indexes)
        y_pred = self.convert_idx_to_name(y_pred, non_pad_indexes)

        return y_true, y_pred

    def score(self, y_true, y_pred):
        """Calculate f1 score.

        Args:
            y_true (list): true sequences.
            y_pred (list): predicted sequences.

        Returns:
            score: f1 score.
        """
        score = f1_score(y_true, y_pred)
        print(' - f1: {:04.2f}'.format(score * 100))
        if self.digits:
            print(classification_report(y_true, y_pred, digits=self.digits))
        return score

    def on_epoch_end(self, epoch, logs={}):
        # check for validation data
        if self.validation_data:
            X = self.validation_data[0]
            y = self.validation_data[1][0]
            y_true, y_pred = self.predict(X, y)
            score = self.score(y_true, y_pred)
            logs['f1'] = score