# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

from itertools import chain
import argparse


def str2bool(v):
    """
    Source: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/43357954#43357954
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')