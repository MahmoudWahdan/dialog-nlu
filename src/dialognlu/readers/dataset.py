# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

class NluDataset:

    def __init__(self, text, tags, intents):
        self.text = text
        self.tags = tags
        self.intents = intents