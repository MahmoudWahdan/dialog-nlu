# -*- coding: utf-8 -*-
"""
Implementation of Layer pruning 

Paper: Poor Man’s BERT: Smaller and Faster Transformer Models

@author: mwahdan
"""

NUM_LAYERS_CONFIG_PARAMETER = {
    'BertConfig': 'num_hidden_layers',
    'DistilBertConfig': 'n_layers',
    'RobertaConfig': 'num_hidden_layers'
}


MODEL_ENCODER_NAME_FORMAT = {
    'TFBertModel': 'layer_._{}',
    'TFDistilBertModel': 'layer_._{}',
    'TFRobertaModel': 'layer_._{}'
}


MODEL_Transformer_PART = {
    'TFBertModel': 'bert.encoder',
    'TFDistilBertModel': 'distilbert.transformer',
    'TFRobertaModel': 'roberta.encoder'
}


def modify_num_of_layers(config, k=None, layers_indexes=None, is_alternate=False):
    if (k is None and layers_indexes is None) or (k is not None and layers_indexes is not None):
        raise Exception("One and only one of `k` and `layers_indexes` should have a value")
    key = config.__class__.__name__
    if key not in NUM_LAYERS_CONFIG_PARAMETER:
        raise Exception('`%s` is not supported in layer_pruning' % key)
    value = NUM_LAYERS_CONFIG_PARAMETER[key]
    original_num_layers = getattr(config, value)
    num_layers = original_num_layers
    
    # handle layers_indexes
    if layers_indexes is not None:
        # make sure indexs are valid
        for i in layers_indexes:
            if i >= num_layers:
                raise Exception("index `%d` can't be greater than or equal number of layers in the model `%d`" % (i, num_layers))
        # compute k
        k = len(set(layers_indexes))
    
    # check for is_alternate
    if is_alternate and k >= int(num_layers / 2):
        raise Exception("k = `%d` in `alternate` strategy can't be greater than or equal the half of the number of layers in the model `%d`" % (k, num_layers))
    # check for k
    if k >= num_layers:
        raise Exception("k = `%d` can't be greater than or equal the number of layers in the model `%d`" % (k, num_layers))
    
    # set new num_layers
#    setattr(config, 'hidden_dim', num_layers - k)
    setattr(config, value, num_layers - k)
    return config, original_num_layers


def rename_layers_in_strategy(model, strategy, original_num_layers, k, layers_indexes, is_odd):
    if strategy == 'top':
        # No need to rename, because they are already in order.
        return model
    elif strategy == 'buttom':
        indexes = list(range(k, original_num_layers))
    elif strategy == 'symmetric':
        lst = list(range(original_num_layers))
        idx = int((original_num_layers - k) / 2)
        indexes = lst[:idx] + lst[-idx:]
    elif strategy == 'custom':
        s = set(layers_indexes)
        indexes = [i for i in range(original_num_layers) if i not in s]
    elif strategy == 'alternate':
        lst = []
        pruned = []
        for i in reversed(range(original_num_layers)):
            if (is_odd and i % 2 == 0 and len(pruned) < k) or (not is_odd and i % 2 == 1 and len(pruned) < k):
                pruned.append(i)
                continue
            lst.append(i)
        indexes = sorted(lst)
    else:
        raise Exception('`%s` is not a supported layer pruning strategy' % strategy)
    model = rename_layers(model, indexes)
    return model


def rename_layers(model, order=None):
    key = model.__class__.__name__
    if key not in MODEL_Transformer_PART or key not in MODEL_ENCODER_NAME_FORMAT:
        raise Exception('`%s` is not supported in layer_pruning' % key)
    _format = MODEL_ENCODER_NAME_FORMAT[key]
    value = MODEL_Transformer_PART[key]
    obj = model
    children = value.split('.')
    for child in children:
        obj = getattr(obj, child)
        
    # temp renaming
    for i in range(len(obj.layer)):
        obj.layer[i]._name = "layer_temp_._{}".format(i)
    
    # handle empty order
    if order is None or len(order) == 0:
        for i in range(len(obj.layer)):
            obj.layer[i]._name = _format.format(i)
    else: # handle specific order
        for i in range(len(obj.layer)):
            obj.layer[i]._name = _format.format(order[i])
        
    return model