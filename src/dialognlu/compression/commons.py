# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

import logging
import os
import h5py
from tensorflow.python.keras.saving import hdf5_format
from transformers.configuration_utils import PretrainedConfig
from transformers.configuration_auto import AutoConfig
from transformers.file_utils import TF2_WEIGHTS_NAME, WEIGHTS_NAME, cached_path, hf_bucket_url, is_remote_url
from transformers.modeling_tf_pytorch_utils import load_pytorch_checkpoint_in_tf2_model
from transformers.modeling_tf_auto import TF_MODEL_MAPPING
from .layer_pruning import modify_num_of_layers, rename_layers_in_strategy, rename_layers


logger = logging.getLogger(__name__)


def get_mandatory_parameter(parameter_name, parameters_map, error_message=None):
    if parameter_name not in parameters_map:
        if error_message is None:
            # default error_message
            error_message = "`%s` is mandatory, but it is misssing from parameters" % parameter_name
        raise Exception(error_message)
    return parameters_map[parameter_name]


"""
Original implementation of from_pretrained: https://huggingface.co/transformers/_modules/transformers/modeling_tf_utils.html
The modified implementation by @mwahdan For layer_pruning support
"""
def from_pretrained_detailed(model_class, pretrained_model_name_or_path, *model_args, **kwargs):
    r"""Instantiate a pretrained TF 2.0 model from a pre-trained model configuration.

    The warning ``Weights from XXX not initialized from pretrained model`` means that the weights of XXX do not come pre-trained with the rest of the model.
    It is up to you to train those weights with a downstream fine-tuning task.

    The warning ``Weights from XXX not used in YYY`` means that the layer XXX is not used by YYY, therefore those weights are discarded.

    Parameters:
        pretrained_model_name_or_path: either:
            - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
            - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
            - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
            - a path or url to a `PyTorch state_dict save file` (e.g. `./pt_model/pytorch_model.bin`). In this case, ``from_pt`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the PyTorch checkpoint in a TensorFlow model using the provided conversion scripts and loading the TensorFlow model afterwards.

        model_args: (`optional`) Sequence of positional arguments:
            All remaning positional arguments will be passed to the underlying model's ``__init__`` method

        config: (`optional`) one of:
                - an instance of a class derived from :class:`~transformers.PretrainedConfig`, or
                - a string valid as input to :func:`~transformers.PretrainedConfig.from_pretrained()`

            Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:
                - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

        from_pt: (`optional`) boolean, default False:
            Load the model weights from a PyTorch state_dict save file (see docstring of pretrained_model_name_or_path argument).

        cache_dir: (`optional`) string:
            Path to a directory in which a downloaded pre-trained model
            configuration should be cached if the standard cache should not be used.

        force_download: (`optional`) boolean, default False:
            Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

        resume_download: (`optional`) boolean, default False:
            Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

        proxies: (`optional`) dict, default None:
            A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
            The proxies are used on each request.

        output_loading_info: (`optional`) boolean:
            Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

        kwargs: (`optional`) Remaining dictionary of keyword arguments:
            Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

            - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
            - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.
            
            - If layer pruning is supported, ``layer_pruning`` will passed as a dictionary contains layer pruning configurations as follows:
                - strategy:
                    can be one of these values: {`top`, `buttom`, `symmetric`, `alternate`, `custom`}
                - k:
                    is the number of layers to prune. mandatory if strategy is one of {`top`, `buttom`, `symmetric`, `alternate`}
                - layers_indexes:
                    is array of layers indexs to prune. mandatory if strategy is `custom`
                - is_odd:
                    is odd alternate or not. mandatory if strategy is `alternate`

    Examples::

        # For example purposes. Not runnable.
        model = BertModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
        model = BertModel.from_pretrained('./test/saved_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        model = BertModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
        assert model.config.output_attention == True
        # Loading from a TF checkpoint file instead of a PyTorch model (slower)
        config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
        model = BertModel.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_pt=True, config=config)

    """
    config = kwargs.pop("config", None)
    cache_dir = kwargs.pop("cache_dir", None)
    from_pt = kwargs.pop("from_pt", False)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    output_loading_info = kwargs.pop("output_loading_info", False)
    local_files_only = kwargs.pop("local_files_only", False)
    use_cdn = kwargs.pop("use_cdn", True)
    
    # mwahdan: Read layer_pruning config if exist
    layer_pruning = kwargs.pop("layer_pruning", None)

    # Load config if we don't provide a configuration
    if not isinstance(config, PretrainedConfig):
        config_path = config if config is not None else pretrained_model_name_or_path
        config, model_kwargs = model_class.config_class.from_pretrained(
            config_path,
            *model_args,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            **kwargs,
        )
    else:
        model_kwargs = kwargs

    # Load model
    if pretrained_model_name_or_path is not None:
        if os.path.isdir(pretrained_model_name_or_path):
            if os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                # Load from a TF 2.0 checkpoint
                archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
            elif from_pt and os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                # Load from a PyTorch checkpoint
                archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
            else:
                raise EnvironmentError(
                    "Error no file named {} found in directory {} or `from_pt` set to False".format(
                        [WEIGHTS_NAME, TF2_WEIGHTS_NAME], pretrained_model_name_or_path
                    )
                )
        elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            archive_file = pretrained_model_name_or_path
        elif os.path.isfile(pretrained_model_name_or_path + ".index"):
            archive_file = pretrained_model_name_or_path + ".index"
        else:
            archive_file = hf_bucket_url(
                pretrained_model_name_or_path,
                filename=(WEIGHTS_NAME if from_pt else TF2_WEIGHTS_NAME),
                use_cdn=use_cdn,
            )

        try:
            # Load from URL or cache if already cached
            resolved_archive_file = cached_path(
                archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
            )
            if resolved_archive_file is None:
                raise EnvironmentError
        except EnvironmentError:
            msg = (
                f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named one of {TF2_WEIGHTS_NAME}, {WEIGHTS_NAME}.\n\n"
            )
            raise EnvironmentError(msg)
        if resolved_archive_file == archive_file:
            logger.info("loading weights file {}".format(archive_file))
        else:
            logger.info("loading weights file {} from cache at {}".format(archive_file, resolved_archive_file))
    else:
        resolved_archive_file = None
        
    # mwahdan: Modify config
    if layer_pruning:
        layer_pruning_k = layer_pruning_layers_indexes = layer_pruning_is_odd = None
        layer_pruning_strategy = get_mandatory_parameter('strategy', layer_pruning)
        if layer_pruning_strategy in {'top', 'buttom', 'symmetric'}:
            layer_pruning_k = get_mandatory_parameter('k', layer_pruning)
            config, original_num_layers = modify_num_of_layers(config, k=layer_pruning_k)
        elif layer_pruning_strategy == 'custom':
            layer_pruning_layers_indexes = get_mandatory_parameter('layers_indexes', layer_pruning)
            config, original_num_layers = modify_num_of_layers(config, layers_indexes=layer_pruning_layers_indexes)
        elif layer_pruning_strategy == 'alternate':
            layer_pruning_k = get_mandatory_parameter('k', layer_pruning)
            layer_pruning_is_odd = get_mandatory_parameter('is_odd', layer_pruning)
            config, original_num_layers = modify_num_of_layers(config, k=layer_pruning_k, is_alternate=True)
        else:
            raise Exception('`%s` is not a supported layer pruning strategy' % layer_pruning_strategy)
    

    # Instantiate model.
    model = model_class(config, *model_args, **model_kwargs)
    
    
    # mwahdan: Rename layers
    if layer_pruning:
        model = rename_layers_in_strategy(model, layer_pruning_strategy, original_num_layers,
                                          layer_pruning_k, layer_pruning_layers_indexes, 
                                          layer_pruning_is_odd)

    if from_pt:
        # Load from a PyTorch checkpoint
        model = load_pytorch_checkpoint_in_tf2_model(model, resolved_archive_file, allow_missing_keys=True)
        # mwahdan: Rename layers
        if layer_pruning is not None:
            model = rename_layers(model)
        return model

    model(model.dummy_inputs, training=False)  # build the network with dummy inputs

    assert os.path.isfile(resolved_archive_file), "Error retrieving file {}".format(resolved_archive_file)
    # 'by_name' allow us to do transfer learning by skipping/adding layers
    # see https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1339-L1357
    try:
        # added skip_mismatch=True because we will prune full layers
        model.load_weights(resolved_archive_file, by_name=True, skip_mismatch=True)
        # mwahdan: Rename layers
    except OSError:
        raise OSError(
            "Unable to load weights from h5 file. "
            "If you tried to load a TF 2.0 model from a PyTorch checkpoint, please set from_pt=True. "
        )

    model(model.dummy_inputs, training=False)  # Make sure restore ops are run
    
    # mwahdan: Rename layers
    if layer_pruning is not None:
        model = rename_layers(model)

    # Check if the models are the same to output loading informations
    with h5py.File(resolved_archive_file, "r") as f:
        if "layer_names" not in f.attrs and "model_weights" in f:
            f = f["model_weights"]
        hdf5_layer_names = set(hdf5_format.load_attributes_from_hdf5_group(f, "layer_names"))
    model_layer_names = set(layer.name for layer in model.layers)
    missing_keys = list(model_layer_names - hdf5_layer_names)
    unexpected_keys = list(hdf5_layer_names - model_layer_names)
    error_msgs = []

    if len(unexpected_keys) > 0:
        logger.warning(
            f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when "
            f"initializing {model.__class__.__name__}: {unexpected_keys}\n"
            f"- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task "
            f"or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPretraining model).\n"
            f"- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint of a model that you expect "
            f"to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
        )
    else:
        logger.warning(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
    if len(missing_keys) > 0:
        logger.warning(
            f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} "
            f"and are newly initialized: {missing_keys}\n"
            f"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
        )
    else:
        logger.warning(
            f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.\n"
            f"If your task is similar to the task the model of the ckeckpoint was trained on, "
            f"you can already use {model.__class__.__name__} for predictions without further training."
        )
    if len(error_msgs) > 0:
        raise RuntimeError(
            "Error(s) in loading weights for {}:\n\t{}".format(model.__class__.__name__, "\n\t".join(error_msgs))
        )
    if output_loading_info:
        loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
        return model, loading_info

    return model




"""
Original implementation of from_pretrained:https://huggingface.co/transformers/_modules/transformers/modeling_tf_auto.html
The modified implementation by @mwahdan For layer_pruning support
"""
def from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs):
    r""" Instantiates one of the base model classes of the library
    from a pre-trained model configuration.

    The `from_pretrained()` method takes care of returning the correct model class instance
    based on the `model_type` property of the config object, or when it's missing,
    falling back to using pattern matching on the `pretrained_model_name_or_path` string:

        - `t5`: TFT5Model (T5 model)
        - `distilbert`: TFDistilBertModel (DistilBERT model)
        - `roberta`: TFRobertaModel (RoBERTa model)
        - `bert`: TFTFBertModel (Bert model)
        - `openai-gpt`: TFOpenAIGPTModel (OpenAI GPT model)
        - `gpt2`: TFGPT2Model (OpenAI GPT-2 model)
        - `transfo-xl`: TFTransfoXLModel (Transformer-XL model)
        - `xlnet`: TFXLNetModel (XLNet model)
        - `ctrl`: TFCTRLModel (CTRL model)

    Params:
        pretrained_model_name_or_path: either:

            - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
            - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
            - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
            - a path or url to a `PyTorch, TF 1.X or TF 2.0 checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In the case of a PyTorch checkpoint, ``from_pt`` should be set to True and a configuration object should be provided as ``config`` argument.

        from_pt: (`Optional`) Boolean
            Set to True if the Checkpoint is a PyTorch checkpoint.

        model_args: (`optional`) Sequence of positional arguments:
            All remaning positional arguments will be passed to the underlying model's ``__init__`` method

        config: (`optional`) instance of a class derived from :class:`~transformers.PretrainedConfig`:
            Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

            - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
            - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
            - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

        state_dict: (`optional`) dict:
            an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
            This option can be used if you want to create a model from a pretrained configuration but load your own weights.
            In this case though, you should check if using :func:`~transformers.PreTrainedModel.save_pretrained` and :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.

        cache_dir: (`optional`) string:
            Path to a directory in which a downloaded pre-trained model
            configuration should be cached if the standard cache should not be used.

        force_download: (`optional`) boolean, default False:
            Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

        resume_download: (`optional`) boolean, default False:
            Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

        proxies: (`optional`) dict, default None:
            A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
            The proxies are used on each request.

        output_loading_info: (`optional`) boolean:
            Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

        kwargs: (`optional`) Remaining dictionary of keyword arguments:
            Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

            - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
            - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

    Examples::

        model = TFAutoModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
        model = TFAutoModel.from_pretrained('./test/bert_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
        model = TFAutoModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
        assert model.config.output_attention == True
        # Loading from a TF checkpoint file instead of a PyTorch model (slower)
        config = AutoConfig.from_json_file('./tf_model/bert_tf_model_config.json')
        model = TFAutoModel.from_pretrained('./pt_model/bert_pytorch_model.bin', from_pt=True, config=config)

    """
    config = kwargs.pop("config", None)
    if not isinstance(config, PretrainedConfig):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

    for config_class, model_class in TF_MODEL_MAPPING.items():
        if isinstance(config, config_class):
            return from_pretrained_detailed(model_class, pretrained_model_name_or_path, *model_args, config=config, **kwargs)
    raise ValueError(
        "Unrecognized configuration class {} for this kind of TFAutoModel.\n"
        "Model type should be one of {}.".format(
            config.__class__, ", ".join(c.__name__ for c in TF_MODEL_MAPPING.keys())
        )
    )