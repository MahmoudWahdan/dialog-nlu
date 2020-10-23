# -*- coding: utf-8 -*-
"""
@author: mwahdan
"""

import tensorflow as tf
import os


def disable_tf_gpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def convert_to_tflite_model(model, save_file_path, conversion_mode="normal"):
    """
    model:
        normal tf keras model
    save_file_path:
        path including filename for output tflite model
    conversion_mode:
        tflite conversion mode. can be one of ["normal", "fp16_quantization", "hybrid_quantization"]
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if conversion_mode == "normal":
        converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
    elif conversion_mode == "fp16_quantization":
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.target_spec.supported_types = [tf.float16]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_converter = True
    elif conversion_mode == "hybrid_quantization":
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        converter.experimental_new_converter = True
    else:
        raise Exception("`%s` is not supported conversion_mode" % conversion_mode)

    tflite_model = converter.convert()
    open(save_file_path, "wb").write(tflite_model)