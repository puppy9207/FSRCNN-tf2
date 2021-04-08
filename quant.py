#!/usr/bin/python
#-*- coding: utf-8 -*-
import tensorflow as tf
float_model = tf.keras.models.load_model('my_model.h5')
from tensorflow_model_optimization.quantization.keras import vitis_quantize
quantizer = vitis_quantize.VitisQuantizer(float_model)
quantized_model = quantizer.quantize_model(calib_dataset='./data')