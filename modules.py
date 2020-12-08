#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/9 0:14
# @Author  : Leslee

import numpy as np
import tensorflow as tf

def ln(inputs, epsilon=1e-8, scope="ln"):
    '''
    todo: 看这篇论文,补充公式的详细说明。
    layer normalization. See https://arxiv.org/abs/1607.06450
    层正则化是对batch_size数据模拟逼近正态分布；
    :param inputs:2维或更高维的Tensor，第一维对应于batch_size
    :param epsilon: 浮点数，避免除零错误
    :param scope: tf域名
    :return:
        和输入数据类型及shape一致的Tensor
    '''
    with tf.variable_scope(scope, reuse= tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:] # 取最后一位的维度
        # 用于计算均值和方差，参数类型为：
        # (x, axes, shift=None, name=None, keep_dim=False)
        # 其中x形如：[batch_size, height, width, kernels]或者 [batch_size, dim]
        # 需要进行求均值/方差的维度，以列表的形式表示，如[0,1,2]表示求第0，1，2三个维度的均值/方差。
        # keep_dims：是否跟输入维度保持一致；
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta  = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ( (variance + epsilon) **(.5) ) # 输入减去均值，除以标准差
        outputs = gamma * normalized + beta
    return outputs













































