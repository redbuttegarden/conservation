"""
VGG19 Implementation based on:
Karen Simonyan and Andrew Zisserman. "Very Deep Convolutional Networks for Large-Scale Image Recognition". In: CoRR
abs/1409.1556 (2014). URL: http://arxiv.org/abs/1409.1556

Code adapted from:
Rosebrock, Adrian. “Training VGGNet on ImageNet.” Deep Learning for Computer Vision with Python, 1.3 ed., PyImageSearch,
2017.
"""
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn


class VGG19(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(VGG19, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')

            # Block 1
            self.features.add(nn.Conv2D(64, kernel_size=3, padding=1,
                                        weight_initializer=mx.init.Xavier(rnd_type='gaussian',
                                                                          factor_type='out',
                                                                          magnitude=2),
                                        bias_initializer='zeros'))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            self.features.add(nn.Conv2D(64, kernel_size=3, padding=1))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            self.features.add(nn.MaxPool2D(strides=2))
            self.features.add(nn.Dropout(rate=0.25))

            # Block 2
            self.features.add(nn.Conv2D(128, kernel_size=3, padding=1))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            self.features.add(nn.Conv2D(128, kernel_size=3, padding=1))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            self.features.add(nn.MaxPool2D(strides=2))
            self.features.add(nn.Dropout(rate=0.25))

            # Block 3
            self.features.add(nn.Conv2D(256, kernel_size=3, padding=1))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            self.features.add(nn.Conv2D(256, kernel_size=3, padding=1))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.Conv2D(256, kernel_size=3, padding=1))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            self.features.add(nn.Conv2D(256, kernel_size=3, padding=1))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            self.features.add(nn.MaxPool2D(strides=2))
            self.features.add(nn.Dropout(rate=0.25))

            # Block 4
            self.features.add(nn.Conv2D(512, kernel_size=3, padding=1))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            self.features.add(nn.Conv2D(512, kernel_size=3, padding=1))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.Conv2D(512, kernel_size=3, padding=1))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            self.features.add(nn.Conv2D(512, kernel_size=3, padding=1))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            self.features.add(nn.MaxPool2D(strides=2))
            self.features.add(nn.Dropout(rate=0.25))

            # Block 5
            self.features.add(nn.Conv2D(512, kernel_size=3, padding=1))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            self.features.add(nn.Conv2D(512, kernel_size=3, padding=1))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.Conv2D(512, kernel_size=3, padding=1))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            self.features.add(nn.Conv2D(512, kernel_size=3, padding=1))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            self.features.add(nn.MaxPool2D(strides=2))
            self.features.add(nn.Dropout(rate=0.25))

            # Block 6
            self.features.add(nn.Dense(4096, activation="relu", weight_initializer="normal",
                                       bias_initializer="zeros"))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.Dropout(rate=0.5))

            # Block 7
            self.features.add(nn.Dense(4096, activation="relu", weight_initializer="normal",
                                       bias_initializer="zeros"))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.Dropout(rate=0.5))

            self.output = nn.Dense(2, weight_initializer="normal", bias_initializer="zeros")

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x
