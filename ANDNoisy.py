#!/usr/bin/python

from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import RandomUniform
from keras.constraints import min_max_norm
import numpy as np

class ANDNoisy(Layer):
    
    def __init__(self, a = 2, **kwargs):
        self.a = a
        print "a = {}".format(self.a)
        super(ANDNoisy, self).__init__(**kwargs)
        
    def build(self, input_shape):
        initializer_uniform = RandomUniform(minval=0, maxval=1)
        constraint_min_max = min_max_norm(min_value=0.0, max_value=1.0)
        self.b = self.add_weight(name='b', shape=(input_shape[-1],), initializer=initializer_uniform, constraint=constraint_min_max, trainable=True)
        super(ANDNoisy, self).build(input_shape)
        
    def call(self, inputs):
        if K.ndim(inputs) == 4:
           inputs = K.squeeze(inputs, 2)
        part1 = K.sigmoid((K.mean(inputs, axis=1) - self.b) * self.a) - K.sigmoid(-self.a * self.b)
        #print K.int_shape(part1)
        part2 = K.sigmoid(self.a * (1 - self.b)) - K.sigmoid(-self.a * self.b)
        #print K.int_shape(part2)
        
        return (part1/part2)
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

