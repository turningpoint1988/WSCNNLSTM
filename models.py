#/usr/bin/python

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, Flatten, Input, concatenate, BatchNormalization, TimeDistributed, Bidirectional, LSTM, Reshape, Activation
from keras import regularizers
from ANDNoisy import ANDNoisy

# DeepBind model
def DeepBind(shape = None, params = None, penalty = 0.005):
    model = Sequential()
    model.add(Conv1D(16, 24, padding='same', activation='relu', kernel_regularizer=regularizers.l2(penalty), input_shape=shape))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(penalty)))
    model.add(Dropout(params['DROPOUT']))
    model.add(Dense(1, activation='sigmoid'))
    
    print model.summary()
    return model

# DanQ model
def DanQ(shape = None, params = None, penalty = 0.005):
    model = Sequential()
    model.add(Conv1D(16, 24, padding='same', activation='relu', kernel_regularizer=regularizers.l2(penalty),input_shape=shape))
    model.add(MaxPooling1D(8))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(params['DROPOUT']))
    model.add(Dense(1, activation='sigmoid'))
    
    print model.summary()
    return model
# 
def WSCNNwithNoisy(shape = None, params = None, penalty = 0.005):
    model = Sequential()
    model.add(Conv2D(16, (1, 24), padding='same', activation='relu', kernel_regularizer=regularizers.l2(penalty), input_shape=shape))
    model.add(MaxPooling2D(pool_size=(1, 120)))
    model.add(Conv2D(32, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(penalty)))
    model.add(Dropout(params['DROPOUT']))
    model.add(Conv2D(1, (1, 1), kernel_regularizer=regularizers.l2(penalty)))
    model.add(ANDNoisy(a=7.5))
    
    print model.summary()
    return model

# 
def WSCNNwithMax(shape = None, params = None, penalty = 0.005):
    model = Sequential()
    model.add(Conv2D(16, (1, 24), padding='same', activation='relu', kernel_regularizer=regularizers.l2(penalty), input_shape=shape))
    model.add(MaxPooling2D(pool_size=(1, 120)))
    model.add(Conv2D(32, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(penalty)))
    model.add(Dropout(params['DROPOUT']))
    model.add(Conv2D(1, (1, 1), activation='sigmoid', kernel_regularizer=regularizers.l2(penalty)))
    model.add(GlobalMaxPooling2D())
    
    print model.summary()
    return model

# 
def WSCNNwithAve(shape = None, params = None, penalty = 0.005):
    model = Sequential()
    model.add(Conv2D(16, (1, 24), padding='same', activation='relu', kernel_regularizer=regularizers.l2(penalty), input_shape=shape))
    model.add(MaxPooling2D(pool_size=(1, 120)))
    model.add(Conv2D(32, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(penalty)))
    model.add(Dropout(params['DROPOUT']))
    model.add(Conv2D(1, (1, 1), activation='sigmoid', kernel_regularizer=regularizers.l2(penalty)))
    model.add(GlobalAveragePooling2D())
    
    print model.summary()
    return model

def WSCNNLSTMwithNoisy(shape = None, params = None, penalty = 0.005):
    model = Sequential()
    model.add(Conv2D(16, (1, 24), padding='same', activation='relu', kernel_regularizer=regularizers.l2(penalty), input_shape=shape))
    model.add(MaxPooling2D((1, 8)))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Bidirectional(LSTM(32))))
    model.add(Dropout(params['DROPOUT']))
    model.add(Conv1D(1, 1, kernel_regularizer=regularizers.l2(penalty)))
    model.add(ANDNoisy(a=7.5))
    
    print model.summary()
    return model

# build other models
def WSCNNLSTMwithMax(shape = None, params = None, penalty = 0.005):
    model = Sequential()
    model.add(Conv2D(16, (1, 24), padding='same', activation='relu', kernel_regularizer=regularizers.l2(penalty), input_shape=shape))
    model.add(MaxPooling2D((1, 8)))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Bidirectional(LSTM(32))))
    model.add(Dropout(params['DROPOUT']))
    model.add(Conv1D(1, 1, activation='sigmoid', kernel_regularizer=regularizers.l2(penalty)))
    model.add(GlobalMaxPooling1D())
    
    print model.summary()
    return model

def WSCNNLSTMwithAve(shape = None, params = None, penalty = 0.005):
    model = Sequential()
    model.add(Conv2D(16, (1, 24), padding='same', activation='relu', kernel_regularizer=regularizers.l2(penalty), input_shape=shape))
    model.add(MaxPooling2D((1, 8)))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Bidirectional(LSTM(32))))
    model.add(Dropout(params['DROPOUT']))
    model.add(Conv1D(1, 1, activation='sigmoid', kernel_regularizer=regularizers.l2(penalty)))
    model.add(GlobalAveragePooling1D())
    
    print model.summary()
    return model 

