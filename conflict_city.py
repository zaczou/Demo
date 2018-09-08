# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 15:36:07 2018

@author: Administrator
"""
from config import config
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras.models import Model
from keras.layers import concatenate

from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint 

from keras.optimizers import Adam
from keras import backend as K


def cnn_model():
    x_in = Input(shape=(config.maxlen, ), name='x_in')
    x_embedded = Embedding(len(word2id), config.embedding_dim, 
                           weights=[config.embedding_matrix],
                           trainable=False,
                           name='x_embedded')(x_in)
    x_dropout = Dropout(config.embedding_drop_prob)(x_embedded)
    
    pooled_tensors = []
    for filter_size in config.filter_sizes:
        x_i = Conv1D(config.no_filters, filter_size, 
                     padding='valid', activation='relu')(x_dropout)
        x_i = GlobalMaxPooling1D()(x_i)
        pooled_tensors.append(x_i)

    x_cnn = pooled_tensors[0] if len(config.filter_sizes) == 1 else concatenate(pooled_tensors, axis=-1)    
    x_dropout = Dropout(config.drop_prob)(x_cnn)
    
    pred = Dense(config.classes, activation='softmax')(x_cnn)
    model = Model(inputs=[x_in], outputs=[pred])
    return model



def amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35):
    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        eps = 1e-12
        y_pred=K.clip(y_pred,eps,1.-eps)#improve the stability of the focal loss and see issues 1 for more information        
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed



# 定义marco f1 score的相反数作为loss
def score_loss(y_true, y_pred):
    loss = 0
    for i in np.eye(4):
        y_true_ = K.constant([list(i)]) * y_true
        y_pred_ = K.constant([list(i)]) * y_pred
        loss += 0.5 * K.sum(y_true_ * y_pred_) / K.sum(y_true_ + y_pred_ + K.epsilon())
    return - K.log(loss + K.epsilon())



# 定义marco f1 score的计算公式
def score_metric(y_true, y_pred):
    y_true = K.argmax(y_true)
    y_pred = K.argmax(y_pred)
    score = 0.
    for i in range(4):
        y_true_ = K.cast(K.equal(y_true, i), 'float32')
        y_pred_ = K.cast(K.equal(y_pred, i), 'float32')
        score += 0.5 * K.sum(y_true_ * y_pred_) / K.sum(y_true_ + y_pred_ + K.epsilon())
    return score



model = cnn_model()
model.summary()
model.compile(loss='sparse_categorical_crossentropy', # 交叉熵作为loss
              optimizer=Adam(1e-3),
              metrics=[score_metric])

early_stopping = EarlyStopping(monitor='val_loss', patience=20) 
tb = TensorBoard(log_dir='./log')
checkpoint = ModelCheckpoint(config.output_path, monitor='val_loss', save_best_only=False, save_weights_only=False, mode='auto')
print('training model...............')
history = model.fit(x_train, y_train,
                    batch_size=config.batch_size,
                    epochs=config.no_epoch,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stopping, tb, checkpoint])

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

print('evaluate ...........')
score = model.evaluate(X_test, y_test, verbose=0)
