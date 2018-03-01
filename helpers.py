
'''
Matthias Rottmann
(c) 2018

DeepBASS: Deep Bayesian Active Semi-Supervised Learning

helper functions
'''


from __future__ import print_function

import sys
import os
import numpy as np

from keras.models import Model, Sequential, load_model
from keras.layers import Input
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras import regularizers, initializers
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import np_utils
from keras.datasets import mnist

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.mlab import griddata


# -------------------------------------------------------------------
# helpers for loading/generating data

def generate2Dcase( n=500, seed=0  ):

  np.random.seed(seed)

  # distribution 1
  def P1(n):
    
    radius = 1 + 0.25*np.random.randn(n)
    angle  = np.pi * ( 0.5+0.33*np.random.randn(n) )
    
    x1 = -0.33 + np.multiply( radius, np.cos(angle) ) 
    x2 = 0.1 - np.multiply( radius, np.sin(angle) )
    x1 = x1.reshape((n,1))
    x2 = x2.reshape((n,1))

    x = np.concatenate((x1,x2),axis=-1)

    return x

  # distribution 2
  def P2(n):
    
    radius = 1 + 0.25*np.random.randn(n)
    angle  = np.pi * ( 0.5+0.33*np.random.randn(n) )
    
    x1 = 0.33 + np.multiply( radius, np.cos(angle) ) 
    x2 = -0.1 + np.multiply( radius, np.sin(angle) )
    x1 = x1.reshape((n,1))
    x2 = x2.reshape((n,1))

    x = np.concatenate((x1,x2),axis=-1)

    return x
  
  # generate data
  X_train = np.concatenate((P1(n),P2(n)),axis=0)
  y_train = np.concatenate((np.zeros(n),np.ones(n)),axis=0)

  X_val   = np.concatenate((P1(n),P2(n)),axis=0)
  y_val   = np.concatenate((np.zeros(n),np.ones(n)),axis=0)

  X_train = X_train.astype('float32')
  X_val   = X_val.astype('float32')
  

  return (X_train,y_train),(X_val,y_val)


def loadMNIST():
  
  img_rows, img_cols, img_chan = 28, 28, 1
  
  (X_train, y_train), (X_val, y_val) = mnist.load_data()

  X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_chan)
  X_val   = X_val.reshape(X_val.shape[0], img_rows, img_cols, img_chan)
  X_train = X_train.astype('float32')
  X_val   = X_val.astype('float32')
  X_train /= 256.0
  X_val   /= 256.0

  return (X_train, y_train),(X_val, y_val)

# -------------------------------------------------------------------
# helpers for model definition

def model2Dcase( hidden_neurons=10, hidden_layers=1, weight_decay=1e-3, dropout_coeff=0.33, seed=0 ):

  initializer = initializers.glorot_uniform(seed=seed)
  input_shape = (2,)
  nclasses    = 2

  inp = Input(input_shape)
  y = inp

  for i in range(hidden_layers):
    y = Dense(hidden_neurons, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=initializer)(y)
    y = LeakyReLU(0.1)(y)
    y = Dropout(dropout_coeff)(y)

  y = Dense(nclasses, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=initializer)(y)
  y = Activation('softmax')(y)

  model = Model(inputs=inp,outputs=y)

  return model


def modelMNIST( kernel_size=(3,3), filters=32, blocks=2, weight_decay=1e-3, dropout_coeff=0.33, seed=0 ):
  
  initializer = initializers.glorot_uniform(seed=seed)
  input_shape = (28,28,1)
  nclasses = 10

  inp = Input(input_shape)
  y = inp

  for i in range(blocks):
    y = Conv2D( filters=filters, kernel_size=kernel_size, padding='same',
                kernel_regularizer=regularizers.l2(weight_decay), strides=1, kernel_initializer=initializer)(y)
    y = LeakyReLU(0.1)(y)
    y = Dropout(dropout_coeff)(y)
    y = Conv2D( filters=filters, kernel_size=kernel_size, padding='same',
                kernel_regularizer=regularizers.l2(weight_decay), strides=1, kernel_initializer=initializer)(y)
    y = LeakyReLU(0.1)(y)
    y = Dropout(dropout_coeff)(y)
    y = MaxPooling2D()(y)

  
  y = Flatten()(y)
  y = Dense(nclasses, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=initializer)(y)
  y = Activation('softmax')(y)

  model = Model(inputs=inp,outputs=y)
  
  return model


# -------------------------------------------------------------------
# dropout inference

def dropout_inference( model, X, batch_size=256, T=100, pred=None, dropout=1 ):

  #from keras import backend as K

  if pred == None:
    pred = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])

  nbatch = int(np.ceil(X.shape[0]/batch_size))

  Yt_hat = np.array( pred([X[ 0:batch_size ], dropout]) )

  for i in range(1,nbatch):
    Yt_tmp = np.array( pred([X[ i*batch_size:np.min(((i+1)*batch_size,X.shape[0])) ], dropout]) )
    Yt_hat = np.concatenate((Yt_hat,Yt_tmp),axis=1)
    
  for j in range(1,T):
    print(' Bayesian inference: j =',j+1,'/',T,end='\r')
    Yt_hat2 = np.array( pred([X[ 0:batch_size ], dropout]) )

    for i in range(1,nbatch):
      Yt_tmp = np.array( pred([X[ i*batch_size:np.min(((i+1)*batch_size,X.shape[0])) ], dropout]) )
      Yt_hat2 = np.concatenate((Yt_hat2,Yt_tmp),axis=1)

    Yt_hat = np.concatenate((Yt_hat,Yt_hat2),axis=0)

  print(' ')

  return Yt_hat

# -------------------------------------------------------------------
# visualization

def visualize2D( X_train, Y_train, X_start, Y_start, model, savedir="video/", savename="frame", T=10 ):
  
  cmap = cm.bwr
  graymap = cm.binary
  
  if not os.path.exists(savedir):
    os.makedirs(savedir)

  NN = 10000

  X1 = 5*np.random.rand(NN,1)-2.5
  X2 = 5*np.random.rand(NN,1)-2.5
  X = np.concatenate((X1,X2),axis=-1)
  Y = dropout_inference( model, X, T=T )
  Y = np.mean(Y,axis=0)

  xi = np.linspace(-2.5,2.5,1000)
  yi = np.linspace(-2.5,2.5,1000)
  zi = griddata(X[:,0],X[:,1],Y[:,1],xi,yi,interp='linear')
  levels = np.linspace(0.0,1.0,100)

  plt.gcf().clear()
  
  cnt = plt.contourf(xi,yi,zi, cmap='bwr', alpha=0.3, linewidths=0, levels=levels, antialiased=True )

  plt.xlim((-2.25,2.25))
  plt.ylim((-2.25,2.25))
  plt.scatter(X_train[:,0], X_train[:,1], c=cmap(Y_train[:,1]), alpha=0.5 )
  plt.scatter(X_start[:,0], X_start[:,1], c=graymap(Y_start[:,1]), marker='x', s=40, alpha=0.8 )
  
  plt.savefig(savedir+savename+'.png', format='png', dpi=380 )
  
  return

# -------------------------------------------------------------------

