
'''
Matthias Rottmann
(c) 2018

DeepBASS: Deep Bayesian Active Semi-Supervised Learning

Run command: "python3 deepbass.py resample_id test_id -- e.g.: python3 deepbass.py 0 0"
Defualt valus: resample_id=0, test_id=3.
Please check the meaning of the different test_id cases in the source code comments below 
'''

from __future__ import print_function

import sys
import os
import numpy as np

from keras.optimizers import Adam
from keras import backend as K
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

from scipy.stats import entropy

from helpers import *
  

def EM( resample_id=0, test_id=3 ):

  seed            = 1000*resample_id
  np.random.seed(seed)

  # -------------------------------------------------------------------
  # parameters

  # choose test_case = '2D' or test_case = 'MNIST'
  test_case       = 'MNIST'
  batch_size      = 256

  # algorithm parameters
  Niter           = 200
  epochs_per_iter = 1
  init_epochs     = 2000

  num_gt          = 100
  weight_gt       = 20
  mcdo_gt         = 100
  mcdo_else       = 10

  stepwise        = True
  augment         = False
  semisup         = True
  add_num         = 10
  add_type        = 'entropymax'

  visualize       = False
  csv_folder      = 'results/csv'
  model_folder    = 'results/models'


  # -------------------------------------------------------------------
  # load data and models

  X_train     = None
  X_val       = None
  y_train     = None
  y_val       = None
  nclasses    = None
  input_shape = None
  model       = None
  test_name   = None
  nclasses    = 0

  if test_case == '2D':
    (X_train,y_train),(X_val,y_val) = generate2Dcase( seed=seed )
    input_shape = (2,)
    nclasses = 2
    model = model2Dcase( hidden_neurons=50, hidden_layers=3, weight_decay=1e-3, dropout_coeff=0.33, seed=seed )
  else:
    (X_train,y_train),(X_val,y_val) = loadMNIST()
    input_shape = (28,28,1)
    nclasses = 10
    model = modelMNIST( kernel_size=(3,3), filters=16, blocks=2, weight_decay=1e-3, dropout_coeff=0.33, seed=seed )
  
  model.summary()

  Y_train = np_utils.to_categorical(y_train, nclasses)
  Y_val   = np_utils.to_categorical(y_val, nclasses)

  # -------------------------------------------------------------------

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  # -------------------------------------------------------------------
  # test selection

  # test_id is the second command line parameter for this script
  if test_id == 0:
    # purely supervised with #num_gt + #Niter ground truth labels
    num_gt     += Niter
    Niter      = 0
    stepwise   = False
    add_num    = 0
    test_name  = 'resamp'+str(resample_id)+'_all'+str(num_gt).zfill(5)+'_initmodel'
  if test_id == 1:
    # purely supervised with #num_gt ground truth labels
    Niter      = 0
    stepwise   = True
    add_num    = 0
    test_name  = 'resamp'+str(resample_id)+'_all'+str(num_gt).zfill(5)+'_initmodel'
  if test_id == 2:
    # semi-supervised, all unlabeled data from the beginning, #num_gt ground truth labels,
    # active, every #add_num iterations we add #add_num ground truth labels with max. classification entropy
    stepwise   = False
    add_type   = 'entropymax'
    test_name  = 'resamp'+str(resample_id)+'_all'+str(num_gt).zfill(5)+'_'+add_type
  if test_id == 3:
    # semi-supervised, unlabeled data added step-wise, #num_gt ground truth labels,
    # active, every #add_num iterations we add #add_num ground truth labels with max. classification entropy
    stepwise   = True
    add_type   = 'entropymax'
    test_name  = 'resamp'+str(resample_id)+'_stepwise'+str(num_gt).zfill(5)+'_'+add_type
  if test_id == 4:
    # semi-supervised, all unlabeled data from the beginning, #num_gt ground truth labels,
    # active, every #add_num iterations we randomly add #add_num ground truth labels with above avg. classification entropy
    stepwise   = False
    add_type   = 'careful'
    test_name  = 'resamp'+str(resample_id)+'_all'+str(num_gt).zfill(5)+'_'+add_type
  if test_id == 5:
    # semi-supervised, unlabeled data added step-wise, #num_gt ground truth labels,
    # active, every #add_num iterations we randomly add #add_num ground truth labels with above avg. classification entropy
    stepwise   = True
    add_type   = "careful"
    test_name  = 'resamp'+str(resample_id)+'_stepwise'+str(num_gt).zfill(5)+'_'+add_type
  if test_id == 6:
    # semi-supervised, all unlabeled data from the beginning, #num_gt ground truth labels,
    # not active
    stepwise   = False
    add_num    = 0
    test_name  = 'resamp'+str(resample_id)+'_all'+str(num_gt).zfill(5)+'_'+'semisup'
  if test_id == 7:
    # semi-supervised, unlabeled data added step-wise, #num_gt ground truth labels,
    # not active
    stepwise   = True
    add_num    = 0
    test_name  = 'resamp'+str(resample_id)+'_stepwise'+str(num_gt).zfill(5)+'_'+'semisup'
  if test_id == 8:
    # active, every #add_num iterations we add #add_num ground truth labels with max. classification entropy
    # not semi-supervised
    stepwise   = False
    semisup    = False
    add_type   = 'entropymax'
    test_name  = 'resamp'+str(resample_id)+'_none'+str(num_gt).zfill(5)+'_'+add_type
  if test_id == 9:
    # active, every #add_num iterations we randomly add #add_num ground truth labels with above avg. classification entropy
    # not semi-supervised
    stepwise   = False
    semisup    = False
    add_type   = 'careful'
    test_name  = 'resamp'+str(resample_id)+'_none'+str(num_gt).zfill(5)+'_'+add_type
  if test_id == 10:
    # purely supervised with maximal number of ground truth labels
    Niter      = 0
    add_num    = 0
    init_epochs= 100
    mcdo_gt    = 10
    num_gt     = X_train.shape[0]
    stepwise   = False
    semisup    = False
    test_name  = 'resamp'+str(resample_id)+'_all'+str(num_gt).zfill(5)+'_initmodel'
  if test_id == 11:
    # semi-supervised, all unlabeled data from the beginning, #num_gt +#Niter ground truth labels,
    num_gt     += Niter
    stepwise   = False
    add_num    = 0
    test_name  = 'resamp'+str(resample_id)+'_all'+str(num_gt).zfill(5)+'_semisup'
  if test_id == 12:
    # semi-supervised, all unlabeled data from the beginning, #num_gt +#Niter ground truth labels,
    num_gt     = 600
    stepwise   = False
    add_num    = 0
    test_name  = 'resamp'+str(resample_id)+'_all'+str(num_gt).zfill(5)+'_semisup'
  if test_id == 13:
    # semi-supervised, all unlabeled data from the beginning, #num_gt +#Niter ground truth labels,
    num_gt     = 1000
    stepwise   = False
    add_num    = 0
    test_name  = 'resamp'+str(resample_id)+'_all'+str(num_gt).zfill(5)+'_semisup'

  test_name = test_case + '_' + test_name

  # -------------------------------------------------------------------
  # prepare data and extract balanced subset

  IP = np.random.permutation(X_train.shape[0])
  X_train = X_train[ IP ]
  Y_train = Y_train[ IP ]
  y_train = y_train[ IP ]

  XC = int(num_gt / nclasses)
  X  = X_train[ y_train == 0 ]
  Y  = Y_train[ y_train == 0 ]
  Xs = X[0:XC]
  Ys = Y[0:XC]
  Xr = X[XC:X.shape[0]]
  Yr = Y[XC:Y.shape[0]]

  for i in range(1,nclasses):
    X = X_train[ y_train == i ]
    Y = Y_train[ y_train == i ]
    Xs = np.concatenate((Xs,X[0:XC]),axis=0)
    Xr = np.concatenate((Xr,X[XC:X.shape[0]]),axis=0)
    Ys = np.concatenate((Ys,Y[0:XC]),axis=0)
    Yr = np.concatenate((Yr,Y[XC:X.shape[0]]),axis=0)

  X_train = np.concatenate((Xs,Xr),axis=0)
  Y_train = np.concatenate((Ys,Yr),axis=0)
  y_train = np.argmax( Y_train, axis=-1 )

  I         = np.asarray(range(num_gt))
  I_start   = np.copy(I)
  I_all     = range(X_train.shape[0])
  I_others  = np.asarray([i for i in I_all if i not in I])

  Y_train_pseudo = np.copy(Y_train)

  X_start = X_train[I_start]
  Y_start = Y_train[I_start]

  # -------------------------------------------------------------------

  if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)
    
  if not os.path.exists(model_folder):
    os.makedirs(model_folder)

  f = open(csv_folder+'/'+test_name+'.csv','w')
  print('iteration,val_acc,gt_acc,pseudolabel_acc,num_traindata,avg_entropy_gt,avg_entropy', file=f );


  datagen = ImageDataGenerator(featurewise_center=False,
                              featurewise_std_normalization=False,
                              channel_shift_range=0.05,
                              rotation_range=10,
                              width_shift_range=0.05,
                              height_shift_range=0.05,
                              horizontal_flip=False,
                              data_format='channels_last')

  # -------------------------------------------------------------------

  for i in range(Niter+1):
    
    for j in range(1,weight_gt):
      I = np.concatenate((I,I_start),axis=0)
    
    print("\nstep",i,'of',Niter,'- training with',len(I)-(weight_gt-1)*num_gt,'samples', )

    X_sample = X_train[I]
    Y_sample = Y_train_pseudo[I]
    
    epochs   = epochs_per_iter
    if i==0:
      epochs = int(init_epochs/weight_gt)
    
    # train
    if augment == False:
      model.fit(X_sample, Y_sample, batch_size=batch_size, epochs=epochs,
                verbose=1, validation_data=None, shuffle=True )
    else:
      model.fit_generator(datagen.flow(X_sample, Y_sample, batch_size=batch_size),
                          steps_per_epoch=X_sample.shape[0]//batch_size, epochs=epochs )
      
    print("\n")
    
    # validate
    score    = model.evaluate(X_val, Y_val, batch_size=batch_size)
    
    if i==0:
      model.save(model_folder+'/'+test_name+'_initmodel.h5')

    # infer, calculate entropies and threshold
    Ysa      = dropout_inference( model, X_start, T=mcdo_gt )

    mean_ysa = np.argmax( np.mean(Ysa,axis=0), axis=-1 )
    mean_Esa = entropy( np.mean(Ysa,axis=0).T )
    y_sample = np.argmax( Y_start, axis=-1 )

    theta    = np.mean(mean_Esa)

    Ytr      = dropout_inference( model, X_train, T=mcdo_else )
    y_pseudo = np.argmax( np.mean(Ytr,axis=0), axis=-1 )
    Y_pseudo = np_utils.to_categorical(y_pseudo,nclasses)
    Entr     = entropy( np.mean(Ytr,axis=0).T )
    
    
    # write history to csv
    meanE    = np.mean(Entr)
    startacc = np.mean(y_sample == mean_ysa)
    I_pseudo =  np.setdiff1d(I, I_start)
    print( str(i)+','+str(score[1])+','+str(startacc)+','+str(np.mean(y_train[I_pseudo] == y_pseudo[I_pseudo]))+','+str(len(I)-(weight_gt-1)*num_gt)+','+str(theta)+','+str(meanE), file=f )


    if visualize == True and test_case == '2D':
      visualize2D( X_train, Y_train, X_start, Y_start, model, T=mcdo_else, savename=test_name+'_'+str(i).zfill(3) )

    # active learning part -- label acquisition
    if add_num > 0 and i%add_num == 0:
      print('adding high entropy samples')
      for j in range(add_num):
        I_switch = []

        if add_type == 'careful':
          I_switch = np.where( Entr[I_others] > theta )
          I_switch = I_others[ I_switch ]
          Ip = np.random.permutation(I_switch.size)
          I_switch = I_switch[Ip]
          I_switch = np.asarray( [ I_switch[0] ] )
        else:
          I_switch = np.argmax( Entr[I_others] )
          I_switch = np.asarray([ I_others[I_switch] ])
        
        I_start = np.concatenate((I_start,I_switch),axis=0)
        I_others = np.asarray([i for i in I_all if i not in I_start])
        #I_others = np.setdiff1d(I_others, I_switch)
        print('index:',I_switch[0],'digit:',np.argmax(Y_train[I_switch]),'entropy:',Entr[I_switch])
        
      X_start = X_train[I_start]
      Y_start = Y_train[I_start]
      num_gt += add_num
    
    
    Y_train_pseudo = np.copy(Y_train)
    if I_others.size > 0:
      print("I_others.size:",I_others.size)
      Y_train_pseudo[I_others] = Y_pseudo[I_others]


    # print some information
    E_true  = mean_Esa[ mean_ysa == y_sample ]
    E_false = mean_Esa[ mean_ysa != y_sample ]
    print(' ')
    print(' ')
    print('                                   val acc:', score[1] )
    print('           accuracy for known ground truth:', startacc )
    print('average entropy of correct classifications:', np.mean(E_true) )
    print('average entropy of   false classifications:', np.mean(E_false) )
    print('                                separation:', theta )
    print('  Correctly labeled data added to training:', np.mean(y_train[I] == y_pseudo[I]) )
    print('  Correctly labeled non-validition samples:', np.mean(y_train[I_all] == y_pseudo[I_all]) )
    
    
    # semi-supervised learning
    if stepwise == True:    
      I2 = np.asarray(np.where( Entr < theta ))
      I2 = I2.reshape(I2.shape[-1])
      I3 = np.union1d( I_start, I2 )
      I  = np.union1d( I, I3 )
    else:
      I = I_all
    if semisup == False:
      I = I_start
        
  # -------------------------------------------------------------------

  f.close()

  model.save(model_folder+'/'+test_name+'_finalmodel.h5')

  if test_case == '2D':
    visualize2D( X_train, Y_train, X_start, Y_start, model, savedir='images/', savename=str(test_id).zfill(3)+'_'+test_name, T=mcdo_else )





if __name__ == "__main__":
 
  resample_id = 0
  test_id     = 3

  if len(sys.argv) > 1:
    resample_id = int(sys.argv[1])
  if len(sys.argv) > 2:  
    test_id     = int(sys.argv[2])

  EM(resample_id,test_id)
  
  
  
  
  
