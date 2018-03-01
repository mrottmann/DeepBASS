
'''
Matthias Rottmann
(c) 2018

DeepBASS: Deep Bayesian Active Semi-Supervised Learning

Script for computing stats from csv files, i.e., averages over resamplings.
'''


import pandas as pd
import os
import numpy as np


# helpers
def stringdiff(str1,str2):
  k = np.abs(len(str1)-len(str2))
  n = np.min((len(str1),len(str2)))
  for i in range(n):
    if str1[i] != str2[i]:
      if i != 12 and i != 9:
        k += 1
  return k

def stringcommon(str1,str2):
  cstr = ''
  n = np.min((len(str1),len(str2)))
  for i in range(n):
    if str1[i] == str2[i]:
      cstr += str1[i]
  return cstr


# parameters
workdir = "results/csv/"
savedir = "results/stats/"


# grouping file names
files = np.asarray( os.listdir(workdir) )
n = len(files)
count = np.asarray(range(n))

#for _ in range(int(np.log(n)/np.log(10))): # required if more than 10 resamplings are used
for i in range(n):
  for j in range(i+1,n):
    if stringdiff(files[i], files[j]) == 0:
      count[j] = count[i]
      
ind = np.unique(count)
flists = list([])
for i in ind:
  flists.append( np.asarray( files[count==i] ) )


# computing stats
for CName in ["val_acc","num_traindata","avg_entropy","gt_acc"]: # table columns to be averaged
  for fl in flists:
    
    sf     = stringcommon(fl[0],fl[1])
    N      = len(fl)
    
    # get mean
    t      = pd.read_csv(workdir+fl[0],delimiter=',')
    j      = 1
    for i in range(1,N):
      try:
        t     += pd.read_csv(workdir+fl[i],delimiter=',')
        j     += 1
      except:
        print("\n\nerror: file",workdir+fl[i],"not complete ...\n\n")
      
    tmean  = t / j

    # get min, max and stdev
    t      = pd.read_csv(workdir+fl[0],delimiter=',')
    tmin   = t
    tmax   = t
    tsigma = (t-tmean)**2
    j      = 1
    for i in range(1,N):
      try:
        t       = pd.read_csv(workdir+fl[i],delimiter=',')
        tsigma += (t-tmean)**2
        tmin    = t.where( t < tmin, other=tmin )
        tmax    = t.where( t > tmax, other=tmax )
        j      += 1
      except:
        print("\n\nerror: file",workdir+fl[i],"not complete ...\n\n")
      
    tsigma = (tsigma/j) ** (0.5)

    print("\nstatistics computed, tables merged ...\n")

    # store results in new data frame
    f      = pd.DataFrame( {"iteration" : tmean.ix[:,0].copy(),
                            CName+"_mean" : tmean.ix[:,CName].copy(),
                            CName+"_min" : tmin.ix[:,CName].copy(),
                            CName+"_max" : tmax.ix[:,CName].copy(),
                            CName+"_sigma" : tsigma.ix[:,CName].copy()} )

    print("sample:")
    print(f.ix[0:4,:])

    print("\n...\n")

    # save data frame
    if not os.path.exists(savedir):
      os.makedirs(savedir)
    f.to_csv( savedir+CName+"_"+sf, sep=',', index=False )

    print("\nstored.\n")

