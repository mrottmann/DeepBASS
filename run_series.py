
'''
Matthias Rottmann
(c) 2018

DeepBASS: Deep Bayesian Active Semi-Supervised Learning

Run script for series of runs
'''


import os
import sys

from deepbass import *

stdoutdir = "output/MNIST1000/"
resample  = range(10)
test      = [3]


if not os.path.exists(stdoutdir):
  os.makedirs(stdoutdir)

for i in resample:
  for j in test:
    print("running net",i,"in test",j)
    
    logfile    = stdoutdir+"resamp"+str(i)+"_test"+str(j)+".txt"
    
    sys.stdout = open(logfile,'w')
    
    EM(i,j)
    
    
