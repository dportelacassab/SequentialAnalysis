import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
import random
from scipy.stats import ortho_group     #for obtaining matrix M

import os
# path = '/Applications/Diego Alejandro/2021-2 (Internship)/Technion Project/1-Discriminant KF/'
path = '/Applications/Diego Alejandro/2021-2 (Internship)/Technion Project/1-Discriminant KF/Burkhart_Try/'
os.chdir(path);os.getcwd()

##############################################################################################
#Data Generation and Specification ###########################################################

d = 1      # number of states variables
n = 35      # number of observation variables
m = 375    # number of time steps
#number of states needs to be smaller than the number of observations?


# Delta matrix definition #################################################################
bn = 5000       # multimodal noise in time
ut = np.random.uniform(-0.5, 0.5, bn*n).reshape(bn,n)
dt = np.random.choice(np.array([3.0, 4.0, 5.0]), bn, replace=True)    #replace=True means that the values can be repeted as much as they need to complete bn samples
sa = np.apply_along_axis(lambda x,y : x+y , 0, ut,dt)                 #0-applies by rows

sa.shape
np.var(sa,0).shape
Delta = np.cov(sa.T)                                                  #this Delta needs to be positive definite
pd.DataFrame(Delta).to_csv("Delta.csv");

# A matrix definition #################################################################
K = np.random.multivariate_normal(np.zeros([d*d]), np.identity(d*d), 1); print(K.shape)
K = K.reshape((d,d));
Ktild = np.matmul(K.T,K)

if(d==1):
    if Ktild[0][0]>1:
        while Ktild[0][0]>1:
            K = np.random.multivariate_normal(np.zeros([d * d]), np.identity(d * d), 1);
            K = K.reshape((d, d));
            Ktild = np.matmul(K.T, K)
            print('fiu')
    A = Ktild
else:
    # A = Ktild/np.linalg.norm(Ktild, 'fro')  # umm A does not need to be positive definite. The only ones are S,Gamma
    A = 0.95 * np.identity(d) - 0.05
    print('forbenious norm applied for obtaining A')

pd.DataFrame( A ).to_csv("A.csv");

S = np.identity( d )
G = S - np.matmul(np.matmul(A, S), A.T);       pd.DataFrame(G).to_csv("G.csv");
Sig0 = S

# # Orthonormal matrix M in nxd
M = ortho_group.rvs(dim=n)
np.matmul(M,M.T)                                  #checking orthogonal propierty
M = M[:,0:d];                                     #take the first d columns
# print(M.shape)
# # M.dot(M.T)                                        #this one here is no longer orthogonal.

# M = np.matmul( np.matmul(A,Sig0) , A.T ) + G
# M.shape

# Generation #######################################################################
z = np.zeros([m+1,d])                       # d hidden variables
x = np.zeros([m  ,n])                       # n observed variables

mean = 10*np.ones([d]);     cov = np.identity(d)
# mean = np.ones([d])
z[0,:] = np.random.multivariate_normal(mean, cov, 1)

et = np.array([m,n])
ct = np.array([m,d])
for t in range(0,m):
    et[m,:] = np.random.uniform(-0.5, 0.5, n) + np.random.choice(np.array([3, 4, 5]), 1, replace=True)
    ct[m,:] = np.random.multivariate_normal(mean, cov, 1)
    x[t  ,:] = np.exp(np.matmul(M, z[t, :])*(1 / 10)) + et[m,:]
    z[t+1,:] = np.matmul(A,z[t,:]) + ct[m,:]

pd.DataFrame(et).to_csv("epsilon_error.csv");
pd.DataFrame(ct).to_csv("cita_error.csv");
# for t in range(0,m):
#     x[t  ,:] = np.exp(np.matmul(M, z[t, :]) * (1 / 10)) + np.random.uniform(-0.5, 0.5, n) + np.random.choice(np.array([3, 4, 5]), 1, replace=True)
#     z[t+1,:] = np.matmul(A,z[t,:]) + np.random.multivariate_normal(mean, cov, 1)


fig = plt.figure(figsize=(20, 20), dpi=1000)
for sta in range( len(z[0,:]) ):                        #let's look at all of the states
    plt.plot(z[:, sta])
plt.savefig("generated_Z.pdf")

###########################################################
# Data Partitioning #######################################
# train - validation - test
# 70%   -    20%     - 10%

k = int(m*0.7);print(k)
x_train = x[0:k,:]
z_train = z[0:k,:]

s = int(m*0.2);print(s)
x_valid = x[k:k+s,:]
z_valid = z[k:k+s,:]

q = int(m*0.1)+1;print(q) # here +1 just to complete the sample
x_test_ = x[k+s:m,:]
z_test_ = z[k+s:m,:]

# k + s + q

pd.DataFrame(x_train).to_csv("x_train.csv");  pd.DataFrame(z_train).to_csv("z_train.csv")
pd.DataFrame(x_valid).to_csv("x_valid.csv");  pd.DataFrame(z_valid).to_csv("z_valid.csv")
pd.DataFrame(x_test_).to_csv("x_test_.csv");  pd.DataFrame(z_test_).to_csv("z_test_.csv")
pd.DataFrame(x).to_csv("x.csv");              pd.DataFrame(z).to_csv("z.csv")

# H = np.matmul(np.matmul( np.linalg.inv(np.matmul(x_train.T, x_train)), x_train.T ), z_train); print(H.shape)
H = np.linalg.inv( x_train.T @ x_train ) @ x_train.T  @ z_train; print(H.shape)
pd.DataFrame(H).to_csv("H.csv");

