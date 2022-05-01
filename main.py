import pandas as pd
import numpy as np
import os
import io
import random
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from numpy.linalg import *
import future
from future.moves import tkinter
import matplotlib.pyplot as plt

path = '/Applications/Diego Alejandro/2021-2 (Internship)/Technion Project/1-Discriminant KF/Burkhart_Try/'
os.chdir(path);os.getcwd()

def svd_inv(Matrix_):
    if len(Matrix_.shape)==1 : return np.array([[ 1.0/Matrix_[0][0] ]])
    if(np.linalg.det(Matrix_)==0): print('Matrix not invertible, its det equals zero')
    U, S, Vt = np.linalg.svd(Matrix_)
    S = [1 / s if s > 0 else 0 for s in S]
    Matrix_inv = np.matmul( np.matmul(Vt.T, np.diag(S)) , U.T )
    # Matrix_inv = np.linalg.inv(Matrix_)
    return Matrix_inv

# Z = np.array([ [s2_cita1t, 0], [0,s2_cita2t ] ])
# Ptt = np.array([ [s2_cita1t, 0], [0,s2_cita2t ] ])
# btt = np.array([x0]).T
#
# def kf_(x0,F,btt,Ptt,c_,s2_eps,y_):
#     n =
#     d =
#     X_kalman = np.zeros([n+1,d]);
#     X_kalman[0,] = x0
#     for i in range(1, n):
#         beta_t_t1 = F @ btt
#         p_t_t1 = F @ Ptt @ F.T + Z
#         q_t_t1 = (c_.T @ p_t_t1 @ c_)[0][0] + s2_eps
#         Kt = p_t_t1 @ c_/q_t_t1
#         Ptt = p_t_t1 - q_t_t1* Kt @ Kt.T
#         y_t_t1 = ( c_.T @ beta_t_t1)[0][0]
#         et = ( y_[i] - y_t_t1 )
#         btt = beta_t_t1 + Kt*et
#         X_kalman[i,] = np.array( [ btt[0][0] , btt[1][0] ] )
#     return X_kalman

def kf_(v0_,b_,Delta_,Phi0,x_,H_,d_):
    m_ = len(x_[:,0])
    Phit = Phi0
    v = np.zeros([m_,d_])
    v[0,:] = v0_
    Id_ = np.identity(d_)
    for i in range(1, m_):
        Kt = Phit @ H_.T @ svd_inv( H_ @ Phit @ H_.T  + Delta_)
        Phit = ( Id_ - Kt @ H_ ) @ Phit
        v[i, :] = v[i - 1, :] + Kt @ ( x_[i, :] - b_ - H_ @ v[i - 1, :] )
    return v

def dkf_(A_,Sig_0,G_,Q_,S_,x_,regressor_):                          #when S_ = 0, then dkf becomes robust dkf.
    d_ = len(A_[0,:]);   m_ = len(x_[:,0]);
    mu = np.zeros([m_,d_]);
    mu[0,:] = regressor_.predict(x_[0, :][None, ...])
    Sig = Sig_0;
    Q_inv = svd_inv(Q_); S_inv = svd_inv(S_)
    for i in range(1, m_):
        f_ = regressor_.predict(x_[i, :][None, ...])
        M_inv = svd_inv( A_ @ Sig @ A_.T + G_ )
        Sig = svd_inv( Q_inv + M_inv - S_inv )
        mu[i,:] = Sig @ ( Q_inv@f_ + M_inv@A_@mu[i-1,:] )
    return mu


if __name__ == '__main__':
    d = 1;           n = 35;         m = 375;
    x_train = pd.read_csv('x_train.csv', header=0).values[:, 1:(n + 1)];  z_train = pd.read_csv('z_train.csv', header=0).values[:, 1:(d + 1)];
    x_valid = pd.read_csv('x_valid.csv', header=0).values[:, 1:(n + 1)];  z_valid = pd.read_csv('z_valid.csv', header=0).values[:, 1:(d + 1)];
    x_test_ = pd.read_csv('x_test_.csv', header=0).values[:, 1:(n + 1)];  z_test_ = pd.read_csv('z_test_.csv', header=0).values[:, 1:(d + 1)];

    A = pd.read_csv('A.csv', header=0).values[:, 1:d + 1];   print(A.shape)
    # G = np.identity(d)
    G = pd.read_csv('G.csv', header=0).values[:, 1:d + 1];
    # G = np.mat(np.cov(z_train[1:, :] - np.matmul(z_train[:-1, :], A), rowvar=False, ))

    S = np.identity(d)
    # S = np.cov(z_train.T)[None,...][None,...]
    S = np.matmul(np.matmul(A, S), A.T) + G;

    Delta = pd.read_csv('Delta.csv', header=0).values[:, 1:n+1];   print(Delta.shape)

    # Phi_0 = A*S*A.T + G;                print(Phi_0.shape)
    Phi_0 = A @ S @ A.T + G;            print(Phi_0.shape)
    Phi_0 = Phi_0 - 0.8;
    b = np.zeros([n])
    v0 = np.matmul(A, np.random.multivariate_normal(10 * np.ones([d]), np.identity(d), 1)[0])
    v0 = 0
    ##############################################################################################################
    # KF  #############################################
    H = pd.read_csv('H.csv', header=0).values[:, 1:d + 1]   #matrix of size nxd by definition
    print(H.shape)
    #       (v0_ , b_, Delta_, Phi0, x_, H_, d_)
    z_kf = kf_(v0,b,Delta,Phi_0,x_test_,H,d)
    print( z_kf.shape )

    ##############################################################################################################
    # DKF #############################################
    regr = MLPRegressor(random_state=1, max_iter=100000).fit(x_train, z_train)
    z_nw = regr.predict(x_valid)
    z_nw.shape

    fig = plt.figure(figsize=(20, 20), dpi=1000)
    if d == 1: plt.plot(z_valid[:, 0], 'b', z_nw[..., None], 'r')
    else:
        for sta in range(len(z_valid[0, :])):
            plt.plot(z_valid[:, sta], 'b', z_nw[:, sta], 'r')

    if len(z_nw.shape)==1: z_nw=z_nw[...,None]
    Q = np.cov( (z_valid - z_nw).T )  # cov of the residuals
    Q.shape
    if(d==1): Q = Q[None,None]

    # S_dkf = A*S*A.T + G;    print(S_dkf.shape)
    # S_dkf = np.matmul(np.matmul(A, S), A.T) + G;     print(S_dkf.shape)
    S_dkf = A @ S @ A.T + G;     print(S_dkf.shape)
    Sig0 = S_dkf
    # Sig0 = np.identity(d)

    #          (A_,Sig_0,G_,Q_,S_,x_, regressor_)
    z_dkf = dkf_(A,Sig0,G,Q,S,x_test_,regr) # Finally, apply the DKF algorithm on the test set...
    print( z_dkf.shape )

    ##############################################################################################################
    # Comparison #################################################################################################

    print('MSE: Real vs KF   '+str( mean_squared_error( z_test_ , z_kf  ) ) )
    print('MSE: Real vs DKF  '+str( mean_squared_error( z_test_ , z_dkf ) ) )

    fig = plt.figure(figsize=(20, 20), dpi=1000)
    for sta in range(len(z_test_[0, :])):
        plt.plot( z_test_[:, sta],'b', z_kf[:,sta],'g', z_dkf[:,sta],'r' )
        plt.legend(['Test', 'KF', 'DKF'], loc="best", prop={'size': 10});
    plt.savefig("comparison.pdf")
    plt.show()
