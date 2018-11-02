#!/usr/bin/python3
"""
@author= Paolo Grasso
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
import copy

np.random.seed(42)
matplotlib.rc('text', usetex = True)  # For LaTeX text in matplotlib plots


class SolveMinProbl(object):
    """It is the main class from which all the algorithms class are made.
    y : vector 
    A : matrix
    yval : 
    Xval :
    Xtest : float m
    mean : float
        Mean value used to de-standardize the data for the plots.
    std : 
        Standard deviation used to de-standardize the data for the plots.
    """

    def __init__(self, y, A, yval, Xval, Xtest, mean, std):
        self.matr = A
        self.Np = y.shape[0]  # Number of patients
        self.Nf = A.shape[1]  # Number of features
        self.vect = y.reshape(self.Np, 1)
        self.sol = np.zeros((self.Nf, 1), dtype=float)
        self.y_val = yval.reshape(len(yval), 1)  # Vector with validation data
        self.X_val = Xval  # Matrix with validation data
        self.X_test = Xtest
        self.err = []  # Mean square error for each iteration
        self.errval = []  # Mean square error for each iteration on validation set
        self.m = mean
        self.s = std

    def plot_w(self, title):
        """It plots the w vector with stem and with the feature labes on xaxis.
        """
        w = self.sol
        n = np.arange(self.Nf)
        plt.figure()
        plt.stem(n,w.reshape(len(w),))
        plt.ylabel('$w(n)$')
        plt.xticks(ticks=range(self.Nf), labels=[r'UPDRS$_{\mathrm{Motor}}$',
            r'Jitter$_{(\%)}$',
            r'Jitter$_{\mathrm{(Abs)}}$',r'Jitter$_{\mathrm{RAP}}$',
            r'Jitter$_{\mathrm{PPQ5}}$',r'Jitter$_{\mathrm{DDP}}$',
            r'Shimmer',
            r'Shimmer$_{\mathrm{(dB)}}$',r'Shimmer$_{\mathrm{APQ3}}$',
            r'Shimmer$_{\mathrm{APQ5}}$',r'Shimmer$_{\mathrm{APQ11}}$',
            r'Shimmer$_{\mathrm{DDA}}$','NHR','HNR','RPDE','DFA','PPE'],
            rotation='vertical')
        plt.title(title)
        plt.grid(which='both')
        #plt.show()
        #plt.ylim([-0.5,0.5])
        plt.subplots_adjust(bottom=0.25)  # margin for labels
        #plt.savefig("w"+title+".pdf")

    def print_result(self, title):
        print('%s:' %title)
        print('The optimum weight vector is:')
        print(self.sol,"\n")

    def plot_err(self, title='Algorithm', logy=1, logx=0):
        """It plots the MSE in different log scales.
        Default: semilogy

        Parameters
        ----------
        logy : boolean
            It modifies the y axis scale into logarithmic
        logx : boolean
            It modifies the x axis scale into logarithmic
        """
        err = self.err
        errval = self.errval
        plt.figure()

        # Linear plot
        if (logy == 0) & (logx == 0):
            plt.semilogy(err, color='tab:blue')
            plt.semilogy(errval, color='tab:gray', linestyle=':')

        # Semilogy plot
        if (logy == 1) & (logx == 0):
            plt.semilogy(err, color='tab:blue')
            plt.semilogy(errval, color='tab:gray', linestyle=':')

        # Semilogx plot
        if (logy == 0) & (logx == 1):
            plt.semilogy(err, color='tab:blue')
            plt.semilogy(errval, color='tab:gray', linestyle=':')

        # Loglog plot
        if (logy == 1) & (logx == 1):
            plt.semilogy(err, color='tab:blue')
            plt.semilogy(errval, color='tab:gray', linestyle=':')

        plt.xlabel('$n$')
        plt.ylabel('$e(n)$')
        plt.title(title+': mean square error')
        #plt.margins(0.01,0.1)
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='xkcd:lilac', linestyle=':')
        plt.grid(b=True, which='major')
        plt.legend(['Training set','Validation set'])
        plt.show()

    def graphics(self,title):
        """
        It de-standardize the data with original std and mean values.
        It plots the histogram and scatter plot graphics.
        Histogram: y-yhat
        Scatter: yhat vs y
        """
        vect = self.vect.reshape(len(self.vect),)*self.s + self.m  # de-standardize
        yhat = self.yhat.reshape(len(self.yhat),)*self.s + self.m  # de-standardize
        yhat_test =  self.yhat_test.reshape(len(self.yhat_test),)*self.s + self.m  

        # Histogram.
        plt.figure()
        plt.hist(vect-yhat, bins=50)
        plt.title(r'$y_{\mathrm{train}} - \hat{y}_{\mathrm{train}}$')
        plt.xlabel('Error')
        plt.ylabel('Number of entries')
        plt.grid()
        plt.title(title)
        #plt.show()

        #  Scatter plot.
        plt.figure()
        #plt.scatter(yhat, vect)
        plt.scatter(yhat, vect, marker="2")
        plt.title(title+': '+r'$\hat{y}_{\mathrm{train}}$ vs $y_{\mathrm{train}}$')
        plt.grid()
        plt.xlabel(r'$y_{\mathrm{train}}$')
        plt.ylabel(r'$\hat{y}_{\mathrm{train}}$')
        plt.axis('equal')
        lined = [min(yhat), max(yhat)]
        plt.plot(lined, lined, color='tab:orange')  # Diagonal line
        #plt.show()
        plt.savefig("scatter_"+title+".pdf")


class SolveLLS(SolveMinProbl):

    def run(self):
        A = self.matr
        y = self.vect
        w = np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),y)
        self.sol = w
        self.min = np.linalg.norm(np.dot(A,w)-y)**2
        self.yhat = np.dot(A,self.sol).reshape(len(y),)
        self.yhat_test = np.dot(self.X_test,self.sol)

class SolveGrad(SolveMinProbl):

    def run(self, gamma=1e-5, Nit=2500, eps=1e-3):
        A = self.matr
        y = self.vect
        w = np.random.rand(self.Nf,1)
        
        for it in range(Nit):
            grad = 2 * np.dot(A.T,(np.dot(A,w)-y))
            w2 = w - gamma*grad

            if np.linalg.norm(w2-w) < eps:
                w = copy.deepcopy(w2)
                print("Gradient descent has stopped after %d iterations, MSE = %4f" %(it,self.err[-1]))
                break

            w=copy.deepcopy(w2)
            self.err.append((np.linalg.norm(np.dot(A,w)-y)**2)/self.Np)
            self.errval.append(np.linalg.norm(np.dot(self.X_val,w)-self.y_val)**2/len(self.y_val))
            #if (gamma*(np.linalg.norm(grad))<eps):
            #   print("Conjugate gradient stopped after %d iterations, ERR = %4f" %(it,self.err[-1]))
            #   break

        self.sol = w
        self.min = self.err[-1]
        self.yhat = np.dot(A,self.sol).reshape(len(y),)
        self.yhat_test = np.dot(self.X_test,self.sol)


class SolveStochasticGradient(SolveMinProbl):

    def run(self, Nit=2500, gamma=1e-5, eps=1e-3):
        A = self.matr
        y = self.vect
        w = np.random.rand(self.Nf,1)

        for it in range(Nit):
            w2 = copy.deepcopy(w)

            for i in range(self.Np):
                grad_i = 2*(np.dot(A[i,:].T,w)-y[i]) * A[i,:].reshape(len(A[i,:]),1)
                w = w - gamma*grad_i

            if np.linalg.norm(w2-w) < eps:
                print("Stochastic gradient descent has stopped after %d iterations, MSE = %4f" %(it,self.err[-1]))
                break

            self.err.append(np.linalg.norm(np.dot(A,w)-y)**2/self.Np)
            #if (gamma*(np.linalg.norm(grad_i))<eps):
            #   print("Stochastic gradient stopped after %d iterations, ERR = %4f" %(it,self.err[-1]))
            #   break

        self.sol = w
        self.min = self.err[-1]
        self.yhat = np.dot(A,self.sol).reshape(len(y),)
        self.yhat_test = np.dot(self.X_test,self.sol)


class SolveConjugateGradient(SolveMinProbl):

    def run(self):
        A = self.matr
        y = self.vect
        w = np.zeros((self.Nf,1),dtype=float)
        Q = np.dot(A.T,A)
        b = np.dot(A.T,y)
        d = b
        g = -b

        for it in range(self.Nf):  # Iterations on number of features
            alpha = -((np.dot(d.T,g))/(np.dot(np.dot(d.T,Q),d)))
            w = w + alpha*d
            #g = np.dot(Q,w) - b
            g = g + alpha*(np.dot(Q,d))
            beta = np.dot(np.dot(g.T,Q),d)/np.dot(np.dot(d.T,Q),d)
            d = -g + beta*d
            self.err.append(np.linalg.norm(np.dot(A,w)-y)**2/self.Np)
            self.errval.append(np.linalg.norm(np.dot(self.X_val,w)-self.y_val)**2/len(self.y_val))
 
        self.sol = w
        self.min = self.err[-1]
        self.yhat = np.dot(A,self.sol).reshape(len(y),)
        self.yhat_test = np.dot(self.X_test,self.sol)


class SolveSteepestDescent(SolveMinProbl):

    def run(self, Nit=2500, eps=1e-3):
        A = self.matr
        y = self.vect
        w = np.random.rand(self.Nf,1)

        for it in range(Nit):        
            H = 4*np.dot(A.T,A)  # Hessian matrix
            grad = 2 * np.dot(A.T,(np.dot(A,w)-y))
            w2 = w - (((np.linalg.norm(grad))**2)/(np.dot(np.dot(grad.T,H),grad)))*grad

            if np.linalg.norm(w2-w) < eps:
                w=copy.deepcopy(w2)
                print("Steepest descent has stopped after %d iterations, MSE = %4f" %(it,self.err[-1]))
                break

            w = copy.deepcopy(w2)
            self.err.append(np.linalg.norm(np.dot(A,w)-y)**2/self.Np)
            self.errval.append(np.linalg.norm(np.dot(self.X_val,w)-self.y_val)**2/len(self.y_val))

        self.sol = w
        self.min = self.err[-1]
        self.yhat = np.dot(A,self.sol).reshape(len(y),)
        self.yhat_test = np.dot(self.X_test,self.sol)


class SolveRidge(SolveMinProbl):

    def run(self, Lambda=range(0,200)):
        """

        """
        self.lambda_range = Lambda

        for L in Lambda:
            A = self.matr
            y = self.vect
            w=np.random.rand(self.Nf,1)
            I = np.eye(self.Nf)
            w = np.dot(np.dot(np.linalg.inv((np.dot(A.T,A) + L*I)),A.T),y)
            self.err.append(float(np.linalg.norm(np.dot(A,w)-y))**2/self.Np)
            self.errval.append(float(np.linalg.norm(np.dot(self.X_val,w)-self.y_val))**2/len(self.y_val))
            self.min=min(self.err)

            if self.err[-1] <= self.min:
                self.sol = w
                self.yhat = np.dot(A,self.sol).reshape(len(y),)
                self.yhat_test = np.dot(self.X_test,self.sol)
        
    def plotRidgeError(self):  
    """
    Plot ridge regression mean square error vs lambda values. 
    It takes all parameters from the main class except from lambda_range that
    is taken from the subclass SolveRidge.
    """
        plt.figure()
        plt.plot(self.lambda_range,self.err,color='tab:blue')
        plt.plot(self.lambda_range,self.errval,color='tab:gray',linestyle=':')
        plt.xlabel(r'$\lambda$')
        plt.ylabel('Mean Square Error')
        plt.title('Ridge regression: mean square error')
        plt.grid()
        plt.show()
        plt.legend(['Training set','Validation set'])


if __name__ == '__main__':

    F0 = 1  # Shimmer column
    data = pd.read_csv("parkinsons_updrs.csv") # Import CSV into a dataframe
    data = data.drop(columns=['subject#','age','sex','test_time']) # Drop the first columns
    data=data.sample(frac=1).reset_index(drop=True) # Shuffle rows and reset index 

    # Submatrices: training, validation and test.
    data_train = data[0:math.ceil(data.shape[0]/2)-1]
    data_val = data[math.floor(data.shape[0]/2):math.floor(3/4*data.shape[0])]
    data_test = data[math.floor(3/4*data.shape[0]):data.shape[0]]

    # Data normalization.
    data_train_norm = copy.deepcopy(data_train)  # To preserve original data
    data_val_norm = copy.deepcopy(data_val)  # To preserve original data
    data_test_norm = copy.deepcopy(data_test)  # To preserve original data
    
    for i in range(data_train.shape[1]):
        mean = np.mean(data_train.iloc[:,i])  # Calculate mean for data_train
        data_train_norm.iloc[:,i] -= mean
        data_val_norm.iloc[:,i] -= mean
        data_test_norm.iloc[:,i] -= mean
        std = np.std(data_train.iloc[:,i])  # Calculate standard deviation for data_train
        data_train_norm.iloc[:,i] /=  std
        data_val_norm.iloc[:,i] /= std
        data_test_norm.iloc[:,i] /= std

    # Mean and standard deviation in order to de-standardize data for the plots.
    m = np.mean(data_train.iloc[:,F0])
    s = np.std(data_train.iloc[:,F0])
    
    y_train = data_train_norm.iloc[:,F0] # F0 column vector
    y_test = data_test_norm.iloc[:,F0] # F0 column vector
    y_val = data_val_norm.iloc[:,F0] # F0 column vector
    X_train = data_train_norm.drop(columns='total_UPDRS') # Remove column F0
    X_test = data_test_norm.drop(columns='total_UPDRS') # Remove column F0
    X_val = data_val_norm.drop(columns='total_UPDRS') # Remove column F0
    
    # Class initializations.
    lls=SolveLLS(y_train.values,X_train.values,y_val.values,X_val.values,X_test.values,m,s)
    gd=SolveGrad(y_train.values,X_train.values,y_val.values,X_val.values,X_test.values,m,s)
    cgd=SolveConjugateGradient(y_train.values,X_train.values,y_val.values,X_val.values,X_test.values,m,s)
    sgd=SolveStochasticGradient(y_train.values,X_train.values,y_val.values,X_val.values,X_test.values,m,s)
    sd=SolveSteepestDescent(y_train.values,X_train.values,y_val.values,X_val.values,X_test.values,m,s)
    ridge=SolveRidge(y_train.values,X_train.values,y_val.values,X_val.values,X_test.values,m,s)
    
    # Linear least squares
    lls.run()
    lls.plot_w('LLS')
    #lls.print_result('LLS')
    #lls.graphics('LLS')

    # Gradient descent
    gd.run()
    gd.plot_w('Gradient descent')
    #gd.print_result('GD')
    #gd.plot_err('GD')
    #gd.graphics('Gradient descent')

    # Conjugate gradient descent
    cgd.run()
    cgd.plot_w('Conjugate gradient descent')
    #cgd.plot_err('Conjugate gradient descent')
    #cgd.graphics('Conjugate gradient')

    # Stochastic gradient descent
    sgd.run()
    sgd.plot_w('SGD')
    #sgd.plot_err('SGD')
    #sgd.graphics('Stochastic gradient descent')

    # Steepest descent
    sd.run()
    sd.plot_w('Steepest descent')
    #sd.plot_err('Steepest descent')
    #sd.graphics('Steepest descent')

    # Ridge regression
    ridge.run()
    ridge.plot_w('Ridge regression')
    #ridge.plotRidgeError()
    #ridge.graphics('Ridge regression')

    plt.show()
    print("--- END ---")