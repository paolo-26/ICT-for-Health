"""
@author: Paolo Grasso
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math
import copy

#import sklearn

np.random.seed(42)

matplotlib.rc('text', usetex = True)

class SolveMinProbl(object):
	def __init__(self,y,A,Xval,yval):
		self.matr=A
		self.Np=y.shape[0]
		self.Nf=A.shape[1]
		self.vect=y.reshape(self.Np,1)
		self.sol=np.zeros((self.Nf,1),dtype=float)
		self.y_val=yval.reshape(len(yval),1)
		self.X_val=Xval
		self.err=[]
		self.errval=[]
		return

	def plot_w(self,title):
		w=self.sol
		n=np.arange(self.Nf)
		plt.figure()
		plt.stem(n,w.reshape(len(w),))
		#plt.xlabel('$n$')
		plt.ylabel('$w(n)$')
		plt.xticks(ticks=range(self.Nf),labels=[r'UPDRS$_{\mathrm{Motor}}$',r'UPDRS$_{\mathrm{Total}}$',r'Jitter$_{(\%)}$',r'Jitter$_{\mathrm{(Abs)}}$',r'Jitter$_{\mathrm{RAP}}$',r'Jitter$_{\mathrm{PPQ5}}$',r'Jitter$_{\mathrm{DDP}}$',r'Shimmer$_{\mathrm{(dB)}}$',r'Shimmer$_{\mathrm{APQ3}}$',r'Shimmer$_{\mathrm{APQ5}}$',r'Shimmer$_{\mathrm{APQ11}}$',r'Shimmer$_{\mathrm{DDA}}$','NHR','HNR','RPDE','DFA','PPE'],rotation='vertical')
		plt.title(title)
		plt.grid(which='both')
		#plt.show()
		plt.ylim([-0.5,0.5])
		plt.subplots_adjust(bottom=0.25)
		#plt.savefig("w"+title+".pdf")
		return

	def print_result(self,title):
		print('%s:' %title)
		print('the optimum weight vector is:')
		print(self.sol,"\n")
		return

	def plot_err(self,title='Algorithm',logy=0,logx=0):
		err=self.err
		errval=self.errval
		#print("err=\n",err)
		plt.figure()
		if (logy==0) & (logx==0):
			plt.plot(err)
			plt.plot(errval)
		if (logy==1) & (logx==0):
			plt.semilogy(err)
		if (logy==0) & (logx==1):
			plt.semilogx(err)
		if (logy==1) & (logx==1):
			plt.loglog(err)
		plt.xlabel('$n$')
		plt.ylabel('$e(n)$')
		plt.title(title+' MSE')
		plt.margins(0.01,0.1)
		plt.grid(which='both')
		#plt.show()
		return

	def graphics(self):
		plt.figure()
		vect=self.vect.reshape(len(self.vect),)
		yhat=self.yhat.reshape(len(self.yhat),)
		plt.hist(vect-yhat,bins=50)
		print("---")
		plt.title(r'$y_{train} - \hat{y}_{train}$')
		plt.xlabel('Error')
		plt.ylabel('Number of entries')
		plt.grid()
		plt.show()


		plt.figure()
		plt.plot(self.yhat,self.vect,'.')
		plt.title(r'$\hat{y}_{train}$ vs $y_{train}$')
		plt.grid()
		plt.xlabel(r'$y_{train}$')
		plt.ylabel(r'$\hat{y}_{train}$')
		plt.show()


class SolveLLS(SolveMinProbl):
	def run(self):
		A=self.matr
		y=self.vect
		w=np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),y)**2
		self.sol=w
		self.min=np.linalg.norm(np.dot(A,w)-y)
		self.yhat=np.dot(A,self.sol).reshape(len(y),)

class SolveGrad(SolveMinProbl):
	def run(self,gamma=1e-5,Nit=250000000000000000000000,eps=1e-6):
		#self.err=np.zeros((Nit,2),dtype=float)
		#self.err = []
		A=self.matr
		y=self.vect
		#y=y.reshape(self.Np,1)
		w=np.random.rand(self.Nf,1)
		
		for it in range(Nit):
			grad=2*np.dot(A.T,(np.dot(A,w)-y))
			#grad=np.dot(np.dot(A.T,w)-y,A)
			#print("grad=",grad)
			w2=w-gamma*grad
			#self.err[it,0]=it
			

			if (np.linalg.norm(w2-w)<eps):
				w=copy.deepcopy(w2)
				print("Gradient descent has stopped after %d iterations, ERR = %4f" %(it,self.err[-1]))
				break
			w=copy.deepcopy(w2)
			self.err.append(np.linalg.norm(np.dot(A,w)-y)**2/self.Np)
			self.errval.append(np.linalg.norm(np.dot(self.X_val,w)-self.y_val)**2/len(self.y_val))
			#if (gamma*(np.linalg.norm(grad))<eps):
			#	print("Conjugate gradient stopped after %d iterations, ERR = %4f" %(it,self.err[-1]))
			#	break

		self.sol=w
		#print(w)
		self.min=self.err[-1]
		self.yhat=np.dot(A,self.sol).reshape(len(y),)

class SolveStochasticGradient(SolveMinProbl):
	def run(self,Nit=6000,gamma=1e-5,eps=1e-5):
		#self.err=np.zeros((Nit,2),dtype=float)
		#self.err=[]
		A=self.matr
		y=self.vect
		#y=y.reshape(self.Np,1)
		w=np.random.rand(self.Nf,1)


		for it in range(Nit):
			w2=copy.deepcopy(w)
			for i in range(self.Np):
				#print(np.dot(A[i,:].T,w)-y[i])
				grad_i = 2*(np.dot(A[i,:].T,w)-y[i])*A[i,:].reshape(len(A[i,:]),1)
				w=w-gamma*grad_i
			if (np.linalg.norm(w2-w)<eps):
				print("Stochastic gradient descent has stopped after %d iterations, ERR = %4f" %(it,self.err[-1]))
				break
			#self.err[it,0]=it
			#self.err[it,1]=np.linalg.norm(np.dot(A,w)-y)**2
			self.err.append(np.linalg.norm(np.dot(A,w)-y)**2)
			#if (gamma*(np.linalg.norm(grad_i))<eps):
			#	print("Stochastic gradient stopped after %d iterations, ERR = %4f" %(it,self.err[-1]))
			#	break
		self.sol=w
		print(w)
		self.min=self.err[-1]
		self.yhat=np.dot(A,self.sol).reshape(len(y),)

class SolveConjugateGradient(SolveMinProbl):
	def run(self):
		#self.err=np.zeros((self.Nf,2),dtype=float)
		#self.err=[]
		A=self.matr
		y=self.vect
		#y=y.reshape(self.Np,1)
		w = np.zeros((self.Nf,1),dtype=float)
		Q = np.dot(A.T,A)
		b = np.dot(A.T,y)
		d = b
		g = -b

		for it in range(self.Nf):
			#if (np.dot(np.dot(d.T,Q),d) == 0):
			#	break
			alpha = -((np.dot(d.T,g))/(np.dot(np.dot(d.T,Q),d)))
			#if np.isnan(alpha):
			#	break
			w = w + alpha * d
			#g = np.dot(Q,w) - b
			g = g+alpha*(np.dot(Q,d))
			beta = np.dot(np.dot(g.T,Q),d)/np.dot(np.dot(d.T,Q),d)
			d = -g +beta*d

			#self.err[it,0]=it
			#self.err[it,1]=np.linalg.norm(np.dot(A,w)-y)**2
			self.err.append(np.linalg.norm(np.dot(A,w)-y)**2/self.Np)
			self.errval.append(np.linalg.norm(np.dot(self.X_val,w)-self.y_val)**2/len(self.y_val))
			#print(self.err[it,1])

		self.sol=w
		self.min=self.err[-1]
		self.yhat=np.dot(A,self.sol).reshape(len(y),)


class SolveSteepestDescent(SolveMinProbl):
	def run(self,Nit=6000,eps=1e-5):
		#self.err=np.zeros((Nit,2),dtype=float)
		#self.err=[]
		A=self.matr
		y=self.vect
		#y=y.reshape(self.Np,1)
		w=np.random.rand(self.Nf,1)

		for it in range(Nit):        
			H=4*np.dot(A.T,A)
			grad=2*np.dot(A.T,(np.dot(A,w)-y))
			w2=w-(((np.linalg.norm(grad))**2)/(np.dot(np.dot(grad.T,H),grad)))*grad

			if (np.linalg.norm(w2-w)<eps):
				w=copy.deepcopy(w2)
				print("Steepest descent has stopped after %d iterations, ERR = %4f" %(it,self.err[-1]))
				break
			w=copy.deepcopy(w2)
			#self.err[it,0]=it
			#self.err[it,1]=np.linalg.norm(np.dot(A,w)-y)**2
			self.err.append(np.linalg.norm(np.dot(A,w)-y)**2)

		self.sol=w
		self.min=self.err[-1]
		self.yhat=np.dot(A,self.sol).reshape(len(y),)

class SolveRidge(SolveMinProbl):
	def run(self,Lambda=range(0,200)):
		#self.err=np.zeros((Nit,2),dtype=float)
		#self.err=[]
		#self.errval=[]
		self.lambda_range=Lambda
		for l in Lambda:
			A=self.matr
			y=self.vect
			#y=y.reshape(self.Np,1)
			w=np.random.rand(self.Nf,1)
			I = np.eye(self.Nf)

			w = np.dot(np.dot(np.linalg.pinv((np.dot(A.T,A) + l*I)),A.T),y)
			self.err.append(float(np.linalg.norm(np.dot(A,w)-y))**2/self.Np)
			self.errval.append(float(np.linalg.norm(np.dot(self.X_val,w)-self.y_val))**2/len(self.y_val))

			print("Ridge: ", self.err[-1])
			self.min=min(self.err)
			if self.err[-1] <= self.min:
				self.sol=w
				self.yhat=np.dot(A,self.sol).reshape(len(y),)
		
	def plotRidgeError(self):
		#matplotlib.ticker.ScalarFormatter()
		plt.figure()
		plt.plot(self.lambda_range,self.err)
		plt.plot(self.lambda_range,self.errval)
		#ax=plt.gca()
		#ax.ticklabel_format(scilimits=None)
		#plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
		plt.xlabel(r'$\lambda$')
		plt.ylabel('Mean Square Error')
		plt.title('Ridge')
		plt.grid()
		#plt.show()
		
		#plt.yticks([min(self.err),max(self.err)])

#def standardizeData(dataset):
	#new_dataset = copy.deepcopy(dataset)

	#for i in range(dataset.shape[1]):
		
		#mean = np.mean(new_dataset.iloc[:,i])
		#new_dataset.iloc[:,i] = new_dataset.iloc[:,i] - mean	

		#std = np.std(new_dataset.iloc[:,i])
		#new_dataset.iloc[:,i] = new_dataset.iloc[:,i] / std

		#print("mean(",i,") = ", np.mean(dataset.iloc[:,i]))
		#print("var = ",  np.var(dataset.iloc[:,i]),"\n")




if __name__ == '__main__':

	data=pd.read_csv("parkinsons_updrs.csv")
	data=data.drop(columns=['subject#','age','sex','test_time'])
	#data=data.sample(frac=1).reset_index(drop=True)
	

	#submatrices TRAIN, VAL, TEST
	data_train = data[0:math.ceil(data.shape[0]/2)-1]
	data_val = data[math.floor(data.shape[0]/2):math.floor(3/4*data.shape[0])]
	data_test = data[math.floor(3/4*data.shape[0]):data.shape[0]]

	#data_train.info()
	#print("\n")
	#data_val.info()
	#print("\n")
	#data_test.info()
	#print("\n")

	#data normalization


	#for i in range(data.shape[1]):

		#mean = np.mean(data_train.iloc[:,i])
		#data_train.iloc[:,i] -= mean

		#std = np.std(data_train.iloc[:,i])
		#data_train.iloc[:,i] /= std
		

		
		
		#print("mean = ", np.mean(data_train.iloc[:,i]))
		#print("var = ",  np.var(data_train.iloc[:,i]),"\n")


	#print(data_train.iloc[:,4])

	#print(data_train['motor_UPDRS'])

	#hist = data_train.hist(bins=100)
	#plt.show()

	data_train_norm = copy.deepcopy(data_train)
	data_val_norm = copy.deepcopy(data_val)
	data_test_norm = copy.deepcopy(data_test)

	for i in range(data_train.shape[1]):
		
		mean = np.mean(data_train.iloc[:,i])
		data_train_norm.iloc[:,i] -= mean	
		data_val_norm.iloc[:,i] -= mean	
		data_test_norm.iloc[:,i] -= mean	

		std = np.std(data_train.iloc[:,i])
		data_train_norm.iloc[:,i] /=  std
		data_val_norm.iloc[:,i] /= std
		data_test_norm.iloc[:,i] /= std
	
	#data_train_norm=standardizeData(data_train)
	#data_test_norm=standardizeData(data_test)
	#data_val_norm=standardizeData(data_val)



	#print("- - -\n\n")
	#print(data_train_norm.min())
	#print(data_train_norm.max())
	#print("\n\n - - -")

	#hist = data_train.hist(bins=100)
	#plt.show()

	F0 = 7 #Shimmer
	y_train = data_train_norm.iloc[:,F0] #column vector
	y_test=data_test_norm.iloc[:,F0]
	y_val=data_val_norm.iloc[:,F0]

	X_train = data_train_norm.drop(columns='Shimmer')
	X_test = data_test_norm.drop(columns='Shimmer')
	X_val = data_val_norm.drop(columns='Shimmer')

	
	#lls=SolveLLS(y_train.values,X_train.values)
	gd=SolveGrad(y_train.values,X_train.values,X_val.values,y_val.values)
	#cgd=SolveConjugateGradient(y_train.values,X_train.values,X_val.values,y_val.values)
	#sgd=SolveStochasticGradient(y_train.values,X_train.values)
	#sd=SolveSteepestDescent(y_train.values,X_train.values)
	ridge=SolveRidge(y_train.values,X_train.values,X_val.values,y_val.values)
	


	#lls.run()
	#lls.plot_w('LLS')

	gd.run()
	gd.plot_w('Gradient descent')
	gd.print_result('GD')
	gd.plot_err('GD')
	gd.graphics()


	#cgd.run()
	#cgd.plot_w('Conjugate gradient descent')
	#cgd.plot_err('Conjugate Gradient')
	#cgd.graphics()
	#sgd.run()
	#sgd.plot_w('SGD')
	#sgd.plot_err('SGD')
	#sgd.graphics()
	#sd.run()
	#sd.plot_w('Steepest descent')
	#sd.plot_err('SD')
	#sd.graphics()
	#lls.plot_w("LLS")
	#gd.plot_w('Gradient Descent')
	#gd.plot_err('Gradient Descent')
	#ridge.run()
	#ridge.plot_w('Ridge regression')
	#ridge.plotRidgeError()
	#ridge.graphics()

	#cgd.plot_w('Conjugate Gradient Descent')
	#cgd.plot_err('Conjugate Gradient Descent',1,0)
	#sgd.plot_w('Stochastic Gradient Descent')
	#sd.plot_w('Steepest Descent')
	#ridge.plot_w('Ridge')

	#print(X_train.values.shape)
	#print(ridge.sol.shape)
	
	#print(y_hat.shape)
	#print("OK")

	
	#print(yhat_train.reshape(2939,))
	#print(yhat_train,y_train.values)
	#plt.plot(yhat_train,y_train.values,'.')
	#plt.title('yhat_train vs y_train')

	plt.show()

	

	print("--- END ---")
