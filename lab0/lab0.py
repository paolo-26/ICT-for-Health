import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(42)
matplotlib.rc('text', usetex = True)

class SolveMinProbl(object):
	def __init__(self,y,A):
		self.matr=A
		self.Np=y.shape[0]
		self.Nf=A.shape[1]
		self.vect=y
		self.sol=np.zeros((self.Nf,1),dtype=float)
		return

	def plot_w(self,title):
		w=self.sol
		n=np.arange(self.Nf)
		plt.figure()
		plt.plot(n,w)
		plt.xlabel('$n$')
		plt.ylabel('$w(n)$')
		plt.xticks(ticks=range(4),labels=range(1,5))
		plt.title(title)
		plt.grid(which='both')
		plt.show()
		return

	def print_result(self,title):
		print('%s:' %title)
		print('the optimum weight vector is:')
		print(self.sol,"\n")
		return

	def plot_err(self,title='Square_error',logy=0,logx=0):
		err=self.err
		plt.figure()
		if (logy==0) & (logx==0):
			plt.plot(err[:,0],err[:,1])
		if (logy==1) & (logx==0):
			plt.semilogy(err[:,0],err[:,1])
		if (logy==0) & (logx==1):
			plt.semilogx(err[:,0],err[:,1])
		if (logy==1) & (logx==1):
			plt.loglog(err[:,0],err[:,1])
		plt.xlabel('$n$')
		plt.ylabel('$e(n)$')
		plt.title(title)
		plt.margins(0.01,0.1)
		plt.grid(which='both')
		plt.show()
		return


class SolveLLS(SolveMinProbl):
	def run(self):
		A=self.matr
		y=self.vect
		w=np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),y)
		self.sol=w
		self.min=np.linalg.norm(np.dot(A,w)-y)

class SolveGrad(SolveMinProbl):
	def run(self,gamma=1e-3,Nit=500):
		self.err=np.zeros((Nit,2),dtype=float)
		A=self.matr
		y=self.vect
		w=np.random.rand(self.Nf,1)

		for it in range(Nit):
			grad=2*np.dot(A.T,(np.dot(A,w)-y))
			w=w-gamma*grad
			self.err[it,0]=it
			self.err[it,1]=np.linalg.norm(np.dot(A,w)-y)

		self.sol=w
		self.min=self.err[it,1]

class SolveSteepestDescent(SolveMinProbl):
	def run(self,Nit=500):
		self.err=np.zeros((Nit,2),dtype=float)
		A=self.matr
		y=self.vect
		w=np.random.rand(self.Nf,1)

		for it in range(Nit):        
			H=4*np.dot(A.T,A)
			grad=2*np.dot(A.T,(np.dot(A,w)-y))
			w=w-(((np.linalg.norm(grad))**2)/(np.dot(np.dot(grad.T,H),grad)))*grad

			self.err[it,0]=it
			self.err[it,1]=np.linalg.norm(np.dot(A,w)-y)

		self.sol=w
		self.min=self.err[it,1]


class SolveStochasticGradient(SolveMinProbl):
	def run(self,Nit=500,gamma=1e-3):
		self.err=np.zeros((Nit,2),dtype=float)
		A=self.matr
		y=self.vect
		w=np.random.rand(self.Nf,1)

		for it in range(Nit):
			for i in range(self.Np):
				grad_i = 2*(np.dot(A[i,:].T,w)-y[i])*A[i,:].reshape(len(A[i,:]),1)
				w=w-gamma*grad_i

			self.err[it,0]=it
			self.err[it,1]=np.linalg.norm(np.dot(A,w)-y)
		self.sol=w
		self.min=self.err[it,1]


class SolveMiniBatches(SolveMinProbl):
	def run(self,L,K,Nit=500,gamma=1e-3):
		self.err=np.zeros((Nit,2),dtype=float)
		A=self.matr
		y=self.vect
		w=np.random.rand(self.Nf,1)
		self.K = int(K)

		for it in range(Nit):
			for i in range(K):
				grad_b = 2*(np.dot(np.dot(A[i*L:i*L+L,:].T,A[i*L:i*L+L,:]),w) - np.dot(A[i*L:i*L+L,:].T,y[i*L:i*L+L]))
				w = w - gamma * grad_b

			self.err[it,0]=it
			self.err[it,1]=np.linalg.norm(np.dot(A,w)-y)

		self.sol=w
		self.min=self.err[it,1]

class SolveConjugateGradient(SolveMinProbl):
	def run(self,Nit=500):
		self.err=np.zeros((Nit,2),dtype=float)
		A=self.matr
		y=self.vect
		w = np.zeros((self.Nf,1),dtype=float)
		Q = np.dot(A.T,A)
		b = np.dot(A.T,y)
		d = b
		g = -b

		for it in range(Nit):
			if (np.dot(np.dot(d.T,Q),d) == 0):
				break
			alpha = -((np.dot(d.T,g))/(np.dot(np.dot(d.T,Q),d)))
			if np.isnan(alpha):
				break
			w = w + alpha * d
			g = np.dot(Q,w) - b
			beta = np.dot(np.dot(g.T,Q),d)/np.dot(np.dot(d.T,Q),d)
			d = -g +beta*d
			self.err[it,0]=it
			self.err[it,1]=np.linalg.norm(np.dot(A,w)-y)

		self.sol=w
		self.min=self.err[it,1]
		


###################
####### MAIN ######
###################

if __name__=="__main__":


	#data = pd.read_csv("parkinsons_updrs.csv")

	Np = 80 
	Nf = 4 # number of features
	K = 8 # number of mini-batches
	L = int(Np/K) # size of mini-batches
	
	Nit = 500 #number of iterations
	A = np.random.randn(Np,Nf) #matrix
	y = np.random.randn(Np,1)


	logx = 0
	logy = 1


	#### LLS ####
	lls=SolveLLS(y,A)
	lls.run()
	lls.print_result('LLS')
	lls.plot_w('LLS')


	#### Gradient Descent####
	gamma=1e-2
	gd=SolveGrad(y,A)
	gd.run()
	gd.print_result('Gradient descent')
	gd.plot_w('Gradient descent')
	gd.plot_err('Gradient descent: square error',logy,logx)

	#### Steepest Descent ####
	ss=SolveSteepestDescent(y,A)
	ss.run()
	ss.print_result('Steepest descent')
	ss.plot_w('Steepest descent')
	ss.plot_err('Steepest descent: square error',logy,logx)

	### Stochastic Gradient Descent ###
	sgd=SolveStochasticGradient(y,A)
	sgd.run()
	sgd.print_result('Stochastic gradient descent')
	sgd.plot_w('Stochastic gradient descent')
	sgd.plot_err('Stochastic gradient descent: square error',logy,logx)	

	### Stochastic Gradient Descent with mini-batches###
	sgdmb=SolveMiniBatches(y,A)
	sgdmb.run(L,K)
	sgdmb.print_result('Stochastic gradient descent with mini-batches')
	sgdmb.plot_w('Stochastic gradient descent with mini-batches')
	sgdmb.plot_err('Stochastic gradient descent with mini-batches: square error',logy,logx)

	### Conjugate gradient method
	cgm=SolveConjugateGradient(y,A)
	cgm.run()
	cgm.print_result('Conjugate gradient')
	cgm.plot_w('Conjugate gradient')
	cgm.plot_err('Conjugate gradient: square error',logy,logx)
