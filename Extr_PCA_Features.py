#Carlyn Lee
#Extr_PCA_Features 
#projects all samples over the indices of specified samples
#Thank you Dr. Charles Lee for helping me write this script

from numpy import *
import random

def Extr_PCA_Features(xData,indInt):

	ns=len(indInt)
	x=xData[:,indInt]
	gridSize= x.shape[0]    
	theta=zeros((ns,ns))
	theta=mat(theta)
	for i in range(0,ns):
		vi=x[:,i]
		for j in range(i,ns):
			vj=x[:,j]
			theta[i,j]= float(vi.transpose() * vj /gridSize/ns)
			theta[j,i]= theta[i,j]


	lam,u=linalg.eig(theta)
	I = argsort(-1*lam)
	lam=lam[I,:]
	u=u[I,:]
	
	#normalize so that first eigenvector has unit length
	for i in range(i,ns):
		u[:,i]=u[:,i]/sum(abs(u[:,i]))


	#normalized eigenvectors
	normPhi=zeros(ns)
	phi=zeros((gridSize,ns))
	phi=mat(phi)
	for i in range(0,ns):
		for j in range(0,ns):
			phi[:,i]=phi[:,i]+u[j,i]*x[:,j]
		phi[:,i]=phi[:,i]/linalg.norm(phi[:,i])
		normPhi[i]=phi[:,i].transpose() * phi[:,i]



	#Determine sign for dominant eigenvector
	sumP1=zeros(ns)
	projMatA=zeros((xData.shape[1],ns))

	for j in range(0,ns):
		sumP1[j]=0
		for i in range(0,ns):
			sumP1[j]=sumP1[j]+sum( x[:,i].transpose() * phi[:,j]/normPhi[j])
		if sumP1[j] < 0:
			phi[:,j]=-phi[:,j]
		for i in range(0, xData.shape[1]):
			projMatA[i,j] = xData[:,i].transpose() * phi[:,j]/normPhi[j]



	return projMatA

