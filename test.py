#Carlyn Lee
#test.py
#wrapper to test Extr_PCA_Feature. Projects over randomly selected negative and positive samples from 'imputed-liver.txt' and generates two dimensional data using primary components
#thank you to Dr. Charles Lee for helping me write this script and for the POD

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


nIndex=range(115,191)
nni=len(nIndex)

primaryTIndexAll=[3]
primaryTIndexAll.extend(range(8,112))
npta=len(primaryTIndexAll)


infile = open('imputed-liver.txt', 'r')
xraw=[]
for line in infile:
	line.strip()
	xraw.append(line.split())

infile.close()

for i in xrange(0,len(xraw)):
	for j in xrange(0,len(xraw[0])):
		xraw[i][j]=float(xraw[i][j])

xraw=mat(xraw)

#extract positives and negatives from data set
pxraw=xraw[:,primaryTIndexAll]
nxraw=xraw[:,nIndex]
xraw=bmat('pxraw nxraw')

#subtract the means of each row to eliminate outliers
mean=xraw.sum(1)/xraw.shape[1]
xraw=xraw-mean

class_num=2
primaryTIndexAll=range(0,npta)
nIndex=range(0+npta,npta+nni)
Ngenes=xraw.shape[0]
Ns=xraw.shape[1]

N_class1=len(primaryTIndexAll)
#divide=random.sample(primaryTIndexAll,npta)

#separate positve and negative sets into train, validation, test sets

TrainIndclass1=primaryTIndexAll[0:70] #2/3 of all primary tumor
ValIndclass1= primaryTIndexAll[70:105]
TestIndclass1=primaryTIndexAll[70:105]
N_class1Train= size(TrainIndclass1)
N_class1Val=size(ValIndclass1)
N_class1Test=size(TestIndclass1)



N_class2=len(nIndex)
divide=random.sample(nIndex,nni)

TrainIndclass2= nIndex[0:49] # 2/3 of the normal data 
ValIndclass2= nIndex[49:76]
TestIndclass2= nIndex[49:76]

N_class2Train=size(TrainIndclass2)
N_class2Val=size(ValIndclass2)
N_class2Test=size(TestIndclass2)


#POD
Cls1projMatA=Extr_PCA_Features(xraw,TrainIndclass1)
Cls2projMatA=Extr_PCA_Features(xraw,TrainIndclass2)

#using only principal components (normalized between 0 and 1)
Nfea=2
a=Cls1projMatA[:,1]
a=mat(a)
a= (  a-float(a.min(1))  )/(  float(a.max(1))  - float(a.min(1))  )

b=Cls2projMatA[:,1]
b=mat(b)
b= (  b-float(b.min(1))  )/(  float(b.max(1))  -  float(b.min(1))  )


LiverData=bmat('a; b')

targets=list( ones(N_class1) )
targets.extend( list(zeros(N_class2)) )
targets=array(targets)

#construct train set and targets
extr_TrainInd=TrainIndclass1
extr_TrainInd.extend(TrainIndclass2)
TrainSet=LiverData[:,extr_TrainInd]
TrainTargets=targets[extr_TrainInd]

#construct validation set and targets
extr_ValInd=ValIndclass1
extr_ValInd.extend(ValIndclass2)
ValSet=LiverData[:,extr_ValInd]
ValTargets=targets[extr_ValInd]

#construct test set and targets
extr_TestInd=TestIndclass1
extr_TestInd.extend(TestIndclass2)
TestSet=LiverData[:,extr_TestInd]
TeTargets=targets[extr_TestInd]
