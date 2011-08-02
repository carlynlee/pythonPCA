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

pxraw=xraw[:,primaryTIndexAll]
nxraw=xraw[:,nIndex]

xraw=bmat('pxraw nxraw')
mean=xraw.sum(1)/xraw.shape[1]
xraw=xraw-mean

class_num=2
primaryTIndexAll=range(0,npta)
nIndex=range(0+npta,npta+nni)
Ngenes=xraw.shape[0]
Ns=xraw.shape[1]

N_class1=len(primaryTIndexAll)
divide=random.sample(primaryTIndexAll,npta)

N_class1Train=int(round(N_class1*0.08))
N_class1Val=int(round(N_class1*0.32))
N_class1Test=int(round(N_class1*0.60))
TrainIndclass1= divide[0:N_class1Train]
ValIndclass1= divide[N_class1Train:N_class1Train+N_class1Val]
TestIndclass1= divide[N_class1Train+N_class1Val:N_class1Train+N_class1Val+N_class1Test]

N_class2=len(nIndex)
divide=random.sample(nIndex,nni)

N_class2Train=int(round(N_class2*0.08))
N_class2Val=int(round(N_class2*0.32))
N_class2Test=int(round(N_class2*0.60))
TrainIndclass2= divide[0:N_class2Train]
ValIndclass2= divide[N_class2Train:N_class2Train+N_class2Val]
TestIndclass2= divide[N_class2Train+N_class2Val:N_class2Train+N_class2Val+N_class2Test]


Cls1projMatA=Extr_PCA_Features(xraw,TrainIndclass1)
Cls2projMatA=Extr_PCA_Features(xraw,TrainIndclass2)

Nfea=2
a=Cls1projMat[:,1]
a=mat(a)
a= (  a-float(a.min(1))  )/(  float(a.max(1))  - float(a.min(1))  )

b=Cls1projMat[:,1]
b=mat(b)
b= (  b-float(b.min(1))  )/(  float(b.max(1))  -  float(b.min(1))  )


LiverData=bmat('a; b')

targets=list( ones(N_class1) )
targets.extend( list(zeros(N_class2)) )
targets=array(targets)

extr_TrainInd=TrainIndclass1
extr_TrainInd.extend(TrainIndclass2)
TrainSet=LiverData[:,extr_TrainInd]
TrainTargets=targets[extr_TrainInd]

extr_ValInd=ValIndclass1
extr_ValInd.extend(ValIndclass2)
ValSet=LiverData[:,extr_ValInd]
ValTargets=targets[extr_ValInd]

extr_TestInd=TestIndclass1
extr_TestInd.extend(TestIndclass2)
TestSet=LiverData[:,extr_TestInd]
TeTargets=targets[extr_TestInd]
