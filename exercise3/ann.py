import numpy as np



HIDDENLAYERS = 3#4
HIDDENNEURONS = 800#5

#back propagation

#derivative last layer dCost/dL
def getdCdZL(Y, X, Blist):
	#feed forward to last layer
	X_Lp1 = feedForward(X,Blist)
	#print "X_LP1",X_Lp1
	dydxlp1 = np.divide(Y,X_Lp1)
	#print "dydxlp1"
	#print dydxlp1
	#
	#from xlp1 determin daL(ZL)
	dCdZL = np.zeros((dydxlp1.shape[0],dydxlp1.shape[1]))
	#print "dCdZL",dCdZL
	for irow in range(len(dydxlp1)):
		
		aL = np.array([ X_Lp1[irow,:] ])
	#	print "aL", aL
	#	print "aLT", aL.T
	#	print "aLTaL", (aL.T).dot(aL)
		daL = np.diag(aL[0]) - (aL.T).dot(aL)
		#print "daL",daL
	#	print "dydxlp1", dydxlp1[irow]
		dydxrow = np.array( [dydxlp1[irow] ] )
		#print "dydxrow", dydxrow
		#print "dot", dydxrow.dot(daL)
		dCdZL[irow] = dydxrow.dot(daL)[0]
	
	#print "dCdZL", dCdZL
	#print "dCdZL here", dCdZL
	return -dCdZL

#LAYERS START AT 1
def getOutputFromLayer(X,Blist, ilayer):
	if(ilayer == 0):
		return X

	H = X 
	for iB in range(len(Blist)-1):
		H = relu( H.dot(Blist[iB]) )

		if( (ilayer-1) == iB ):
			return H
	
	H = softmax(H,Blist[-1] )
	return H

def BackProp(Y,X, Blist, alpha):
	#print "IN BACkPROP"
	Bnewlist= [1]*len(Blist)	
	#print "BLIST -1", Blist[-1]
	#Bnew = B +/- alpha dC/dB
	#b_last uses softmax
	########work on last layer
	dCdZL = getdCdZL(Y,X,Blist)
	#print "dCdZL", dCdZL
	#x input into the last layer
	XL = getOutputFromLayer(X,Blist, len(Blist)-1)
	#print "XTL",XL.T	
	dCdBL = (XL.T).dot(dCdZL)
	#print "dCdBL",dCdBL
	Bnewlist[-1] = Bnewlist[-1] - alpha[-1]*(dCdBL)
	#print "Bnew-1", Bnewlist[-1]
	#print "BLIST-1 LATER", Blist[-1]
	#print "DIFFERENCE LAST"
	#print np.subtract(Bnewlist[-1],Blist[-1])


	dCdZlp1 = dCdZL
	#print "dCdZlp1", dCdZlp1
	#with each new B back prop to find earlier derivatives
	for iB in range(2,len(Blist)+1):
		#print "iB", iB
		#print "B", Blist[-iB]
		Blp1 = Blist[-iB + 1 ]
		Xl = getOutputFromLayer(X,Blist, len(Blist)-iB)
		#print "Xl", Xl
		#print "relXl", drelu(Xl)
		
		alzl = getOutputFromLayer(X,Blist,len(Blist)-iB+1)
		dalzl = drelu(alzl)
		#print "Blp1.T", Blp1.T
		dCdZl = np.multiply(dCdZlp1.dot(Blp1.T), dalzl)
		#print "dCdZl",dCdZl
		dCdBl = (Xl.T).dot(dCdZl)
		#print "dCdBl",dCdBl
		Bnewlist[-iB] = Blist[-iB] - alpha[-iB]*(dCdBl)
		#print "Bnew ",-iB,Bnewlist[-iB]
		dCdZlp1 = dCdZl	


	return Bnewlist	
	#the rest of b use relu


	#returns new blist

#use relu for all but last layer
def feedForward(X,Blist):
	#copy X 
	H = X
	for iB in range(len(Blist)-1):
		H = relu( H.dot(Blist[iB]) )
		#print "layer ",iB, H

	#do last layer softmax
	H = softmax(H,Blist[-1])
	#print"last layer ",H
	return H	

#create list of B_(i)
def initB(HIDDENLAYERS,HIDDENNEURONS, xcol,ycol):

	B = []

	#Bfirst = np.ones((xcol, HIDDENNEURONS))
	Bfirst = np.random.normal(0,1,(xcol,HIDDENNEURONS))
	B.append(Bfirst)
	
	for i in range(HIDDENLAYERS-2):
		#B.append( np.ones((HIDDENNEURONS,HIDDENNEURONS)) ) 
		B.append( np.random.normal(0,1,(HIDDENNEURONS,HIDDENNEURONS) ))

	#Blast = np.ones((HIDDENNEURONS,ycol))
	Blast = np.random.normal(0,1,(HIDDENNEURONS,ycol))
	B.append(Blast)

	#print "initial B", B
	return B


def relu(x):
	#hidden elementwise activation
	return np.maximum(x, 0)

def drelu(x):
	x[x<0] = 0
	x[x>=0] = 1
	return x
	

def softmax(X,B):
	XB = X.dot(B)
	eXB = np.exp(XB)
	#print "pre norm ", eXB
	for irow in range(len(eXB)):
		Norm = sum(eXB[irow,:])
	#	print "Norm", Norm
		eXB[irow] = np.divide(eXB[irow,:],Norm)
		
		
	#print "post norm exb", eXB
	return eXB
"""
X = np.array([[0.,0.,1.],[0.,1.,1.],[1.,0.,1.],[1.,1.,1.]])
#Y = np.array([[0.,1.],[1.,0.],[1.,0.],[0.,1.]])
Y = np.array([[0.],[1.],[1.],[0.]])
#normalize X to small
X = np.divide(X,100.)
print "normX", X
"""
import sys
#np.set_printoptions(threshold=sys.maxsize)

IMGDIM = 28*28
NDATA = 300
#NDATA = 5

arr0 = np.load("arr_0.npy")
arr1 = np.load("arr_1.npy")
arr2 = np.load("arr_2.npy")
print "arr0 shape ", arr0.shape,"arr1 shape ", arr1.shape,"arr2 shape ", arr2.shape

#print "Xshape", X.shape
#print "Yshape", Y.shape

print "Define Y categories: 1=[1 0 0] 4=[0 1 0] 8=[0 0 1]"
############################################## Y preprocess
Y = np.array([[0.,0.,0.]])
for label in np.nditer(arr1):

	if label == 1:
		Y=np.append(Y,[[1.,0.,0.]],axis=0)
	if label == 4:
		Y=np.append(Y,[[0.,1.,0.]],axis=0)
	if label == 8:
		Y=np.append(Y,[[0.,0.,1.]],axis=0)

Y = Y[1:NDATA+1]
#print Y
print "Y shape: ", Y.shape
############################################# X preprocess
print "flatted 28x28 x matrices to 1D of length 768 + constant 1"
X = np.zeros((1,IMGDIM))
for i in range(NDATA):

	temp = np.reshape(arr0[i],IMGDIM)
	X = np.append(X,[temp],axis=0)
	#stack 1
	

new_col = np.ones( (NDATA+1,1) )
#print new_col.shape
#print new_col
#(210,1)
X = np.concatenate((X, new_col), axis=1)
#print X
X=X[1:NDATA+1]
#normalize to 1
X = X/255.
X = X/1000.
#print X[0]
print "X shape: ",X.shape
###########################################

Blist = initB(HIDDENLAYERS, HIDDENNEURONS, len(X[0]),3)
print Blist[0],"shape", Blist[0].shape
print Blist[1],"shape", Blist[1].shape
print Blist[2],"shape", Blist[2].shape

#print Blist[3]
#print Blist
#print B.shape
"""
Output = feedForward(X,Blist)
print Output

print "test get iLayer X0 is input to 1st layer"
print "Get layer 0 output: (X0)"
print X

print "Get layer 1 output: (al1Zl1)=X1"
print getOutputFromLayer(X,Blist,1);

print "Get layer 2 ouput: (al2Zl2)=X3"
print getOutputFromLayer(X,Blist,2);

print "Get layer 3: (al3Zl3)=X3"
print getOutputFromLayer(X,Blist,3);

print "Get layer 4 (al4zl4)=X4:"
print getOutputFromLayer(X,Blist,4);
"""

def getdBs(Blist,Bnew,HIDDENLAYERS):
	#print"lists here",Blist
	#print "list here again",Bnew
	dB = np.zeros(HIDDENLAYERS)
	#print "dB here", dB
	for i in range(len(Blist)):
		delta = np.subtract(Blist[i],Bnew[i])
		#print "delta ",i,delta
		delta = np.absolute(delta)
		dB[i] = np.amax(delta)
		
	#print "dB there", dB
	return dB
#print "diag tests"
#print "aLT aL"
#print (Output.T).dot(Output)

#getdCdZL(Y, X, Blist)
#alpha = [2e-3,2e-3,2e-3,2e-3]
#alpha = [2e-3,2e-4,2e-5,2e-6]
#alpha = [1,2,3,4]
#alpha = [1.,1.,1.,1.]
alpha = [2e-4]*HIDDENLAYERS
#dB= 1
dB = np.ones(HIDDENLAYERS)
print dB
it_cnt = 1
dBmax = 1

print "Run with no batch training"
print "begin descent with parameters: alpha= ",alpha," with stopping condition dB > 0.001"
print "define dB such that  dB = max| B[i,j] - Bnew[i,j] |"
while dBmax > 0.0001:	

	Bnew =  BackProp(Y,X, Blist, alpha)
	dB = getdBs(Blist,Bnew,HIDDENLAYERS)
	#print "dB",dB
	dBmax = np.amax(dB)
	#print "dBmax",dBmax
	Blist = Bnew
	#print "Bnew ", Bnew
	it_cnt = it_cnt+1
	
	print it_cnt, "iteration dB = ", dBmax
	

print "TESTING ON Xtest"
Xtest = np.zeros((1,IMGDIM))
for i in range(3):

	temp = np.reshape(arr2[i],IMGDIM)
	Xtest = np.append(Xtest,[temp],axis=0)
	#stack 1
	

new_col = np.ones( (3+1,1) )
#print new_col.shape
#print new_col
#(210,1)
Xtest = np.concatenate((Xtest, new_col), axis=1)
Xtest=Xtest[1:]
Xtest = Xtest/255.
Xtest = Xtest/1000.

print Xtest.shape
print feedForward(Xtest,Blist)
#print Y


print "run with batch training"
"""
#batch training and data multplication
#BatchAll = np.array([0.,0.,1.,0.,1.])
BatchAll = np.array([0.,0.,1.,0.])
#BatchTag = np.array([0.,1.])
#print BatchAll, BatchAll.shape
#print BatchTag, BatchTag.shape
for x in range(300):
	#BatchAll = np.vstack( (BatchAll, np.array([0.,0.,1.,0.,1.])) )
	#BatchAll = np.vstack( (BatchAll, np.array([0.,1.,1.,1.,0.])) )
	#BatchAll = np.vstack( (BatchAll, np.array([1.,0.,1.,1.,0.])) )
	#BatchAll = np.vstack( (BatchAll, np.array([1.,1.,1.,0.,1.])) )
	BatchAll = np.vstack( (BatchAll, np.array([0.,0.,1.,0.])) )
	BatchAll = np.vstack( (BatchAll, np.array([0.,1.,1.,1.])) )
	BatchAll = np.vstack( (BatchAll, np.array([1.,0.,1.,1.])) )
	BatchAll = np.vstack( (BatchAll, np.array([1.,1.,1.,0.])) )


#print BatchAll
#print BatchTag

#print BatchAll
np.random.shuffle( BatchAll )
print "shuffling"
#print BatchAll
BatchTag = BatchAll[:,3:] 
#print BatchTag
BatchAll = BatchAll[:,:-1]
print "all shape ",BatchAll.shape,BatchAll


n = 50
BatchX = [BatchAll[i * n:(i + 1) * n] for i in range((len(BatchAll) + n - 1) // n )]
BatchY = [BatchTag[i * n:(i + 1) * n] for i in range((len(BatchTag) + n - 1) // n )]

#print BatchX
#print BatchY

#Batchbingsus
#print BatchAll
#alpha = [1.,1.,1.,1.]
alpha = [6e-6]*HIDDENLAYERS
#dB= 1
dB = np.ones(HIDDENLAYERS)
print dB
it_cnt = 1
dBmax = 1
print "Batch training "
print "begin descent with parameters: alpha= ",alpha," with stopping condition dB > 0.001"
print "define dB such that  dB = max| B[i,j] - Bnew[i,j] |"
while dBmax > 0.0001:
	
	for iBatch in range(len(BatchX )):
		print "Training on Batch ", iBatch
		dBMax = 1
		#it_cnt = 1
		thisbatch = BatchX[iBatch]
		thistag = BatchY[iBatch]

	#while dBmax > 0.0001:	

		Bnew =  BackProp(thistag,thisbatch, Blist, alpha)
		dB = getdBs(Blist,Bnew,HIDDENLAYERS)
	#print "dB",dB
		dBmax = np.amax(dB)
	#print "dBmax",dBmax
		Blist = Bnew
	#print "Bnew ", Bnew
	it_cnt = it_cnt+1
	
	print it_cnt, "iteration dB = ", dBmax
	

print "testing model"
print "X ",X
print feedForward(X,Blist)

	
"""


