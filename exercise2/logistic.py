import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

IMGDIM = 28*28
NDATA = 300
#NDATA = 5

arr0 = np.load("arr_0.npy")
arr1 = np.load("arr_1.npy")
arr2 = np.load("arr_2.npy")
print "arr0 shape ", arr0.shape,"arr1 shape ", arr1.shape,"arr2 shape ", arr2.shape

#test = arr2[2]
#print test
from PIL import Image
for i in range(25):
	img = Image.fromarray(arr0[i])
#	img.show()
	

print "create custom test image which is mirrored version of arr_2[2] which is an 8"
print "..displaying image"
customImg = np.flip(arr2[2])
#img = Image.fromarray(customImg)
#img.show()
#img = Image.fromarray(arr2[2])
#img.show()

#need to craft y 300x3
#let 1=[1 0 0] 4=[0 1 0] 8=[0 0 1]
print "Define Y categories: 1=[1 0 0] 4=[0 1 0] 8=[0 0 1]"
############################################## Y preprocess
Y = np.array([[0,0,0]])
for label in np.nditer(arr1):

	if label == 1:
		Y=np.append(Y,[[1,0,0]],axis=0)
	if label == 4:
		Y=np.append(Y,[[0,1,0]],axis=0)
	if label == 8:
		Y=np.append(Y,[[0,0,1]],axis=0)

Y = Y[1:NDATA+1]
#print Y
print "Y shape: ", Y.shape
###############################################

############################################### X preprocess
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
#print X[0]
print "X shape: ",X.shape
###############################################
#debug use n =1
############################################### B preprocess
#dim is length of x= nxp B= pxk y = nxk
print "initialize B with uniform values 0.1"
B = np.ones( (len(X[0]), len(Y[0])) )
B = B/10.
#let B be close to 0
print "B Shape: ",B.shape
#print B
###############################################

def getP(X,B):	
	#print "X", X
	#print "B", B
	XB = X.dot(B)
	#print "XB",XB
	#print XB
	eXB = np.exp(XB)

	#do row normalization
	#print "exB", eXB
	for i in range(len(eXB)):
		#get ith row norm
		Norm = 1.

		for j in range(len(eXB[i])):
			#if(j != i):
			Norm += eXB[i][j]
	
	
		#print Norm
		eXB[i] = eXB[i]/Norm
		
	#print "exBN", eXB
	return eXB

def getdLdB(Bold, alpha, X, Y):

	P = getP(X,Bold)
	YmP = np.subtract(Y,P)
	XtYmP = (X.T).dot(YmP)
	return  XtYmP

dB= 1
it_cnt = 1
alpha = 2e-3

print "begin descent with parameters: alpha= ",alpha," with stopping condition dB > 0.001"
print "define dB such that  dB = max| B[i,j] - Bnew[i,j] |"
while dB > 0.001:	

	Bnew = B + alpha * getdLdB(B,alpha,X,Y)
	delta = np.subtract(B,Bnew)
	delta = np.absolute(delta)
	dB = np.amax(delta)
	B = Bnew

	it_cnt = it_cnt+1
	
	print it_cnt, "iteration dB = ", dB



print "Model trained, now testing with arr2"
print "arr 2 contents [ [1],[4],[8], [8] ], last 8 is custom image appended to test array"
arr2 = np.append(arr2,[customImg],axis=0)
################################testing on arr2
Xtest = np.zeros((1,IMGDIM))
for i in range(4):

	temp = np.reshape(arr2[i],IMGDIM)
	Xtest = np.append(Xtest,[temp],axis=0)
	#stack 1
	

new_col = np.ones( (4+1,1) )
#print new_col.shape
#print new_col
#(210,1)
Xtest = np.concatenate((Xtest, new_col), axis=1)
Xtest=Xtest[1:]
Xtest = Xtest/255.
#print Xtest.shape



#print Xtest.dot(B)  
Ytest = Xtest.dot(B)
print "P for Xtest and B"
probstest = getP(Xtest,B)
print probstest
print "Model prediction for test[i], with P > 0.5"
for i in range(len(probstest)):
	if probstest[i][0] > 0.5:
		print "test digit i=",i," classified as a 1"
	if probstest[i][1] > 0.5:
		print "test digit i=",i," classified as a 4"
	if probstest[i][2] > 0.5:
		print "test digit i=",i," classified as a 8"
#test digits are Ytest= [ [1] [4] [8] ]





