import csv
def printmat(matrix):
	for row in matrix:
		print(row)
def column(matrix, i):
	col = i[0]
	return [row[col] for row in matrix]

def transpose(matrix):
	return zip(*matrix) 
		
def prodElement(arow,bcol):# this is row * a column summed 
	return sum(map(lambda x,y:x*y,arow,bcol))
	
def prodRow(A, B, bcol):#all rows of a times specified col of B #returns a row of a matrix (C)
	return [ prodElement(A[irow],column(B,[bcol])) for irow in range(len(A)) ]
		
def mult(A,B):
	return transpose([ prodRow(A,B,bcol) for bcol in range(len(B[0])) ])  	

def scalar(alpha, A):
	return  [list(map((alpha).__mul__, arow)) for arow in A]

def add(A, B):
	return [[A[i][j] + B[i][j]  for j in range(len(A[0]))] for i in range(len(A))] 

def sub(A, B):#A-B
	return [[A[i][j] - B[i][j]  for j in range(len(A[0]))] for i in range(len(A))]

def maxDelta(Bold,Bnew):
	deltaB = sub(Bold,Bnew)
	deltaB = [[ abs(deltaB[i][j]) for j in range(len(deltaB[0])) ] for i in range(len(deltaB))]
	#print "deltaB", deltaB
	#print deltaB 
	#maxdB = [ max(x) for x in deltaB ]
	#deltaB =map(abs,deltaB)	
	return max(map(max, deltaB))
	#return max(maxdB)
	#return


def Bp1(B,alpha,Xt,X,Cinv,Y):
	#let Cinv b Identity
	#XtC = mult(Xt,Cinv)
	XtC = Xt
	a2XtC = scalar(alpha*2,XtC)
	XB = mult(X,B)
	YmXB = sub(Y,XB)

	return add(B, mult(a2XtC,YmXB ))
	

with open('PRSA_Data_Shunyi_20130301-20170228.csv') as csvfile:
	list_1 = csv.reader(csvfile, delimiter=',', quotechar='|') #reads in raw data into matrix
	list_2 = [x for x in list_1 if 'NA' not in x] 	# deletes entries which contain NA



#create a subset value based matrix with labels extracted
#list_3 = list_2[1:]
list_3 = list_2[1:]
	
#print list_3B,alpha,Xt,X,Cinv,Y
#for row in list_3:
#	print(', '.join(row))

# labels/vars of interest for output/input
xElements = ["month", "hour", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"] 
yElements = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]



#get data labels
labels = list_2[0]
#process labels (extract weird " ")
labels = [ x[1:-1] for x in labels ]
#print labels


#we extract columnwise -> each list in the 2d list is the respective elements of that variable
#this means columnwise will make x and y be transposed
#Xt, Yt=[],[]
Xt = [ column(list_3, [ilabel for ilabel in range(len(labels)) if labels[ilabel] == element] ) for element in xElements]
#x goes by nx(P+1) add a constant column
#const = [ 1. for x in Xt[0] ]
#Xt.append(const)
X = transpose(Xt)
#convert to fp
Xt = [ map(float, a) for a in Xt ]
X = [ map(float, a) for a in X ]
#print xElements
#print "xt " 
#printmat(Xt)

#print "x "
#printmat(X)

Yt = [ column(list_3, [ilabel for ilabel in range(len(labels)) if labels[ilabel] == element] ) for element in yElements]
Y = transpose(Yt)
Y = [ map(float, a) for a in Y ]
#print yELements
#print Yt
#print "y "
#printmat(Y)

#print data information
print "Shunyi Data -- part1 -- using only python standard libraries"
print "Using X input labels:"
print xElements
print "Using Y ouput labels:"
print yElements
print "Data contains ", len(Y), " valid entries"


#create first B matrix intitialized to all 1s of proper dimension
# y(nxk) = X(nxp)B(pxk) + E(nxk)
B = [ len(Y[0])*[1] for colx in range(len(X[0])) ]
Bcopy = B
#B = []
#print "B "
print "initial B matrix: "
printmat(B)

alpha = 3e-12
print "learning paramter alpha ", alpha


""" testing 
Bnew = Bp1(B,alpha,Xt,X,1,Y)
#printmat(Bnew)
Bnewnew = Bp1(Bnew,alpha,Xt,X,1,Y)
printmat(Bnewnew)
"""

Cinv = 1
dB = 1
it_cnt =1 
Bnew = []
print "Stopping criteria defined by dB < max( |Bnew[i,j] - Bold[i,j]| )"
print "Stopping point set to: .001"
while dB > .001:
	Bnew = Bp1(Bcopy,alpha,Xt,X,Cinv,Y)
	dB = maxDelta(Bcopy, Bnew)
	Bcopy = Bnew
	print it_cnt, "iteration dB = ", dB
	it_cnt = it_cnt+1
	#if(it_cnt >= 6):
	#	break
	
print "final Bnew:" 
printmat(Bnew)


print " "
print "start numpy checks to verify output"

import numpy 
x1 = numpy.array(X)
y1 = numpy.array(Y)
B1 = numpy.array(B)
print "initial B matrix"
print B1
#print x1
#print y1
#print B1


"""
xb = numpy.dot(x1,B1)
ymxb = numpy.subtract(y1,xb)
Bnew1 = numpy.add(B1, numpy.dot(alpha*2.*x1.T, ymxb))
#print Bnew1
xb = numpy.dot(x1,Bnew1)
ymxb = numpy.subtract(y1,xb)
Bnew2 = numpy.add(Bnew1, numpy.dot(alpha*2.*x1.T,ymxb))
print Bnew2
"""
dB1 = 1
it_cnt1 = 1
while dB1 > .001:
	xb = numpy.dot(x1,B1)
	ymxb = numpy.subtract(y1,xb)
	a2xt = (alpha*2.)*x1.T
	Bnew1 = numpy.add(B1, numpy.dot(a2xt, ymxb) )
	#dB1 = maxDelta(B1, Bnew1)
	delta = numpy.subtract(B1,Bnew1)
	delta = numpy.absolute(delta)
	dB1 = numpy.amax(delta)
	B1 = Bnew1
	it_cnt1 = it_cnt1+1
	
	print it_cnt1, "iteration dB = ", dB1
	
print "final Bnew from numpy:" 
print Bnew1


