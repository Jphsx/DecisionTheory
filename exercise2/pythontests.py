

print "T- test: "
from scipy import stats
import numpy as np


G = [np.random.normal(loc=2,scale=5) for i in range(25) ]
mean = np.mean(G)
var = np.var(G)
print "Gauss",G
print "mean",mean
print "var",var
T = (mean - 2)/np.sqrt(var/25)
print "T",T
pval = stats.t.cdf(x=T,df=2)
print "pval",pval


print ""
print "Linear model:"
Et = [np.random.normal(loc=0,scale=1) for i in range(100) ]
#print Et
Et = np.array([Et])
E = Et.T
#print E
print "E noise is random normal, shape: ",E.shape


X = np.random.random((100, 3))
const = np.ones((100,1))
X = np.concatenate([X,const], axis=1)
#print X
print "X is random uniform with 1s stacked, shape: ", X.shape
B = np.array([[1],[2],[3],[4]])
print "B",B

Y = X.dot(B) + E
#print Y
print "Y = XB+E, shape: ", Y.shape

print "Bhat = (XtX)^-1 XtY"
XtX = (X.T).dot(X)
XtY = (X.T).dot(Y)
XtXinv = np.linalg.inv(XtX)
Bhat = XtXinv.dot(XtY)
print "Bhat = ",Bhat

 
