import numpy as np
#import pylab as plt
import matplotlib.pyplot as plt

Nneurons = np.linspace(100,1000,10)
Nepochs = list(range(5))
Nepochs = Nepochs[1:]

#shape (nepochs, nneurons)

#LOSS
loss = np.array([[0.91757671, 0.84155528, 0.8198293,  0.80058601, 0.82703572, 0.80864556,0.80153062, 0.82368246, 0.80725366, 0.82707906],
 [0.98339571, 0.88627326, 0.84660732, 0.82507119, 0.81244886, 0.7831818,0.78683029, 0.81188418, 0.7884214,  0.80610418],
 [1.00593989, 0.90815896, 0.83628447, 0.87658624, 0.80857401, 0.8255535,0.80662978, 0.78288964, 0.78242879, 0.79473128],
 [1.08034854, 0.91099405, 0.90537819, 0.85485225, 0.87447416, 0.79538175,0.84117935, 0.80572116, 0.8245176,  0.79919004]])


vloss= np.array([[0.62190615, 0.54386517, 0.54389848, 0.54256755, 0.73138941, 0.67608955,
  0.61826208, 0.56564943, 0.72454967, 0.83441247],
 [0.59748279, 0.66089074, 0.66272009, 0.55882516, 0.61064435, 0.80877977,
  0.61154953, 0.53777542, 0.53289538, 0.85667731],
 [0.67242394, 0.5599634,  0.6090177, 0.56987762, 0.53952976, 0.5558026,
  0.60943217, 0.51261323, 0.66926688,0.5046541, ],
 [0.65516901, 0.69840379, 0.56495321, 0.5132113,  0.55781567, 0.55024433,
  0.57679587, 0.56000556, 0.54253622, 0.6752183, ]])


print(loss)
print(vloss)
"""
X1 = Nneurons
Y1 = loss[0]
Y2 = loss[1]
Y3 = loss[2]
Y4 = loss[3]

plt.scatter(X1,Y1,color='k')
plt.scatter(X1,
plt.show()
"""
def plotbyrow(loss , vloss, legtitle, axis, xname):
	color = 1
	#lt.subplot(211)
	fig, (ax0,ax1) = plt.subplots(2,1)

	ax0.set_ylim(0.5,1.1)
	ax1.set_ylim(0.5,1.1)
	ax1.set_xlabel(xname)
	ax0.set_ylabel("loss")
	ax1.set_ylabel("validation loss")
	#ig,ax = plt.subplots()
	
	for row in loss:
		#lt.plot( Nneurons, row, color, label='ass')
		
		ax0.plot(Nneurons,row,color,label=str(axis[color-1]))
		color = color + 1
	
#	hand, labl = ax0.get_legend_handles_labels()
#	print(labl)
#	labl = labl[::2]
#	plt.legend(np.unique(labl))
	
	color = 1
	#lt.subplot(212)
	for row in vloss:
		#lt.plot( Nneurons, row, color) 
		ax1.plot(Nneurons,row,color,label=str(axis[color-1]))
		color=color+1

	leg0 = ax0.legend(loc="upper left", bbox_to_anchor=[0.2, 1], ncol=4, shadow=True, title="Number of "+legtitle, fancybox=True)
	#leg1 = ax1.legend(loc="upper left", bbox_to_anchor=[0, 1],
        #         ncol=4, shadow=True, title="Number of "+title, fancybox=True)	
	#plt.legend()
	#hand, labl = ax1.get_legend_handles_labels()
	#print(labl)
	#plt.legend(np.unique(labl))
	plt.show()





plotbyrow(loss,vloss, "epochs",Nepochs, "N Neurons")
print(loss.T)
plotbyrow(loss.T,vloss.T, "neurons", Nneurons, "N Epochs")





