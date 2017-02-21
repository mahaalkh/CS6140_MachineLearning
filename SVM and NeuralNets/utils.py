import scipy.io 
import numpy as np 

def getData(fileName): 
	"""
	"""
	Data = scipy.io.loadmat(fileName)
	return Data

def sortData(fileName): 
	"""
	"""
	Data = getData(fileName)
	# ------------- training data ---------
	Y_trn = Data['Y_trn']
	X_trn = Data['X_trn']
	#------------ test data ----------
	Y_tst = Data['Y_tst']
	X_tst = Data['X_tst']
	# print Data
	return [X_trn, Y_trn, X_tst, Y_tst]

def dataToVectors(fileName): 
	"""
	"""	
	X_trn, Y_trn, X_tst, Y_tst = sortData(fileName)

	X_trn_V = list(X_trn)

	Y_trn_V = list(Y_trn)

	X_tst_V = list(X_tst)

	Y_tst_V = list(Y_tst)

	x_trn = []
	y_trn = []
	x_tst = []
	y_tst = []

	for i in range(len(X_trn_V)): 
		x =  list(X_trn_V[i])
		x_trn.append(x)

	for i in range(len(Y_trn_V)):
		y  = list(Y_trn_V[i])[0]
		y_trn.append(y)

	for i in range(len(X_tst_V)): 
		x =  list(X_tst_V[i])
		x_tst.append(x)

	for i in range(len(Y_tst_V)):
		y  = list(Y_tst_V[i])[0]
		y_tst.append(y)

	return [np.array(x_trn), np.array(y_trn), np.array(x_tst), np.array(y_tst)]


def truncate(X, s): 
	return X[0 : s, :]

def truncateData(X, i, s): 
	X_t = np.transpose(X)[0].tolist()
	# print "in truncate data"
	# print X_t
	X_hold_out = X_t[i:s]
	X_rest = X_t[0:i]
	X_rest1 = X_t[s:]
	X_rest.extend(X_rest1)

	X_hold_out2 = np.array([X_hold_out], float)
	X_hold_in = np.array([X_rest], float)


	X_hold_out3 = np.transpose(X_hold_out2)
	X_hold_in2 = np.transpose(X_hold_in)


	return [X_hold_out3, X_hold_in2]


# l = dataToVectors("HW2_Data/data.mat")
# for i in l: 
# 	print i 