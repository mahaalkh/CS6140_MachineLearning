import scipy as sc 
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

	return [X_trn, Y_trn, X_tst, Y_tst]

def dataToVectors(fileName): 
	"""
	"""
	X_trn, Y_trn, X_tst, Y_tst = sortData(fileName)

	X_trn_V = np.array(X_trn, float)

	Y_trn_V = np.array(Y_trn, float)

	X_tst_V = np.array(X_tst, float)

	Y_tst_V = np.array(Y_tst, float)

	return [X_trn_V, Y_trn_V, X_tst_V, Y_tst_V]


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

