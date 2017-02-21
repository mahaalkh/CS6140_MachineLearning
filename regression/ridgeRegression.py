import numpy as np 
from utils import dataToVectors, truncateData, truncate
from funcs_linear_ridge import getW, getError
from scipy.optimize import minimize
#--------------------------------------	


def getError_ridge(lam, n, data): 
	"""
	"""
	X , Y = data

	X_hold_out, X_hold_in = X
	Y_hold_out, Y_hold_in = Y

	w = getW(X_hold_in, Y_hold_in, n, lamda = lam)
	hold_out_error = getError(X_hold_out, Y_hold_out, w, n)
	return hold_out_error

def getAverageHoldOut(lam, n, N, k, data): 


	# optimize over lamda for given k and n
	error = []
	for l in range(k): 
		i = l*(N//k)
		s = ((l + 1)* (N//k))
		d = createFold(i, s, data)
		error_l = getError_ridge(lam, n, d)
		error.append(error_l)

	average = float(sum(error) / len(error))
	return average


def lamdaThatMinimizes(n, N, k, data): 
	"""
	"""
	res = minimize(lambda lam: getAverageHoldOut(lam, n, N, k, data), 0)
	## (lamda, test error)
	return (res.x[0], res.fun)

def getW_ridge_and_errors(lam, n, data): 
	"""
	"""
	X_trn, Y_trn , X_tst, Y_tst = data
	w = getW(X_trn, Y_trn, n, lamda = lam)
	training_error = getError(X_trn, Y_trn, w, n)
	test_error = getError(X_tst, Y_tst, w, n)
	return (w, training_error, test_error) 

def createFold(l, k, data): 
	""" 
	"""
	X_trn, Y_trn, _ , _ = data
	return [truncateData(X_trn, l, k), truncateData(Y_trn, l, k)]


def k_fold(data): 
	X_trn, Y_trn, X_tst, Y_tst = data
	N = len(X_trn)
	for n in [2, 5, 10, 20]:
		for k in [2, 5, 10, N]:
			lam, min_error = lamdaThatMinimizes(n, N, k, data)
			w, training_error, test_error =  getW_ridge_and_errors(lam, n, data)
			print "n = {}".format(n)
			print  "\t k = {}".format(k)
			print "\t \t lambda = {}".format(lam)
			print "\t \t hold out error = {}".format(min_error)
			print "\t \t w = {}".format(np.transpose(w))
			print "\t \t training error = {}".format(training_error)
			print "\t \t test error = {}".format(test_error)

			


def run(fileName): 
	
	data = dataToVectors(fileName)
	k_fold(data)


run("HW1_Data/linear_regression.mat")
