import scipy as sc 
import scipy.io 
import numpy as np 
from scipy.linalg import block_diag
from utils import dataToVectors, truncate
from scipy.optimize import minimize

##### _____________________ ######
## N : number of data points
## K : number of classes 
## q : number of parameters for each of the classes 
## beta : [beta_1 ; beta_2  ; ... ; beta_k_1 ]   (column vector)
## beta_1 : parameters for class 1
## Y : [y_1, ..., y_K_1]
## y_1 = [y_1^1; ...; y_1^N]
## ====================================

def prob_x_k_1(beta_all, q, x, k): 
	"""
	calculates x transpose beta_k for specific class k
	x : is a data point 
	"""
	k = k[0]
	
	beta_k_T = beta_all[k*q: (k+1)*q] 


	beta_k_T_x = np.dot(beta_k_T, x)

	result = beta_k_T_x

	return result


def prob_x_k(beta_all, q, x, k): 
	"""
	calculates x transpose beta_k for specific class k
	x : is a data point 
	"""
	
	beta_k_T = beta_all[k*q: (k+1)*q] 


	beta_k_T_x = np.dot(beta_k_T, x)

	result = beta_k_T_x

	return result

def prob_x(beta_all, q, x, K):
	"""
	calculates the natural log of 1 + sum 
	of the exponent of the various classes except for class k - 1
	"""
	probs = 0 
	for k in range(K - 1): 
		prob_x_k_i = np.exp(prob_x_k(beta_all, q, x, k))
		probs = probs + (prob_x_k_i)

	sum_prob = 1 + probs
	result = np.log(sum_prob)

	return result

def prob_all_x(beta_all, q, X_frd, Y_frd, K): 
	"""
	calculate the log likelihood ; 
	note the special case of the last class 
	"""

	N = len(Y_frd)
	tot = 0 
	for i in range(N): 
		x_i = X_frd[i]
		y_i = Y_frd[i][0] 
		u = prob_x(beta_all, q, x_i, K)

		if y_i == K - 1: 
			v = 0
		else: 
			v = prob_x_k(beta_all, q, x_i, y_i)
		tot = v - u + tot

	return tot

def get_maximizing_beta(q, X_frd, Y_frd, K):
	"""
	gets the maximizing beta for the log likelihood function 
	"""
	b =  np.zeros((1, 4))
	func = lambda beta: (-1 * prob_all_x(beta, q, X_frd, Y_frd, K))
	res = minimize(func, b)

	return res.x


def get_maximizing_k(beta, q, x, K):
	res = minimize(lambda k: -1 * prob_x_k_1(beta, q, x, k), 0)

	if res.fun > 0:	
		res.x[0] = K - 1

	return res.x[0]

def getPredicted_k_and_error(beta_opt, q, X_frd, Y_frd, K): 
	"""
	"""
	N = len(X_frd)
	total_incorrect = 0 
	for n in range(N): 
		predicted_k = get_maximizing_k(beta_opt, q, X_frd[n], K)
		actual_k = Y_frd[n][0]
		if predicted_k != actual_k: 
			total_incorrect = total_incorrect + 1
	
	return float(total_incorrect) / float(N)


def run(fileName, q, K): 
	X_trn, Y_trn, X_tst, Y_tst = dataToVectors(fileName)

	
	beta_opt =  get_maximizing_beta(q, X_trn, Y_trn, K)
	print "optimal w = {}".format(beta_opt)
	print "training error = {}".format(getPredicted_k_and_error(beta_opt, q, X_trn, Y_trn, K))
	print "test error = {}".format(getPredicted_k_and_error(beta_opt, q, X_tst, Y_tst, K))

run("HW1_Data/logistic_regression.mat", 2, 3)