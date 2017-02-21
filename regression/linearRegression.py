import numpy as np 
from utils import dataToVectors, truncateData, truncate
from funcs_linear_ridge import getW, getError
#-------------------------------------- START OF LINEAR REGRESSION CALCULATION OF w ----------------------------

# ######## USED EVERYWHERE
# def phiX(X, n):
# 	"""
# 	computes Phi(x)
# 	X_trn: the x_i's of the  data 
# 	n: the degree of the polynomial 
# 	:return: a matrix of size N * n 
# 		where N is the number of data points 
# 		and n is the degree of the polynomial  
# 	"""
# 	N = len(X)
# 	X_t = X.transpose()[0]

# 	phiX = []
# 	previousArray = []
# 	for i in range(n + 1): 
# 		if i == 0: 
# 			previousArray = np.array([1] * N, float).transpose()
# 		else: 
# 			previousArray = previousArray * X_t
# 		phiX.append(previousArray.tolist())

# 	phiX = np.array(phiX, float)

# 	return phiX.transpose()

# def getW(X_trn, Y_trn, n, lamda = 0):
# 	"""
# 	computes w 
# 	X_trn: the x_i's of the training data 
# 	Y_trn: the y_i's of the training data 
# 	n: the degree of the polynomial 
# 	:return: a vector of size n * 1
# 	"""
# 	A = phiX(X_trn, n)
	
# 	A_t = A.transpose()

# 	# matrix multlipication 
# 	A_t_A = np.dot(A_t, A)

# 	#----------------------------
# 	m = len(A_t_A)
# 	I = np.identity(m) 
# 	lamda_I = lamda * I 
# 	A_t_A = A_t_A + lamda_I
# 	#----------------------------

# 	# matrix inversion
# 	A_t_A_i = np.linalg.inv(A_t_A)
# 	# matrix multlipication 
# 	w =  np.dot(A_t_A_i, np.dot(A_t, Y_trn)) 
# 	return w
# -------------------------------------- END OF LINEAR REGRESSION CALCULATION OF w ----------------------------

# ------------------------------------- START OF ERROR CALCULATION ----------------------------

# def getResidualError(X, Y, w, n): 
# 	A = phiX(X, n)
# 	A_w = np.dot(A, w)
# 	res_error = Y - A_w

# 	return res_error

# def getError(X, Y, w, n): 
# 	res_error = getResidualError(X, Y, w, n)
# 	res_error_t = res_error.transpose()
# 	res_res_t = np.dot(res_error_t, res_error)

# 	res_res_t_Num = res_res_t[0][0]

# 	return  res_res_t_Num ** 0.5


#------------------------------------- END OF ERROR CALCULATION ------------------------------


#------------------------------------- RUN THE WHOLE PROGRAM -----------------------
def run(fileName): 
	data = dataToVectors(fileName)
	X_trn, Y_trn, X_tst, Y_tst  = data

	print "RESULTs for linear regression"

	print "____________ optimal w's ____________"
	print "n = 2"
	w2 =  getW(X_trn,Y_trn, 2)
	print "w = {}".format(np.transpose(w2))

	print "n = 5"
	w5 =  getW(X_trn,Y_trn, 5)
	print "w = {}".format(np.transpose(w5))

	print "n = 10"
	w10 =  getW(X_trn,Y_trn, 10)
	print "w = {}".format(np.transpose(w10))

	print "n = 20"
	w20 = getW(X_trn,Y_trn, 20)
	print "w = {}".format(np.transpose(w20))

	# ############################### ERRORS data ##################################:

	print "__________________ errors_________________" 
	print "n = 2, training:" 
	error_2_trn =  getError(X_trn, Y_trn, w2, 2)
	print error_2_trn

	print "n = 2, test:" 
	error_2_tst =  getError(X_tst, Y_tst, w2, 2)
	print error_2_tst

	print "============================"
	#-----------------------------
	print "n = 5, training:"
	error_5_trn = getError(X_trn, Y_trn, w5, 5)
	print error_5_trn

	print "n = 5, test:"
	error_5_tst = getError(X_tst, Y_tst, w5, 5)
	print error_5_tst

	print "============================"
	
	#----------------------------

	print "n = 10, training:"
	error_10_trn = getError(X_trn, Y_trn, w10, 10)
	print error_10_trn
	print "n = 10, test:"
	error_10_tst = getError(X_tst, Y_tst, w10, 10)
	print error_10_tst

	print "============================"

	#------------------------------
	print "n = 20, training:"
	error_20_trn = getError(X_trn, Y_trn, w20, 20)
	print error_20_trn

	print "n = 20, test:"
	error_20_tst = getError(X_tst, Y_tst, w20, 20)
	print error_20_tst
	print "============================"
	

	print "2, {}, {}".format(error_2_trn, error_2_tst)
	print "5, {}, {}".format(error_5_trn, error_5_tst)
	print "10, {}, {}".format(error_10_trn, error_10_tst)
	print "20, {}, {}".format(error_20_trn, error_20_tst)

run("HW1_Data/linear_regression.mat")
