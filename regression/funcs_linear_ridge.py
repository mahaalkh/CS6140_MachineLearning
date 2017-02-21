import numpy as np 

################ LINEAR REGRESSION #################

def phiX(X, n):
	"""
	computes Phi(x)
	X_trn: the x_i's of the  data 
	n: the degree of the polynomial 
	:return: a matrix of size N * n 
		where N is the number of data points 
		and n is the degree of the polynomial  
	"""
	N = len(X)
	X_t = X.transpose()[0]

	phiX = []
	previousArray = []
	for i in range(n + 1): 
		if i == 0: 
			previousArray = np.array([1] * N, float).transpose()
		else: 
			previousArray = previousArray * X_t
		phiX.append(previousArray.tolist())

	phiX = np.array(phiX, float)

	return phiX.transpose()

def getW(X_trn, Y_trn, n, lamda = 0):
	"""
	computes w 
	X_trn: the x_i's of the training data 
	Y_trn: the y_i's of the training data 
	n: the degree of the polynomial 
	:return: a vector of size n * 1
	"""
	A = phiX(X_trn, n)
	
	A_t = A.transpose()

	# matrix multlipication 
	A_t_A = np.dot(A_t, A)

	#----------------------------
	m = len(A_t_A)
	I = np.identity(m) 
	lamda_I = lamda * I 
	A_t_A = A_t_A + lamda_I
	#----------------------------

	# matrix inversion
	A_t_A_i = np.linalg.inv(A_t_A)
	# matrix multlipication 
	w =  np.dot(A_t_A_i, np.dot(A_t, Y_trn)) 
	return w

######################## BOTH RIDGE AND LINEAR ########################
def getResidualError(X, Y, w, n): 
	A = phiX(X, n)
	A_w = np.dot(A, w)
	res_error = Y - A_w

	return res_error

def getError(X, Y, w, n): 
	res_error = getResidualError(X, Y, w, n)
	res_error_t = res_error.transpose()
	res_res_t = np.dot(res_error_t, res_error)

	res_res_t_Num = res_res_t[0][0]

	return  res_res_t_Num ** 0.5
