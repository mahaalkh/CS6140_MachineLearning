import numpy as np
import random
import utils_2 

########## 
"""
m is number of data, 
n is number of features
"""
keepC = 0 

def get_y(Y, i): 
	y = Y[i][0][0]
	if y == keepC: 
		y_i = +1 
	else: 
		y_i = -1
	return y_i  

def innerProduct(a, b): 
	assert np.shape(a) == np.shape(b)
	return np.dot(a, b.transpose())[0][0]


def randomly_select_j(i, m):
	"""
	"""
	while True:
		j = random.randrange(0, m)
		if j != i:
			return j 
#---------------------------- Functions -----------------------
# (2) 
def f_x(i, alphas, X, Y, bias): 
	"""
	i is the index of the x_i being used 
	alphas is a vector of size m * 1
	X is the list of the inputs 
	Y is the list of labels 
	bias is a number
	:return: number 
	"""
	assert len(X) == len(Y)
	x_i = X[i]
	total = 0
	for k in range(len(X)):
		y_k = get_y(Y, k)
		x_k = X[k]
		alpha_k = alphas[k]
		innerProduct_k = innerProduct(x_k, x_i)
		tot = alpha_k * y_k * innerProduct_k
		# print "alpha_k = {}, y_k = {}, innerProduct_k = {}".format(alpha_k, y_k, innerProduct_k)
		total = total + tot
	return bias + total 

#(10) or (11)
def LandH(i, j, alphas, Y, c): 
	"""
	i  is an index
	j is is an index
	"""
	y_i = get_y(Y, i)
	y_j = get_y(Y, j)

	alpha_i = alphas[i]
	alpha_j = alphas[j]

	if y_i == y_j: 
		L = max(0, alpha_i + alpha_j - c)
		H = min(c, alpha_i + alpha_j)
	else: 
		L = max(0, alpha_j - alpha_i)
		H = min(c, c + alpha_j - alpha_i)
	return [L, H]


#(12)
def update_alpha(i, j, alphas, X, Y, bias): 
	"""
	"""
	
	y_j = get_y(Y, j)
	alpha_j = alphas[j]
	E_i = E(i, alphas, X, Y, bias) 
	E_j = E(j, alphas, X, Y, bias) 
	num = y_j * (E_i - E_j)
	# print "i = {}, j = {}, y_i = {}, E_i = {}, E_j = {}".format(i, j, y_j, E_i, E_j)
	new_alpha_j = alpha_j - (num / eta(i, j, X))
	# print "num = {}, updated_alpha_j = {}, eta = {}".format(num, new_alpha_j, eta(i, j, X))
	return new_alpha_j

#(13)
def E(i, alphas, X, Y, bias):
	"""
	"""
	y_i = get_y(Y, i)
	# print "in E"
	# print y_i
	# print "f_x = {}".format(f_x(i, alphas, X, Y, bias))

	return f_x(i, alphas, X, Y, bias) - float(y_i)

#(14)
def eta(i, j, X): 
	x_i = X[i]
	x_j = X[j]
	result = 2.0 * innerProduct(x_i, x_j) - innerProduct(x_i, x_i) - innerProduct(x_j, x_j)
	return result

#(15)
def clip_alpha(j, alphas, L, H): 
	"""
	"""
	alpha_j = alphas[j]
	# print "L = {}, H = {}, alpha_j = {}".format(L, H, alpha_j)

	if alpha_j > H: 
		return H
	elif alpha_j >= L and alpha_j <= H: 

		return alpha_j
	elif alpha_j < L: 
		# print "in less than L"
		return L


#(16)
def solve_for_alpha(i, j, alphas, old_alphas, Y): 
	"""
	"""
	y_i = get_y(Y, i)
	y_j = get_y(Y, j)
	alpha_i = alphas[i]
	alpha_j = alphas[j]
	old_alpha_j = old_alphas[j]
	updated_alpha_i = alpha_i + (y_i * y_j * (old_alpha_j - alpha_j))
	return updated_alpha_i

#(17) #(18) #(19)
def calculate_b(i, j, alphas, old_alphas, Y, X, E_i, E_j, bias, c): 
	"""
	"""
	y_i = get_y(Y, i)
	y_j = get_y(Y, j)
	alpha_i = alphas[i]
	old_alpha_i = old_alphas[i]
	alpha_j = alphas[j]
	old_alpha_j = old_alphas[j]
	x_i = X[i]
	x_j = X[j]
	innerProduct_i_i = innerProduct(x_i, x_i)
	innerProduct_i_j = innerProduct(x_i, x_j)
	innerProduct_j_j = innerProduct(x_j, x_j)
	
	part1 = y_i * (alpha_i - old_alpha_i)
	part2 = y_j * (alpha_j - old_alpha_j)

	b_1 = bias - E_i - (part1 * innerProduct_i_i) - (part2 * innerProduct_i_j) 
	#### sudo code is same as this but python code is not
	# b_2 = bias - E_j - (part1 * innerProduct_i_i) - (part2 * innerProduct_j_j)
	b_2 = bias - E_j - (part1 * innerProduct_i_j) - (part2 * innerProduct_j_j)

	if alpha_i < c and alpha_i  > 0: 
		new_bias = b_1

	elif alpha_j < c and alpha_j > 0: 
		new_bias = b_2 

	else: 
		new_bias = (b_1 + b_2) / 2.0 

	return new_bias

#----------------------------------------------

def smo(c, tol, max_passes, X, Y):
	"""
	outputs the alphas and the bias
	"""
	assert len(X) == len(Y)
	m = len(X)
	alphas = [0] * m
	bias = 0
	passes = 0

	while (passes < max_passes): 
		num_changed_alphas = 0

		for i in range(m): 
			y_i = get_y(Y, i)
			alpha_i = alphas[i]
			E_i = E(i, alphas, X, Y, bias)
			if  ((((y_i * E_i) < -tol) and alpha_i < c) or (((y_i * E_i) > tol) and alpha_i > 0)):
				j = randomly_select_j(i, m)
				E_j = E(j, alphas, X, Y, bias)
				old_alphas = list(alphas) 
				old_alpha_j = old_alphas[j]
				L, H = LandH(i, j, alphas, Y, c)
				if L == H: 
					continue
				eta_i_j = eta(i, j, X)
				if eta_i_j >= 0: 
					continue
				alphas[j] = update_alpha(i, j, alphas, X, Y, bias)
				alphas[j] = clip_alpha(j, alphas, L, H)
				alpha_j = alphas[j]
				if (abs(alpha_j - old_alpha_j) < 10.0 ** -5): 
					continue
				alphas[i] = solve_for_alpha(i, j, alphas, old_alphas, Y)
				bias = calculate_b(i, j, alphas, old_alphas, Y, X, E_i, E_j, bias, c)
				num_changed_alphas = num_changed_alphas + 1

		if(num_changed_alphas == 0): 
			passes += 1
		else:
			passes = 0
	return [alphas, bias]


def trainDiff(c, tol, max_passes, x_trn, y_trn): 
	global keepC
	keepC = 0
	alphas0, bias0 = smo(c, tol, max_passes, x_trn, y_trn)
	print alphas0
	keepC = 1
	alphas1, bias1 = smo(c, tol, max_passes, x_trn, y_trn)
	print alphas1	
	keepC = 2
	alphas2, bias2 = smo(c, tol, max_passes, x_trn, y_trn)	
	print alphas2
	return [(alphas0, bias0), (alphas1, bias1), (alphas2, bias2)]


def calculate_w(alphas, X, Y, keepL): 
	"""
	"""
	assert len(X) == len(Y) and len(X) == len(alphas)
	global keepC
	keepC = keepL
	alphas_array = np.array(alphas)
	w = np.zeros(np.shape(X[0]))
	for l in range(len(X)):
		w = w + (alphas[l] * Y[l] * X[l]) 
	return w  

def classify(z, alphas_bias, w_s):
	"""
	"""
	[(alphas0, bias0), (alphas1, bias1), (alphas2, bias2)] =  alphas_bias
	[w0, w1, w2] = w_s

	num0 = np.dot(w0, z.transpose()) + bias0
	num1 = np.dot(w1, z.transpose()) + bias1
	num2 = np.dot(w2, z.transpose()) + bias2

	nums = [num0, num1, num2]
	return nums.index(max(nums))


def test(w_s, X, Y, alphas_bias): 
	"""
	"""
	assert len(X) == len(Y)
	num_missed = 0
	for l in range(len(X)): 
		z = X[l]
		predicted_y = classify(z, alphas_bias, w_s)
		y_l = Y[l][0][0]
		# print y_l, predicted_y
		if predicted_y != y_l: 
			num_missed = num_missed + 1.0

	percent_missed = num_missed / len(X)
	return percent_missed

def run(c, tol = 0.0001, max_passes = 100):
	"""
	"""
	data = utils_2.dataToVectors("HW2_Data/data.mat")
	x_trn, y_trn, x_tst, y_tst = data
	# alphas_bias = trainDiff(c, tol, max_passes, x_trn, y_trn)
	# [(alphas0, bias0), (alphas1, bias1), (alphas2, bias2)] =  alphas_bias
	# print alphas_bias
	# result from c = 1, tol = 0.0001, max_passes = 100
	# alphas_bias = [([0, 0, 0, 0, 0, 1.0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.76089874438604188, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.43655021488091217, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0.96888070961146233, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.22855715477666458, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], -2.8070285112324145), ([1.0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.71780210619033935, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 1.0, 0, 0, 0, 0, 1.0, 1.0, 0, 0, 0, 0, 0, 1.0, 0.16554535759560363, 0, 0.009388719617720984, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0.8927367392649509, 1.0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0], -0.64615392330664312), ([0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3434326855143166, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.92521186457220794, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0.035420167471281343, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2332243826152435, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0], -0.08211676223165)]
	# result from c = 1, tol = 0.0001, max_passes = 10
	# alphas_bias = [([0, 0, 0, 0, 0, 1.0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.19394008958698766, 0, 0, 0.98619386669174047, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0, 0, 0, 0, 1.0, 0, 5.5511151231257827e-17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.21584635843685984, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0.68813287879433582, 0, 0, 0, 0.98315648769872699, 0, 0, 0, 0, 0, 0, 0, 0, 0.36821831358786239, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.92477991776094026, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], -3.0204153760045371), ([1.0, 0, 1.0, 0, 0, 0, 0, 0.43581641515484582, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0.061718665054459734, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0.81277087482785526, 0, 0, 0, 0, 1.0, 1.0, 0, 0, 0, 0, 0, 1.0, 0.51448808693896275, 0, 0.43311433958687245, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.055037701695379526, 1.0, 0, 0, 0, 0, 0, 0, 0.33123784955792557, 1.0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0], -0.59529202619031418), ([0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0.24346795775914776, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.30033204572378303, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.45129510708865744, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.21841747061285377, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0.24422483474983483, 0, 0.36735543929274128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.60193230714186563, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0], -0.049520899172622)]
	[(alphas0, bias0), (alphas1, bias1), (alphas2, bias2)] =  alphas_bias
	w0 = calculate_w(alphas0, x_trn, y_trn, 0)
	w1 = calculate_w(alphas1, x_trn, y_trn, 1)
	w2 = calculate_w(alphas2, x_trn, y_trn, 2)
	w_s = [w0, w1, w2]
	print "training error = {}".format(test(w_s, x_trn, y_trn, alphas_bias))
	print "test error = {}".format(test(w_s, x_tst, y_tst, alphas_bias))




run(1.0, tol = 0.0001, max_passes = 10)


