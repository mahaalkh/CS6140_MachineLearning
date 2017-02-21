import numpy as np
from utils import dataToVectors
import argparse

### for running with command line arguments
### python NeuralNetwork alpha lambda s2

################# Extract arguments ###########
parser = argparse.ArgumentParser()
parser.add_argument(dest = "alpha", default = 0.01, type = float)
parser.add_argument(dest = "lamda", default = 0.01, type = float)
parser.add_argument(dest = "s2", default = 10, type = int)
args = parser.parse_args()
args = vars(args)
alpha = args['alpha']
lamda = args['lamda']
s2 = args['s2']

######### functions to use as activation functions 

def softmax(z): 
    """
    the sofmax function ; used in the output layer 
    """
    exp_z = np.exp(z)                                                            
    a3 = exp_z / np.sum(exp_z, axis=1, keepdims=True)  
    return a3                   

def sigmoid(z): 
    """
    the sigmoid function ; used in the hidden layer
    """
    result = 1.0 / (1.0 + np.exp(-1 * z)) 
    return result

########### derivatives of the activation functions
def sigmoid_derivative(z): 
    """
    derivative with repect to z 
    """
    result = sigmoid(z) * (1 - sigmoid(z))
    return result          


# ================================================================

num_features = 3  									
num_classes = 3  										


def initialize_W_b(s2, range_w): 
    np.random.seed(0)
    W1 = np.random.randn(num_features, s2) * range_w                  
    b1 = np.zeros((1, s2))                                                                         
    W2 = np.random.randn(s2, num_classes) * range_w                               
    b2 = np.zeros((1, num_classes))                                                          

    # This is what we return at the end
    W_b = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}	
    return W_b					


def initialize_dW_db(s2): 
    dW1 = np.zeros((num_features, s2))                
    db1 = np.zeros((1, s2))                                                                         
    dW2 = np.zeros((s2, num_classes))                         
    db2 = np.zeros((1, num_classes))                                                          

    # This is what we return at the end
    dW_db = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}  
    return dW_db          									

def feedForward(W_b, X, single=True): 
    """
    calculates the needed value
    """
    W1, b1, W2, b2 = W_b['W1'], W_b['b1'], W_b['W2'], W_b['b2']
    z2 = X.dot(W1) + b1                                                               
    a2 = sigmoid(z2)                                                                  
    z3 = a2.dot(W2) + b2  
    a3 = softmax(z3)                                                            
   
    values = {'a3': a3, 'a2': a2, 'z2': z2, 'z3': z3}
    return  values


def gradientDescentStep(W_b, dW_db, values, N, X, y):
    """
    updates the weights and biases
    """
    a3, a2, z2 = values['a3'], values['a2'], values['z2']
    W1, b1, W2, b2 = W_b['W1'], W_b['b1'], W_b['W2'], W_b['b2']
    dW1, db1, dW2, db2 = dW_db['dW1'], dW_db['db1'], dW_db['dW2'], dW_db['db2']

    delta3 = a3                                                                 
    delta3[range(N), y] -= 1                                                
    dW2 = (a2.T).dot(delta3)                                                                
    db2 = np.sum(delta3, axis=0, keepdims=True)                                                     
    delta2 = delta3.dot(W2.T) * sigmoid_derivative(z2)                                  
    dW1 = np.dot(X.T, delta2)                                                                   
    db1 = np.sum(delta2, axis=0)                                                                        

        # Add regularization terms (b1 and b2 don't have regularization terms)
    dW2 += lamda * W2                                                        
    dW1 += lamda * W1                                                    

        # Gradient descent parameter update
    W1 += -alpha * dW1                                                           
    b1 += -alpha * db1                                                       
    W2 += -alpha * dW2                                                           
    b2 += -alpha * db2

    updated_W_b = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}  
    dW_db = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return updated_W_b, dW_db                                                         

def gradientDescent(X, y, s2, num_passes=20000, range_w = 0.01):
    N = len(X)
    W_b = initialize_W_b(s2, range_w)
    # print "initialized weights and baises"
    dW_db = initialize_dW_db(s2)
    # Gradient descent. For each batch...
    for i in range(0, num_passes):															
        # Forward propagation
        values = feedForward(W_b, X)
        W_b, dW_db = gradientDescentStep(W_b, dW_db, values, N, X, y)
      
    return W_b

                                           

def predict(W_b, x):                                       
    # Forward propagation
    values = feedForward(W_b, x)
    a3 = values['a3']
    return np.argmax(a3, axis=1)    

def performance(X, Y, W_b):
    assert len(X) == len(Y)
    err = 0.0 
    N = len(Y)
    for i in range(N): 
        y = Y[i]
        y_hat = predict(W_b, X[i])[0]
        if y != y_hat: 
            err += 1.0
    return err/N


def main():
    X_trn, Y_trn, X_tst, Y_tst = dataToVectors("HW2_Data/data.mat")
    W_b = gradientDescent(X_trn, Y_trn, s2)
    ## S2 lambda alpha trn_err tst_err
    print "{} \t {} \t {} \t {} \t {}".format(s2, lamda, alpha, performance(X_trn, Y_trn, W_b), performance(X_tst, Y_tst, W_b))

if __name__ == "__main__":
    main()