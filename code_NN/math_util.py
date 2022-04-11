# Sean Fauteaux 00794289

## delete the `pass` statement in every function below and add in your own code. 


import numpy as np



# Various math functions, including a collection of activation functions used in NN.




class MyMath:

    def tanh(x):
        ''' tanh function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh of the corresponding element in array x
        '''
        return np.tanh(x)

    
    def tanh_de(x):
        ''' Derivative of the tanh function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh derivative of the corresponding element in array x
        '''
        # derivative of tanh = 1 - (tanh(x)^2), or -(tanh(x)^2) + 1
        left = MyMath.tanh(x) * MyMath.tanh(x) * -1
        return left + 1

    
    def logis(x):
        ''' Logistic function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic of 
                    the corresponding element in array x
        '''
        # logistic function = 1 / (1 + e^-x)
        return 1/(1 + np.exp(-1 * x))

    
    def logis_de(x):
        ''' Derivative of the logistic function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic derivative of 
                    the corresponding element in array x
        '''
        # derivative of logistic function = logis(x) * (1 - logis(x))
        sigX = MyMath.logis(x)
        return sigX * (1 - sigX)

    
    def iden(x):
        ''' Identity function
            Support vectorized operation
            
            x: an array type of real numbers
            return: the numpy array where every element is the same as
                    the corresponding element in array x
        '''
        return np.asarray(x)

    
    def iden_de(x):
        ''' The derivative of the identity function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array of all ones of the same shape of x.
        '''
        x = MyMath.iden(x)
        return np.ones(x.size)
        

    def relu(x):
        ''' The ReLU function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is the max of: zero vs. the corresponding element in x.
        '''
        x = MyMath.iden(x)
        with np.nditer(x, op_flags=['readwrite']) as it:
            for i in it:
                i[...] = max(0, i)
        return x

    
    def _relu_de_scaler(x):
        ''' The derivative of the ReLU function. Scaler version.
        
            x: a real number
            return: 1, if x > 0; 0, otherwise.
        '''
        if x > 0:
            return 1
        return 0

    
    def relu_de(x):
        ''' The derivative of the ReLU function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is the _relu_de_scaler of the corresponding element in x.   
        '''
        x = MyMath.iden(x)
        vec_relu_de = np.vectorize(MyMath._relu_de_scaler)
        return vec_relu_de(x)

