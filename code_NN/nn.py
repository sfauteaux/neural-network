# Sean Fauteaux 00794289

# Implementation of the forwardfeed neural network using stochastic gradient descent via backpropagation
# Support parallel/batch mode: process every (mini)batch as a whole in one forward-feed/backtracking round trip.


import numpy as np
from code_NN.nn_layer import NeuralLayer
import code_NN.math_util as mu
import code_NN.nn_layer


class NeuralNetwork:
    
    def __init__(self):
        self.layers = []     # the list of L+1 layers, including the input layer. 
        self.L = -1          # Number of layers, excluding the input layer. 
                             # Initting it as -1 is to exclude the input layer in L. 

    def add_layer(self, d = 1, act = 'tanh'):
        ''' The newly added layer is always added AFTER all existing layers.
            The firstly added layer is the input layer.
            The most recently added layer is the output layer. 
            
            d: the number of nodes, excluding the bias node, which will always be added by the program. 
            act: the choice of activation function. The input layer will never use an activation function even if it is given. 
            
            So far, the set of supported activation functions are (new functions can be easily added into `math_util.py`): 
            - 'tanh': the tanh function
            - 'logis': the logistic function
            - 'iden': the identity function
            - 'relu': the ReLU function
        '''
        new_layer = NeuralLayer(d, act)
        self.layers.append(new_layer)
        self.L += 1

    def _init_weights(self):
        ''' Initialize every layer's edge weights with random numbers from [-1/sqrt(d),1/sqrt(d)], 
            where d is the number of nonbias node of the layer
        '''
        i = 1
        while i <= self.L:
            layer = self.layers[i]
            low = -1/(np.sqrt(layer.d))
            high = 1/(np.sqrt(layer.d))
            dim1 = self.layers[i-1].d + 1
            dim2 = layer.d
            layer.W = np.random.default_rng().uniform(low, high, size=(dim1, dim2))
            i += 1

    def forward_feed(self, X):
        x = np.insert(X, 0, 1, axis=1)  # bias column
        self.layers[0].X = x
        # forward feed, without saving S for each layer
        for i in range(1, self.L+1):
            S = self.layers[i-1].X @ self.layers[i].W
            self.layers[i].X = self.layers[i].act(np.insert(S, 0, 1, axis=1))
        return np.delete(self.layers[self.L].X, 0, axis=1)

    def fit(self, X, Y, eta = 0.01, iterations = 1000, SGD = True, mini_batch_size = 1):
        ''' Find the fitting weight matrices for every hidden layer and the output layer. Save them in the layers.
          
            X: n x d matrix of samples, where d >= 1 is the number of features in each training sample
            Y: n x k vector of lables, where k >= 1 is the number of classes in the multi-class classification
            eta: the learning rate used in gradient descent
            iterations: the maximum iterations used in gradient descent
            SGD: True - use SGD; False: use batch GD
            mini_batch_size: the size of each mini batch size, if SGD is True.  
        '''
        self._init_weights()  # initialize the edge weights matrices with random numbers.
        X = np.insert(X, 0, 1, axis=1)  # add bias column
        N = X.shape[0]

        # indices for the minibatches
        start = 0
        end = mini_batch_size

        # Stochastic Gradient Descent for NN using minibatches
        for k in range(iterations):
            # create mini batch using mini_batch_size number of elements
            # Check start and end indices to ensure they are within bounds
            if start >= N:
                start = 0
                end = mini_batch_size
            elif end > N:
                end = N

            # D is the minibatch of samples. Takes all the columns from the rows X[start] to X[end]
            D = X[start:end, :]
            Yp = Y[start:end, :]

            Np = D.shape[0]

            self.layers[0].X = D
            # forward feeding, while saving S(ell) and X(ell)
            for i in range(1, self.L+1):
                self.layers[i].S = self.layers[i-1].X @ self.layers[i].W
                self.layers[i].X = self.layers[i].act(np.insert(self.layers[i].S, 0, 1, axis=1))

            # initialize delta(L) and gradient G(L)
            out_layer = self.layers[self.L]
            out_layer.Delta = 2 * ((np.delete(out_layer.X, 0, axis=1) - Yp) * out_layer.act_de(out_layer.S))
            out_layer.G = np.einsum('ij, ik -> jk', self.layers[self.L - 1].X, out_layer.Delta) / Np

            # back propagation
            i = self.L - 1
            while i > 0:
                self.layers[i] = self.layers[i]
                self.layers[i].Delta = self.layers[i].act_de(self.layers[i].S) * (self.layers[i+1].Delta @ np.delete(self.layers[i+1].W, 0, axis=0).T)
                self.layers[i].G = np.einsum('ij, ik -> jk', self.layers[i-1].X, self.layers[i].Delta) / Np
                i -= 1

            # update weights
            for i in range(1, self.L+1):
                self.layers[i].W = self.layers[i].W - eta*self.layers[i].G

            # increment minibatch indices
            start = end
            end = start + mini_batch_size


    def predict(self, X):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column.
            
            return: n x 1 matrix, n is the number of samples, every row is the predicted class id.
         '''
        X = self.forward_feed(X)
        return np.argmax(X, axis=1)

    def error(self, X, Y):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column. 
               n is the number of samples. 
               d is the number of (non-bias) features of each sample. 
            Y: n x k matrix, the labels of the input n samples. Each row is the label of one sample, 
               where only one entry is 1 and the rest are all 0. 
               Y[i,j]=1 indicates the ith sample belongs to class j.
               k is the number of classes. 
            
            return: the percentage of misclassfied samples
        '''
        # get the indices of the predicted and actual class for each sample
        predict = self.predict(X)
        actual = np.argmax(Y, axis=1)
        N = X.shape[0]
        # get total number of misclassified samples, divided by N for the percentage
        wrong = np.sum(predict != actual)
        return wrong/N
