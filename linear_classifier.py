import numpy as np

class LinearClassifier(object):
    
    
    def __init__(self,
                 X,
                 y):
        self.X = X
        self.y = y
        
        
    def sigmoid(self, z):
        '''
        Calculates the sigmoid function
            
            Input:
            z = (X;w;b)
            
            Output:
            sgmd = n-dimensional vector with values between 0/1 for class labels
        '''
        sgmd = 1/(1+np.exp(-z))
        return sgmd
    
    def y_pred(self, w, b=0):
        '''
        Calculates P(Y|x;w) using the sigmoid function

            Input:
            X: nxd matrix
            w: d-dimensional vector
            b: scalar (optional, if not passed on is treated as 0)

            Output:
            prob: n-dimensional vector

        '''
        prob = self.sigmoid((np.dot(self.X,w.T)+b))
        return np.array(prob)
    
    def log_loss(self, w, b=0):
        '''
        Calculates the log loss, which is

            Input:
            X: nxd matrix
            y: n-dimensional vector with labels (+1 or -1)
            w: d-dimensional vector

            Output:
            a scalar
        '''
        assert np.sum(np.abs(self.y))==len(self.y) # check if all labels in y are either +1 or -1
    
        ls = -np.sum(np.log(self.sigmoid(self.y*(np.dot(self.X,w.T)+b))))
        return ls

    
    def gradient(self, w, b):
        '''
        Calculates the gradient with respect to w and b
        
            Input:
            X: nxd matrix
            y: n-dimensional vector with labels (+1 or -1)
            w: d-dimensional vector
            b: a scalar bias term

            Output:
            wgrad: d-dimensional vector with gradient
            bgrad: a scalar with gradient
        '''

        n, d = self.X.shape
        wgrad = np.zeros(d)
        bgrad = 0.0

        bgrad = np.sum((-self.y*self.sigmoid(-self.y*(np.dot(self.X,w.T)+b))))
        wgrad = np.dot(self.X.T,-self.y*self.sigmoid(-self.y*(np.dot(self.X,w.T)+b)))

        return wgrad, bgrad
    
    def logistic_regression(self, max_iter, alpha, threshold):
        n, d = self.X.shape
        w = np.zeros(d)
        b = 0.0   
        losses = []

        for step in range(max_iter):
            wgrad, bgrad = self.gradient(w,b)
            w_update = alpha * wgrad
            b_update = alpha * bgrad
            if np.all(np.abs(np.concatenate([np.array([b_update]),w_update])) <= threshold):
                  break
            w -= w_update
            b -= b_update
            losses.append(self.log_loss(w, b))

        return w, b, losses